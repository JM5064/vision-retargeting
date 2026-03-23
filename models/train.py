import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
import time

import torch
from metrics.mpjpe import mpjpe_3D
from metrics.pck import pck_2D, pck_3D
from utils import to_device, log_results


def validate(model, val_loader, loss_func, image_size):
    model.eval()
    all_preds = []
    all_labels = []
    total_combined_loss = 0.0
    total_regression_loss = 0.0
    total_heatmap_loss = 0.0
    total_offset_loss = 0.0

    mpjpe = 0.0
    pck3D_thresholds = [20, 40]
    pck3Ds = np.zeros(len(pck3D_thresholds))

    with torch.no_grad():
        for inputs, keypoints, heatmaps, offset_masks, Ks, wrist_depths in tqdm(val_loader):
            inputs = to_device(inputs)
            keypoints = to_device(keypoints)
            heatmaps = to_device(heatmaps)
            offset_masks = to_device(offset_masks)
            Ks = to_device(Ks)
            wrist_depths = to_device(wrist_depths)

            # Get predictions for regression and heatmap paths
            regression_outputs, heatmap_outputs = model(inputs)
            loss, regression_loss, heatmap_loss, offset_loss = loss_func(
                regression_outputs, keypoints, heatmap_outputs, heatmaps, offset_masks
            )
            total_combined_loss += loss.item()
            total_regression_loss += regression_loss.item()
            total_heatmap_loss += heatmap_loss.item()
            total_offset_loss += offset_loss.item()

            all_preds.extend(regression_outputs.cpu().numpy().squeeze())
            all_labels.extend(keypoints.cpu().numpy())

            # Calculate mpjpe on batch
            batch_mpjpe = mpjpe_3D(regression_outputs, keypoints, Ks, wrist_depths, image_size)
            # Multiply by batch size to get total pjpe for the batch
            mpjpe += batch_mpjpe.item() * keypoints.shape[0]

            # Calculate 3D pcks on batch
            for i in range(len(pck3D_thresholds)):
                batch_pck = pck_3D(regression_outputs, keypoints, pck3D_thresholds[i], Ks, wrist_depths, image_size)
                pck3Ds[i] += batch_pck.item() * keypoints.shape[0]


    # Divide by # of images
    mpjpe /= len(all_preds)
    pck3Ds /= len(all_preds)

    average_val_loss = total_combined_loss / len(val_loader)
    average_val_regression_loss = total_regression_loss / len(val_loader)
    average_val_heatmap_loss = total_heatmap_loss / len(val_loader)
    average_val_offset_loss = total_offset_loss / len(val_loader)
    
    # Flatten
    all_preds_flattened = np.concatenate(all_preds, axis=0)
    all_labels_flattened = np.concatenate(all_labels, axis=0)

    preds_concat = torch.cat([torch.tensor(pred) for pred in all_preds_flattened])
    labels_concat = torch.cat([torch.tensor(label) for label in all_labels_flattened])

    mae = torch.mean(torch.abs(preds_concat - labels_concat)).item()

    # Reshape to [batch, num_keypoints, 3]
    preds_kp = preds_concat.view(-1, 21, 3)
    labels_kp = labels_concat.view(-1, 21, 3)

    # Calculate pck metrics
    # For FreiHAND: p1 = 9 (middle finger bottom), p2 = 12 (middle finger top) (not conventional)
    pck005 = pck_2D(preds_kp[..., :2], labels_kp[..., :2], 0.05, 9, 12).item()
    pck02 = pck_2D(preds_kp[..., :2], labels_kp[..., :2], 0.2, 9, 12).item()

    metrics = {
        "mae": mae,
        "pck@0.05": pck005,
        "pck@0.2": pck02,
        "pck@20mm": pck3Ds[0],
        "pck@40mm": pck3Ds[1],
        "mpjpe": mpjpe,
        "average_val_loss": average_val_loss,
        "average_val_regression_loss": average_val_regression_loss,
        "average_val_heatmap_loss": average_val_heatmap_loss,
        "average_val_offset_loss": average_val_offset_loss
    }

    return metrics


def train(
        model,
        ik_model,
        num_epochs,
        train_loader,
        val_loader,
        test_loader,
        loss_func,
        optimizer,
        scheduler,
        start_epoch=0,
        unfreeze_epoch=40,
        image_size=224,
        runs_dir="models/BlazePoseFreiHAND/runs",
    ):
    log_directory = runs_dir
    # create log file for a new training session
    if start_epoch == 0:
        time = str(datetime.now())
        os.mkdir(runs_dir + "/" + time)
        log_directory = runs_dir + "/" + time
    best_pck005 = 0

    # training loop
    for i in range(start_epoch, num_epochs):
        print(f'Epoch {i+1}/{num_epochs}')

        if i == unfreeze_epoch:
            print("Unfreezing initial layer(s)")
            for param in model.bb1.parameters():
                param.requires_grad = True

        total_combined_loss = 0.0
        total_regression_loss = 0.0
        total_heatmap_loss = 0.0
        total_offset_loss = 0.0

        model.train()
        for inputs, keypoints, heatmaps, offset_masks, Ks, wrist_depths in tqdm(train_loader):
            inputs = to_device(inputs)
            keypoints = to_device(keypoints)
            heatmaps = to_device(heatmaps)
            offset_masks = to_device(offset_masks)

            optimizer.zero_grad()

            # Get predictions for regression and heatmap paths
            regression_outputs, heatmap_outputs = model(inputs)

            # Calculate predicted qpos
            ik_model()

            # Calculate loss
            loss, regression_loss, heatmap_loss, offset_loss = loss_func(
                regression_outputs, keypoints, heatmap_outputs, heatmaps, offset_masks
            )
            loss.backward()
            optimizer.step()

            total_combined_loss += loss.item()
            total_regression_loss += regression_loss.item()
            total_heatmap_loss += heatmap_loss.item()
            total_offset_loss += offset_loss.item()

        # print and log metrics
        average_train_loss = total_combined_loss / len(train_loader)
        average_train_regression_loss = total_regression_loss / len(train_loader)
        average_train_heatmap_loss = total_heatmap_loss / len(train_loader)
        average_train_offset_loss = total_offset_loss / len(train_loader)

        metrics = validate(model, val_loader, loss_func, image_size)
        metrics["average_train_loss"] = average_train_loss
        metrics["average_train_regression_loss"] = average_train_regression_loss
        metrics["average_train_heatmap_loss"] = average_train_heatmap_loss
        metrics["average_train_offset_loss"] = average_train_offset_loss


        print(f'Epoch {i+1} Results:')
        print(f'Train Loss: {average_train_loss} | Regression: {average_train_regression_loss}'
             f' | Heatmap: {average_train_heatmap_loss} | Offset: {average_train_offset_loss}')
        print(f'Val Loss:   {metrics["average_val_loss"]} | Regression: {metrics["average_val_regression_loss"]}'
             f' | Heatmap: {metrics["average_val_heatmap_loss"]} | Offset: {metrics["average_val_offset_loss"]}')
        print(f'PCK@0.05: {metrics["pck@0.05"]}\tPCK@0.2: {metrics["pck@0.2"]}')
        print(f'PCK@20mm: {metrics["pck@20mm"]}\tPCK@40mm: {metrics["pck@40mm"]}')
        print(f'MPJPE: {metrics["mpjpe"]}')

        log_results(log_directory + "/metrics.csv", metrics)

        # Step scheduler
        if scheduler:
            scheduler.step()

        # save best model
        checkpoint = {
            'epoch': i + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }

        # Save best model best on pck@0.05
        pck005 = metrics['pck@0.05']
        if pck005 > best_pck005:
            torch.save(checkpoint, log_directory + "/best.pt")
            best_pck005 = pck005

        # Save last model
        torch.save(checkpoint, log_directory + "/last.pt")

        # Save a model every 20 epochs
        if (i+1) % 20 == 0:
            torch.save(checkpoint, f'{log_directory}/epoch{i+1}.pt')


    # test model and print/log testing metrics
    print("Testing Model")
    metrics = validate(model, test_loader, loss_func, image_size)
    print("Testing Results")
    print(f'PCK@0.05: {metrics["pck@0.05"]}\tPCK@0.2: {metrics["pck@0.2"]}')
    print(f'PCK@20mm: {metrics["pck@20mm"]}\tPCK@40mm: {metrics["pck@40mm"]}')
    print(f'MPJPE: {metrics["mpjpe"]}')
    print(f'Test Loss: {metrics["average_val_loss"]}')

    test_logfile_path = log_directory + "/test_metrics.csv"
    log_results(test_logfile_path, metrics)

