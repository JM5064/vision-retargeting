import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
import time

import torch
from metrics.mpjpe import mpjpe_3D
from metrics.pck import pck_2D, pck_3D
from models.utils import DEVICE, log_results

from datasets.FreiHAND.heatmap_inference import marginal_soft_argmax

from models.math_utils import xyZ2XYZ


def validate(model, val_loader, loss_func, image_size):
    model.eval()

    total_combined_loss = 0.0
    total_keypoint_loss = 0.0
    total_heatmap_loss = 0.0

    mpjpe = 0.0
    pck3D_thresholds = [20, 40]
    pck3Ds = np.zeros(len(pck3D_thresholds))

    total_abs_error = 0.0
    total_elements = 0

    pck005_total = 0.0
    pck02_total = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, keypoints, heatmaps, Ks, wrist_depths, scales in tqdm(val_loader):
            inputs = inputs.to(DEVICE)
            keypoints = keypoints.to(DEVICE)
            heatmaps = heatmaps.to(DEVICE)
            Ks = Ks.to(DEVICE)
            wrist_depths = wrist_depths.to(DEVICE)
            scales = scales.to(DEVICE)

            # Get predictions for heatmap path and depth
            heatmap_outputs = model(inputs)

            # Get keypoint predictions from heatmap
            keypoint_predictions = marginal_soft_argmax(heatmap_outputs)

            # Convert xyZ back to XYZ
            labels_XYZ = xyZ2XYZ(keypoints, image_size, Ks, wrist_depths, scales)
            pred_XYZ = xyZ2XYZ(keypoint_predictions, image_size, Ks, wrist_depths, scales)

            # Calculate losses
            loss, keypoint_loss, heatmap_loss = loss_func(
                keypoint_predictions, keypoints, heatmap_outputs, heatmaps
            )
            total_combined_loss += loss.item()
            total_keypoint_loss += keypoint_loss.item()
            total_heatmap_loss += heatmap_loss.item()

            batch_size = keypoints.shape[0]

            # Calculate MAE
            total_abs_error += torch.abs(keypoint_predictions - keypoints).sum().item()
            total_elements += keypoint_predictions.numel()

            # Calculate mpjpe
            batch_mpjpe = mpjpe_3D(pred_XYZ, labels_XYZ)
            # Multiply by batch size to get total pjpe for the batch
            mpjpe += batch_mpjpe.item() * batch_size

            # Calculate 3D pcks
            for i in range(len(pck3D_thresholds)):
                batch_pck = pck_3D(pred_XYZ, labels_XYZ, pck3D_thresholds[i])
                pck3Ds[i] += batch_pck.item() * batch_size

            # Calculate 2D pcks
            # For FreiHAND: p1 = 9 (middle finger bottom), p2 = 12 (middle finger top) (not conventional)
            pck005_total += pck_2D(keypoint_predictions[..., :2], keypoints[..., :2], 0.05, 9, 12).item() * batch_size
            pck02_total += pck_2D(keypoint_predictions[..., :2], keypoints[..., :2], 0.2, 9, 12).item() * batch_size
            total_samples += batch_size

    # Divide by # of images
    mae = total_abs_error / total_elements
    mpjpe /= total_samples
    pck3Ds /= total_samples
    pck005 = pck005_total / total_samples
    pck02 = pck02_total / total_samples

    average_val_loss = total_combined_loss / len(val_loader)
    average_val_keypoint_loss = total_keypoint_loss / len(val_loader)
    average_val_heatmap_loss = total_heatmap_loss / len(val_loader)
    
    metrics = {
        "mae": mae,
        "pck@0.05": pck005,
        "pck@0.2": pck02,
        "pck@20mm": pck3Ds[0],
        "pck@40mm": pck3Ds[1],
        "mpjpe": mpjpe,
        "average_val_loss": average_val_loss,
        "average_val_keypoint_loss": average_val_keypoint_loss,
        "average_val_heatmap_loss": average_val_heatmap_loss,
    }

    return metrics


def train(
        model,
        num_epochs,
        train_loader, val_loader, test_loader,
        loss_func, optimizer, scheduler,
        start_epoch=0, 
        image_size=224,
        runs_dir="runs",
    ):
    log_directory = runs_dir
    # create log file for a new training session
    if start_epoch == 0:
        time = str(datetime.now())
        os.mkdir(runs_dir + "/" + time)
        log_directory = runs_dir + "/" + time
    best_pck005 = 0

    # training loop
    steps_per_epoch = len(train_loader) // 8
    train_iterator = iter(train_loader)

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        total_combined_loss = 0.0
        total_keypoint_loss = 0.0
        total_heatmap_loss = 0.0

        model.train()
        for _ in tqdm(range(steps_per_epoch)):
            try:
                # Grab the next batch from the iterator
                inputs, keypoints, heatmaps, _, _, _ = next(train_iterator)
            except StopIteration:
                # Reset the iterator when finished one real epoch
                print("Finished loop of dataset")
                train_iterator = iter(train_loader)
                inputs, keypoints, heatmaps, _, _, _ = next(train_iterator)

            inputs = inputs.to(DEVICE)
            keypoints = keypoints.to(DEVICE)
            heatmaps = heatmaps.to(DEVICE)

            optimizer.zero_grad()

            # Get predictions for heatmap path and depth
            heatmap_outputs = model(inputs)

            # Get keypoint predictions from heatmap
            keypoint_predictions = marginal_soft_argmax(heatmap_outputs)

            # Calculate loss
            loss, keypoint_loss, heatmap_loss = loss_func(
                keypoint_predictions, keypoints, heatmap_outputs, heatmaps
            )
            loss.backward()
            optimizer.step()

            total_combined_loss += loss.item()
            total_keypoint_loss += keypoint_loss.item()
            total_heatmap_loss += heatmap_loss.item()

        # print and log metrics
        average_train_loss = total_combined_loss / steps_per_epoch
        average_train_keypoint_loss = total_keypoint_loss / steps_per_epoch
        average_train_heatmap_loss = total_heatmap_loss / steps_per_epoch

        metrics = validate(model, val_loader, loss_func, image_size)
        metrics["average_train_loss"] = average_train_loss
        metrics["average_train_keypoint_loss"] = average_train_keypoint_loss
        metrics["average_train_heatmap_loss"] = average_train_heatmap_loss

        print(f'Epoch {epoch+1} Results:')
        print(f'Train Loss: {average_train_loss}  |  Keypoint: {average_train_keypoint_loss}  |  Heatmap: {average_train_heatmap_loss}')
        print(f'Val Loss:   {metrics["average_val_loss"]}  |  Keypoint: {metrics["average_train_keypoint_loss"]}  |  Heatmap: {metrics["average_train_heatmap_loss"]}')
        print(f'PCK@0.05: {metrics["pck@0.05"]}\tPCK@0.2: {metrics["pck@0.2"]}')
        print(f'PCK@20mm: {metrics["pck@20mm"]}\tPCK@40mm: {metrics["pck@40mm"]}')
        print(f'MPJPE: {metrics["mpjpe"]}')

        log_results(log_directory + "/metrics.csv", metrics)

        # Step scheduler
        if scheduler:
            scheduler.step()

        # save best model
        checkpoint = {
            'epoch': epoch + 1,
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
        if (epoch+1) % 20 == 0:
            torch.save(checkpoint, f'{log_directory}/epoch{epoch+1}.pt')


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

