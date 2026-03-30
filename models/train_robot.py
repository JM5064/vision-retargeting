import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
import time

import torch
import pytorch_kinematics as pk
from metrics.mpjpe import mpjpe_3D
from metrics.pck import pck_2D, pck_3D
from models.utils import DEVICE, log_results

from datasets.FreiHAND.heatmap_inference import heatmap_inference

from models.math_utils import xyZ2XYZ, get_positions
from losses.pinch_loss import PinchLoss
from losses.hand_pose_loss import HandPoseLoss
from losses.hand_shape_loss import HandShapeLoss


def validate(model, val_loader, loss_func, image_size):
    model.eval()
    all_preds = []
    all_labels = []
    total_combined_loss = 0.0
    total_heatmap_loss = 0.0
    total_depth_loss = 0.0

    mpjpe = 0.0
    pck3D_thresholds = [20, 40]
    pck3Ds = np.zeros(len(pck3D_thresholds))

    with torch.no_grad():
        for inputs, keypoints, heatmaps, Ks, wrist_depths, scales in tqdm(val_loader):
            inputs = inputs.to(DEVICE)
            keypoints = keypoints.to(DEVICE)
            heatmaps = heatmaps.to(DEVICE)
            Ks = Ks.to(DEVICE)
            wrist_depths = wrist_depths.to(DEVICE)
            scales = scales.to(DEVICE)

            # Get predictions for heatmap path and depth
            heatmap_outputs, depth_outputs = model(inputs)

            # Get keypoint predictions from heatmap
            keypoint_predictions = heatmap_inference(heatmap_outputs)

            # Combine xy and depth
            keypoint_predictions = torch.cat([keypoint_predictions, depth_outputs.unsqueeze(-1)], dim=-1)

            # Convert xyZ back to XYZ
            labels_XYZ = xyZ2XYZ(keypoints, image_size, Ks, wrist_depths, scales)
            pred_XYZ = xyZ2XYZ(keypoint_predictions, image_size, Ks, wrist_depths, scales)

            # Calculate losses
            loss, heatmap_loss, depth_loss = loss_func(
                heatmap_outputs, heatmaps, depth_outputs, keypoints[:, :, 2]
            )
            total_combined_loss += loss.item()
            total_heatmap_loss += heatmap_loss.item()
            total_depth_loss += depth_loss.item()

            all_preds.extend(keypoint_predictions.cpu().numpy().squeeze())
            all_labels.extend(keypoints.cpu().numpy())

            # Calculate mpjpe on batch
            batch_mpjpe = mpjpe_3D(pred_XYZ, labels_XYZ)
            # Multiply by batch size to get total pjpe for the batch
            mpjpe += batch_mpjpe.item() * keypoints.shape[0]

            # Calculate 3D pcks on batch
            for i in range(len(pck3D_thresholds)):
                batch_pck = pck_3D(pred_XYZ, labels_XYZ, pck3D_thresholds[i])
                pck3Ds[i] += batch_pck.item() * keypoints.shape[0]


    # Divide by # of images
    mpjpe /= len(all_preds)
    pck3Ds /= len(all_preds)

    average_val_loss = total_combined_loss / len(val_loader)
    average_val_heatmap_loss = total_heatmap_loss / len(val_loader)
    average_val_depth_loss = total_depth_loss / len(val_loader)
    
    # Flatten
    all_preds_flattened = np.concatenate(all_preds, axis=0)
    all_labels_flattened = np.concatenate(all_labels, axis=0)

    preds_concat = torch.cat([torch.tensor(pred) for pred in all_preds_flattened])
    labels_concat = torch.cat([torch.tensor(label) for label in all_labels_flattened])

    mae = torch.mean(torch.abs(preds_concat - labels_concat)).item()

    # Reshape to [batch, num_keypoints, 3]
    preds_kp = preds_concat.view(-1, 21, 3)
    labels_kp = labels_concat.view(-1, 21, 3)

    # Calculate 2D pck metrics
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
        "average_val_depth_loss": average_val_depth_loss,
        "average_val_heatmap_loss": average_val_heatmap_loss,
    }

    return metrics


def train(
        model,
        ik_model, fk_model,
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
    steps_per_epoch = len(train_loader)
    train_iterator = iter(train_loader)

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        total_combined_loss = 0.0
        total_heatmap_loss = 0.0
        total_depth_loss = 0.0

        model.train()
        for _ in tqdm(range(steps_per_epoch)):
            try:
                # Grab the next batch from the iterator
                inputs, keypoints, heatmaps, Ks, wrist_depths, scales = next(train_iterator)
            except StopIteration:
                # Reset the iterator when finished one real epoch
                print("Finished loop of dataset")
                train_iterator = iter(train_loader)
                inputs, keypoints, heatmaps, Ks, wrist_depths, scales = next(train_iterator)

            inputs = inputs.to(DEVICE)
            keypoints = keypoints.to(DEVICE)
            heatmaps = heatmaps.to(DEVICE)
            Ks = Ks.to(DEVICE)
            wrist_depths = wrist_depths.to(DEVICE)
            scales = scales.to(DEVICE)

            optimizer.zero_grad()

            # Get predictions for heatmap path and depth
            heatmap_outputs, depth_outputs = model(inputs)

            # Get keypoint predictions from heatmap
            keypoint_predictions = heatmap_inference(heatmap_outputs)

            # Combine xy and depth
            keypoint_predictions = torch.cat([keypoint_predictions, depth_outputs.unsqueeze(-1)], dim=-1)

            # Convert xyZ back to XYZ
            XYZ_GT = xyZ2XYZ(keypoints, image_size, Ks, wrist_depths, scales)
            XYZ_pred = xyZ2XYZ(keypoint_predictions, image_size, Ks, wrist_depths, scales)

            # Subtract roots
            XYZ_GT = XYZ_GT - XYZ_GT[:, 0:1, :]
            XYZ_pred = XYZ_pred - XYZ_pred[:, 0:1, :]

            # Calculate predicted and GT qpos
            gt_qpos = ik_model.forward_batch(XYZ_GT)
            pred_qpos = ik_model.forward_batch(XYZ_pred)

            # Calculate FK
            gt_pos = get_positions(fk_model.forward_kinematics(gt_qpos))
            pred_pos = get_positions(fk_model.forward_kinematics(pred_qpos))


            pinch_loss = PinchLoss().forward(pred_pos, XYZ_GT)
            hand_pose_loss = HandPoseLoss().forward(pred_pos, gt_pos)
            hand_shape_loss = HandShapeLoss().forward(pred_pos, gt_pos)

            print("Pinch loss:", pinch_loss)
            print("Hand pose loss:", hand_pose_loss)
            print("Hand shape loss:", hand_shape_loss)

            # Calculate loss
            loss, heatmap_loss, depth_loss = loss_func(
                heatmap_outputs, heatmaps, depth_outputs, keypoints[:, :, 2]
            )
            loss.backward()
            optimizer.step()

            total_combined_loss += loss.item()
            total_heatmap_loss += heatmap_loss.item()
            total_depth_loss += depth_loss.item()

        # print and log metrics
        average_train_loss = total_combined_loss / steps_per_epoch
        average_train_heatmap_loss = total_heatmap_loss / steps_per_epoch
        average_train_depth_loss = total_depth_loss / steps_per_epoch

        metrics = validate(model, val_loader, loss_func, image_size)
        metrics["average_train_loss"] = average_train_loss
        metrics["average_train_depth_loss"] = average_train_depth_loss
        metrics["average_train_heatmap_loss"] = average_train_heatmap_loss


        print(f'Epoch {epoch+1} Results:')
        print(f'Train Loss: {average_train_loss} | Heatmap: {average_train_heatmap_loss}'
            f' | Depth: {average_train_depth_loss}')
        print(f'Val Loss:   {metrics["average_val_loss"]} | Heatmap: {metrics["average_val_heatmap_loss"]}'
            f' | Depth: {metrics["average_val_depth_loss"]}')
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

