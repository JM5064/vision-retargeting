"""Usage:
python -m models.train_main
"""

import random
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.transforms import v2
import albumentations as A

from models.model import SimpleBaselines
from models.train import train
from models.utils import load_checkpoint, DEVICE
from datasets.FreiHAND.freihand_dataset import FreiHAND

from losses.combined_loss import CombinedLoss


if __name__ == "__main__":
    random.seed(5064)
    torch.manual_seed(5064)
    np.random.seed(5064)

    train_transform = A.Compose([
        A.Rotate(limit=[-45, 45]),
        A.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2], hue=[-0.05, 0.05]),
        A.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
        A.ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    transform = A.Compose([
        A.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
        A.ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    # Full dataset
    train_images_dir = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/training/rgb'
    test_images_dir = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation/rgb'

    train_kpts_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/training_xyz.json'
    train_intrinsics_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/training_K.json'
    train_scale_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/training_scale.json'

    test_kpts_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_xyz.json'
    test_intrinsics_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_K.json'
    test_scale_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_scale.json'

    # 64 images for testing
    # train_images_dir = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/rgb'
    # test_images_dir = train_images_dir

    # train_kpts_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/training_xyz.json'
    # train_intrinsics_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/training_K.json'
    # train_scale_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/training_scale.json'

    # test_kpts_json = train_kpts_json
    # test_intrinsics_json = train_intrinsics_json
    # test_scale_json = train_scale_json

    # Train set as test set for testing
    # test_images_dir = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation/rgb'
    # train_images_dir = test_images_dir

    # test_kpts_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_xyz.json'
    # test_intrinsics_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_K.json'
    # test_scale_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_scale.json'

    # train_kpts_json = test_kpts_json
    # train_intrinsics_json = test_intrinsics_json
    # train_scale_json = test_scale_json


    train_dataset = FreiHAND(
        images_dir=train_images_dir, 
        keypoints_json=train_kpts_json, 
        intrinsics_json=train_intrinsics_json,
        scale_json=train_scale_json,
        transform=train_transform,
        percent=0.18
    )

    val_dataset = FreiHAND(
        images_dir=train_images_dir, 
        keypoints_json=train_kpts_json, 
        intrinsics_json=train_intrinsics_json,
        scale_json=train_scale_json,
        transform=train_transform,
        percent=-0.02
    )
    
    test_dataset = FreiHAND(
        images_dir=test_images_dir, 
        keypoints_json=test_kpts_json, 
        intrinsics_json=test_intrinsics_json,
        scale_json=train_scale_json,
        transform=transform,
    )

    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    num_keypoints = 21


    # --- INITIALIZE KEYPOINT MODEL ---

    model = SimpleBaselines(num_keypoints=num_keypoints)
    
    # Load in pretraining weights
    weights = torch.load("runs/mpii_sb.pt", map_location=torch.device('cpu'))
    filtered_weights = {
        k: v for k, v in weights.items()
        if k.startswith("backbone") 
        or k.startswith("deconv_layers")
    }
    model.load_state_dict(filtered_weights, strict=False)

    model = model.to(DEVICE)

    adamW_params = {
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    }
    optimizer = optim.AdamW(model.parameters(), **adamW_params)

    def convnext_scheduler(optimizer, num_warmup_epochs, total_epochs):
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs-num_warmup_epochs, eta_min=1e-5)

        if num_warmup_epochs == 0:
            return cosine_scheduler

        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1/num_warmup_epochs, total_iters=num_warmup_epochs)

        return optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[num_warmup_epochs])

    warmup_epochs = 10
    total_epochs = 50
    scheduler = convnext_scheduler(optimizer, warmup_epochs, total_epochs)

    # model, optimizer, scheduler, epoch = load_checkpoint("models/BlazePoseFreiHAND/runs/test/last.pt", model, optimizer, scheduler)

    train(
        model, 
        num_epochs=total_epochs, 
        start_epoch=0,
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader, 
        loss_func=CombinedLoss(), 
        optimizer=optimizer, 
        scheduler=scheduler,
        runs_dir='runs'
    )
