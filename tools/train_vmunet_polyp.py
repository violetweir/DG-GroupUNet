import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from networks.vmunet_network import UltraLight_VM_UNet
from utils.dataloader_polyp import get_loader
from utils.utils import AvgMeter, cal_params_flops, clip_gradient


def structure_loss(pred, mask, w=1):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (w * (wbce + wiou)).mean()


def dice_coefficient(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    total = predicted_flat.sum() + labels_flat.sum()
    return (2.0 * intersection + smooth) / (total + smooth)


def iou(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    union = predicted_flat.sum() + labels_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def primary_output(output):
    return output[0] if isinstance(output, list) else output


def evaluate(model, path, dataset, opt, device):
    data_path = os.path.join(path, dataset)
    image_root = f"{data_path}/images/"
    gt_root = f"{data_path}/masks/"
    model.eval()

    test_loader = get_loader(
        image_root=image_root,
        gt_root=gt_root,
        batchsize=opt.test_batchsize,
        trainsize=opt.img_size,
        shuffle=False,
        split="test",
        color_image=opt.color_image,
    )

    dsc, iou_sum, total_images = 0.0, 0.0, 0
    with torch.no_grad():
        for pack in test_loader:
            images, gts, original_shapes, _ = pack
            images = images.to(device)
            gts = gts.to(device).float()
            predictions = primary_output(model(images))

            for i in range(len(images)):
                h_orig, w_orig = int(original_shapes[0][i]), int(original_shapes[1][i])
                p = predictions[i].unsqueeze(0)
                pred_resized = F.interpolate(p, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
                pred_resized = pred_resized.sigmoid().squeeze()
                pred_resized = (pred_resized - pred_resized.min()) / (pred_resized.max() - pred_resized.min() + 1e-8)

                g = gts[i].unsqueeze(0)
                gt_resized = F.interpolate(g, size=(h_orig, w_orig), mode="nearest").squeeze()
                input_binary = (pred_resized >= 0.5).float()
                target_binary = (gt_resized >= 0.2).float()

                total_images += 1
                dsc += dice_coefficient(input_binary, target_binary).item()
                iou_sum += iou(input_binary, target_binary).item()

    return dsc / total_images, iou_sum / total_images, total_images


def train_one_epoch(train_loader, model, optimizer, epoch, opt, model_name, device):
    model.train()
    global best, test_dice_at_best_val, total_train_time, dict_plot

    epoch_start = time.time()
    loss_record = AvgMeter()
    size_rates = [1] if opt.no_multiscale else [0.75, 1, 1.25]
    total_step = len(train_loader)

    for i, (images, gts) in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images = Variable(images).to(device)
            gts = Variable(gts).float().to(device)

            if rate != 1:
                trainsize = int(round(opt.img_size * rate / 32) * 32)
                images = F.interpolate(images, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode="nearest")

            pred = primary_output(model(images))
            loss = structure_loss(pred, gts)
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        if i % 100 == 0 or i == total_step:
            print(
                f"{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], "
                f"Step [{i:04d}/{total_step:04d}], "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}, Loss: {loss_record.show():.4f}"
            )

    total_train_time += time.time() - epoch_start

    save_path = opt.train_save
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-last.pth"))

    epoch_results = {}
    for ds in ["test", "val"]:
        d_dice, d_iou, _ = evaluate(model, opt.test_path, ds, opt, device)
        epoch_results[ds] = d_dice
        logging.info(f"Epoch: {epoch}, Dataset: {ds}, Dice: {d_dice:.4f}, IoU: {d_iou:.4f}")
        print(f"Epoch: {epoch}, Dataset: {ds}, Dice: {d_dice:.4f}, IoU: {d_iou:.4f}")
        dict_plot[ds].append(d_dice)

    if epoch_results["val"] > best:
        logging.info(f"### Best Model Saved (Dice improved from {best:.4f} to {epoch_results['val']:.4f}) ###")
        print(f"### Best Model Saved (Dice improved from {best:.4f} to {epoch_results['val']:.4f}) ###")
        best = epoch_results["val"]
        test_dice_at_best_val = epoch_results["test"]
        torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-best.pth"))


if __name__ == "__main__":
    dataset_name = "ClinicDB"

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--test_batchsize", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=352)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--color_image", default=True)
    parser.add_argument("--augmentation", default=True)
    parser.add_argument("--no_multiscale", action="store_true")
    parser.add_argument("--train_path", type=str, default=f"./data/polyp/target/{dataset_name}/train/")
    parser.add_argument("--test_path", type=str, default=f"./data/polyp/target/{dataset_name}/")
    parser.add_argument("--train_save", type=str, default="")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "UltraLight_VM_UNet"

    for run in [1, 2, 3, 4, 5]:
        dict_plot = {"val": [], "test": []}
        best = 0.0
        test_dice_at_best_val = 0.0
        total_train_time = 0

        timestamp = time.strftime("%H%M%S")
        run_id = (
            f"{dataset_name}_{model_name}_bs{opt.batchsize}_lr{opt.lr}_"
            f"e{opt.epoch}_aug{opt.augmentation}_run{run}_t{timestamp}"
        )
        opt.train_save = f"./model_pth/{run_id}/"
        os.makedirs(opt.train_save, exist_ok=True)

        log_path = os.path.join(opt.train_save, f"train_log_{run_id}.log")
        logging.basicConfig(filename=log_path, level=logging.INFO, format="[%(asctime)s] %(message)s", force=True)

        model = UltraLight_VM_UNet(num_classes=1, in_channels=3).to(device)

        print(f"Network: {model_name}")
        print(f"Run directory: {opt.train_save}")
        print(f"Log file: {log_path}")
        print(f"Multiscale: {not opt.no_multiscale}")
        cal_params_flops(model, opt.img_size, logging)

        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-6)

        train_loader = get_loader(
            image_root=f"{opt.train_path}/images/",
            gt_root=f"{opt.train_path}/masks/",
            batchsize=opt.batchsize,
            trainsize=opt.img_size,
            shuffle=True,
            augmentation=opt.augmentation,
            split="train",
            color_image=opt.color_image,
        )

        for epoch in range(1, opt.epoch + 1):
            train_one_epoch(train_loader, model, optimizer, epoch, opt, run_id, device)
            scheduler.step()

        summary = (
            f"\n{'=' * 40}\nFINAL RESULTS: {run_id}\n"
            f"Best Val Dice: {best:.4f}\n"
            f"Test Dice at Best Val: {test_dice_at_best_val:.4f}\n"
            f"Total Train Time: {total_train_time:.2f}s\n{'=' * 40}"
        )
        print(summary)
        logging.info(summary)
