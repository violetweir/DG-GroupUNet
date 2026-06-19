import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from medpy.metric.binary import hd95
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from networks.swinunet_network import SwinUNet
from utils.dataloader_polyp import get_loader


def validate_img_size(img_size):
    if img_size % 224 != 0:
        raise ValueError(
            "SwinUNet requires --img_size to be a multiple of 224 with the default "
            "patch_size=4 and window_size=7. Use --img_size 224 or --img_size 448. "
            f"Got {img_size}."
        )


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


def get_binary_metrics(pred, gt):
    tp = (pred * gt).sum().item()
    tn = ((1 - pred) * (1 - gt)).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)

    try:
        if pred.sum() > 0 and gt.sum() > 0:
            hd_val = hd95(pred.cpu().numpy(), gt.cpu().numpy())
        else:
            hd_val = 100.0
    except Exception:
        hd_val = 100.0

    return sensitivity, specificity, precision, hd_val


def test(model, path, dataset, opt, device, save_base=None):
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
    detailed_results = []

    with torch.no_grad():
        for pack in tqdm(test_loader, desc=f"Inference on {dataset}"):
            images, gts, original_shapes, names = pack
            images = images.to(device)
            gts = gts.to(device).float()
            predictions = model(images)

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

                d = dice_coefficient(input_binary, target_binary).item()
                io = iou(input_binary, target_binary).item()
                sens, spec, prec, hd = get_binary_metrics(input_binary, target_binary)

                dsc += d
                iou_sum += io
                total_images += 1

                detailed_results.append(
                    {
                        "Name": names[i],
                        "Dice": d,
                        "IoU": io,
                        "Sensitivity": float(f"{sens:.4f}"),
                        "Specificity": float(f"{spec:.4f}"),
                        "Precision": float(f"{prec:.4f}"),
                        "HD95": float(f"{hd:.4f}"),
                    }
                )

                if save_base:
                    pred_img = (input_binary.cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_base, names[i]), pred_img)

    return dsc / total_images, iou_sum / total_images, detailed_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="ID of the run to test")
    parser.add_argument("--dataset_name", type=str, default="ClinicDB")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--test_batchsize", type=int, default=1)
    parser.add_argument("--color_image", default=True)
    parser.add_argument("--test_path", type=str, default="./data/polyp/target/")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validate_img_size(opt.img_size)
    save_base = f"./predictions_polyp/{opt.run_id}/{opt.dataset_name}/{opt.split}"
    os.makedirs(save_base, exist_ok=True)
    os.makedirs("results_polyp", exist_ok=True)

    model_path = os.path.join(f"./model_pth/{opt.run_id}/", f"{opt.run_id}-best.pth")
    opt.test_path = f"{opt.test_path}/{opt.dataset_name}/"

    model = SwinUNet(num_classes=1, in_channels=3, img_size=opt.img_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    mean_dice, mean_iou, results = test(model, opt.test_path, opt.split, opt, device, save_base=save_base)

    df = pd.DataFrame(results)
    mean_row = df.mean(numeric_only=True).to_dict()
    mean_row["Name"] = "AVERAGE"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    excel_name = f"results_polyp/Results_{opt.run_id}_{opt.dataset_name}_{opt.split}.xlsx"
    df.to_excel(excel_name, index=False)

    print(f"\nFinal Results for {opt.run_id}:")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Excel report saved to: {excel_name}")

    summary_file = "All_Runs_Summary_Polyp.xlsx"
    avg_data = {
        "run_id": opt.run_id,
        "network": "SwinUNet",
        "dataset": opt.dataset_name,
        "split": opt.split,
        "dice": mean_dice,
        "iou": mean_iou,
        "sensitivity": mean_row["Sensitivity"],
        "specificity": mean_row["Specificity"],
        "precision": mean_row["Precision"],
        "HD95": mean_row["HD95"],
    }
    df_summary_new = pd.DataFrame([avg_data])

    if os.path.exists(summary_file):
        df_summary_existing = pd.read_excel(summary_file)
        df_summary_combined = pd.concat([df_summary_existing, df_summary_new], ignore_index=True)
        df_summary_combined.to_excel(summary_file, index=False)
    else:
        df_summary_new.to_excel(summary_file, index=False)

    print(f"Summary appended to {summary_file}")
