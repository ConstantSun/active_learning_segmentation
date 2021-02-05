"""
Calculate ECE (Measure calibration only on the maximum
      prediction for each datapoint)
"""
import time
import numpy as np
import uncertainty_metrics.numpy as um
import torch


def cal_ece(probs: np.array, labels: np.array, num_bins=10):
    """
    probs, shape (n,)  : output probability (the model generates segmentation map, and it is flattened after that)
    labels, shape (n,) : corresponding class label (the ground truth map)
    """
    confidences = np.where(probs >= 0.5, probs, 1 - probs)
    # print("confident")
    # print(confidences)
    # print("prob: ", probs)
    preds = (probs >= 0.5).astype(np.float32)
    # print("preds: ", preds)
    # labels = np.random.randint(0, 2, 10)
    # labels = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    # print("label: ", labels)
    bins = np.arange(0, 1, 1 / num_bins, dtype=np.float32)[1:]
    bins = np.append(bins, 1)
    # print("bins: ", bins)
    index = np.digitize(confidences, bins)
    # print("index: ", index)

    calibration_error = []
    true_positive = (preds == labels).astype(np.int)
    # print(true_positive)

    for i in range(num_bins):
        # print("bins", i)
        indexes = np.where(index == i)
        # print("***************len: ", len(indexes), indexes[0])
        if len(indexes[0]) == 0:
            continue
        # print("index for element", indexes)
        bins_confidence = np.mean(confidences[indexes])
        # print(bins_confidence)
        bins_accuracy = np.mean(true_positive[indexes])
        # print("accuracy")
        # print(bins_accuracy)
        ece_each_bin = np.abs(bins_accuracy - bins_confidence) * len(indexes[0])/len(probs)
        # print("ece_each_bin")
        # print(ece_each_bin)
        calibration_error.append(ece_each_bin)
    # print("calibration list ", calibration_error)
    # print("ece overall: ", np.array(calibration_error).sum())
    res = sum(calibration_error)
    res = np.log2(res)
    return res


def get_segmentation_mask_uncertainty(gened_mask: torch.tensor, gt_mask: torch.tensor):
    """
    gened_mask, Shape: batchx(channel)xHxW - ex: 4x(1)xHxW : generated mask, by taking average of output masks when using MC dropout.
    gt_mask, Shape: batchx(channel)xHxW - ex: 4x(1)xHxW    : ground truth mask.
    return: a list of ece value of n=batch images.

    """
    # flattening mask
    # print("gened mask shape: ", gened_mask.shape)
    # print("gt mask shape : ", gt_mask.shape)
    batch = gened_mask.shape[0]
    flattened_gened_mask = gened_mask.reshape(batch, -1)
    flattened_gt_mask = gt_mask.reshape(batch, -1)
    ece_list = []
    for i in range(batch):
        ece_list.append(cal_ece(flattened_gened_mask[i].cpu().detach().numpy(),
                                flattened_gt_mask[i].cpu().detach().numpy(), num_bins=10))
    return ece_list


# Ngoài ra khi visualize giữa ảnh trước và sau segment thì còn dùng cả CROSS ENTROPY LOSS nữa.
# Vẽ đồ thị minh họa ece của tất cả các ảnh training.
def get_uncertainty_map(avg_probs_maps: torch.tensor):
    """
    avg_probs_map: torch.tensor, shape: batchxCxHxW
    """
    seg_uncertainty_map = 2*avg_probs_maps*(1-avg_probs_maps)

