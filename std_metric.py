import torch


def get_segmentation_mask_uncertainty(gened_std_mask: torch.tensor):
    """
    gened_std_mask, Shape: batchx(channel)xHxW - ex: 4x(1)xHxW : generated std mask, by taking std of output masks when using MC dropout.
    return: a list of sum of std value of a image in n=batch images.

    """
    # flattening mask
    # print("gened mask shape: ", gened_mask.shape)
    # print("gt mask shape : ", gt_mask.shape)
    batch = gened_std_mask.shape[0]
    flattened_gened_mask = gened_std_mask.reshape(batch, -1)
    return gened_std_mask.sum(dim=1).tolist()[0]

