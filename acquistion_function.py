import torch


def get_entropy_in_a_prob_mask(mask: torch.tensor):
    """
    mask: batch x (c) x h x w.
    return shape: batch.
    """
    batch = mask.shape[0]
    flattend = mask.reshape(batch, -1)
    res = flattend*torch.log(flattend) + (1-flattend)*torch.log(1-flattend)
    res = res.sum(dim=1)
    return res


def category_first_entropy(GAUSS_ITERATION, net, imgs):
    """
    input:
    imgs: batch of imgs, shape: batchx(c)xhxw.
    output:
    category first entropy of that batch, shape: batch.

    This query function is calculating the entropy between all the classes of one
    pixel first, then average it with multiple models.
    """
    entropy_list = []
    for i in range(GAUSS_ITERATION):
        with torch.no_grad():
            logits = net(imgs)
            y_pred = torch.sigmoid(logits)
            y_pred = y_pred[:, :1, :, :] # batch x 1 x hxw
            entropy_list.append(get_entropy_in_a_prob_mask(y_pred)) # shape: ind, batch
    res = torch.cuda.FloatTensor(entropy_list)
    res = res.mean(dim=1)
    return res


def mean_first_entropy(GAUSS_ITERATION, net, imgs):
    """
    input:
    imgs: batch of imgs, shape: batchx(c)xhxw.
    output:
    mean_first_entropy of that batch, shape: batch.

    This query function is extracting mean of probability from multiple models
    first, then calculating the entropy based on the output.
    """
    y_pred_samples = []
    for i in range(GAUSS_ITERATION):
        with torch.no_grad():
            logits = net(imgs)
            y_pred = torch.sigmoid(logits)
            # y_pred = (y_pred > 0.5).float()
            y_pred = y_pred[:, :1, :, :]
            y_pred_samples.append(y_pred[:, 0, :, :])  # y_pred_samples's shape: (inx, bat, H, W )
    y_pred_samples = torch.stack(y_pred_samples, dim=0)
    y_pred_samples = y_pred_samples.type(torch.cuda.FloatTensor)
    mean_y_pred = y_pred_samples.mean(dim=0)  # shape: batch, H, W
    res = get_entropy_in_a_prob_mask(mean_y_pred)
    return res


def mutual_information(GAUSS_ITERATION, net, imgs):
    """
    This query function calculates the difference of two entropy calculated above.
    Hmean − Hcato
    """
    res = mean_first_entropy(GAUSS_ITERATION, net, imgs) - category_first_entropy(GAUSS_ITERATION, net, imgs)
    return res
