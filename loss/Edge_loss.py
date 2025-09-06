from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss


def edgeSCE_loss(pred, target, edge, weight=None, p=None):
    BinaryCrossEntropy_fn = SoftBCEWithLogitsLoss(smooth_factor=None, reduction='None')
    if weight is not None:
        edge_weight = weight
    else:
        edge_weight = 4.
    loss_sce = BinaryCrossEntropy_fn(pred, target)
    edge = edge.clone()
    edge[edge == 0] = 1.
    edge[edge > 0] = edge_weight
    loss_sce = loss_sce * edge
    if p is not None:
        p = p
    else:
        p = 0.5
    loss_sce_, ind = loss_sce.contiguous().view(-1).sort()
    min_value = loss_sce_[int(p * loss_sce.numel())]
    loss_sce = loss_sce[loss_sce >= min_value]
    loss_sce = loss_sce.mean()
    loss = loss_sce
    return loss
