import torch


def capsule_loss_fn(y_pred, y_true, coefficient: float = 0.5):
    """The main loss function of Capsule Network.
    Args:
        y_pred: predicted labels by CapsNet, size=[batch, classes]
        y_true: true labels, one hot label, size=[batch, classes]

    Attributions:
        :math:`Capsule loss = Margin loss + lam_recon * reconstruction loss`
        However, for this task, we only use margin loss

    Returns:
        Capsule loss value.
    """

    margin_loss = (
        y_true * torch.clamp(0.9 - y_pred, min=0.0) ** 2
        + coefficient * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.0) ** 2
    )
    return margin_loss.sum(dim=-1).mean()
