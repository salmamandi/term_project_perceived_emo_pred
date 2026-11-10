import torch

def ccc_loss(pred, gt):
    """
    Differentiable Lin's Concordance Correlation Coefficient (CCC) Loss.
    CCC measures agreement between two continuous variables (-1 to 1).
    Loss is (1 - CCC), which is 0 at perfect agreement and bounded in [0, 2].
    """
    mean_pred = torch.mean(pred)
    mean_gt = torch.mean(gt)
    var_pred = torch.var(pred, unbiased=False)
    var_gt = torch.var(gt, unbiased=False)
    cov = torch.mean((pred - mean_pred) * (gt - mean_gt))
    ccc = (2.0 * cov) / (var_pred + var_gt + (mean_pred - mean_gt) ** 2 + 1e-8)
    return 1.0 - ccc

def compute_loss(pred_v, pred_a, gt_v, gt_a, delta, alpha=0.5):
    """
    Hybrid loss combining MSE, CCC loss, and emotion transition regularization.
    alpha: weight of CCC loss (0.0 = pure MSE, 1.0 = pure CCC).
    """
    mse_v = torch.mean((pred_v - gt_v) ** 2)
    mse_a = torch.mean((pred_a - gt_a) ** 2)

    c_loss_v = ccc_loss(pred_v, gt_v)
    c_loss_a = ccc_loss(pred_a, gt_a)

    transition_loss = torch.mean(delta ** 2)

    loss_v = (1.0 - alpha) * mse_v + alpha * c_loss_v
    loss_a = (1.0 - alpha) * mse_a + alpha * c_loss_a

    return loss_v + loss_a + 0.01 * transition_loss

def compute_loss_task(pred, gt, delta, alpha=0.5):
    """
    Single-task hybrid loss combining MSE, CCC loss, and transition regularization.
    """
    mse = torch.mean((pred - gt) ** 2)
    c_loss = ccc_loss(pred, gt)
    transition_loss = torch.mean(delta ** 2)
    return (1.0 - alpha) * mse + alpha * c_loss + 0.01 * transition_loss