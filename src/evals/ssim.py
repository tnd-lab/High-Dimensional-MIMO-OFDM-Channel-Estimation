def ssim_func(pred, target, eps=1e-8):
    """
    SSIM for batched complex-valued or real-valued channel matrices.
    Input:
        pred, target: shape (B, C, H, W)
    Output:
        mean SSIM across the batch
    """
    B = pred.size(0)
    pred = pred.view(B, -1)  # flatten (C*H*W)
    target = target.view(B, -1)

    # Compute means
    mu_pred = pred.mean(dim=1)
    mu_target = target.mean(dim=1)

    # Compute variances
    sigma_pred_sq = ((pred - mu_pred.unsqueeze(1)) ** 2).mean(dim=1)
    sigma_target_sq = ((target - mu_target.unsqueeze(1)) ** 2).mean(dim=1)

    # Compute covariance
    covariance = (
        (pred - mu_pred.unsqueeze(1)) * (target - mu_target.unsqueeze(1))
    ).mean(dim=1)

    # Compute c1 and c2
    c1 = (0.01 * target.max(dim=1).values) ** 2 + eps
    c2 = (0.03 * target.max(dim=1).values) ** 2 + eps

    # SSIM formula
    numerator = (2 * mu_pred * mu_target + c1) * (2 * covariance + c2)
    denominator = (mu_pred**2 + mu_target**2 + c1) * (
        sigma_pred_sq + sigma_target_sq + c2
    )
    ssim = numerator / (denominator + eps)

    return ssim.mean().item()
