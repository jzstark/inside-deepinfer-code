def cal_diff(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float, float]:
    if torch.equal(x, y):
        return 0.0, 0.0, 0.0, 0.0
    x, y = x.double(), y.double()
    rmse = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    chebyshev_dis = (x - y).abs().max().item()
    kl = ((torch.exp(x - y) - 1) - (x - y)).mean().item()
    return cos_diff, rmse, chebyshev_dis, kl
