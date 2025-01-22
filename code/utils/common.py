import torch


def summary(tensor: torch.Tensor):
    summary = {
        # "shape": tensor.shape,
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "mean": tensor.mean().item(),
        "var": tensor.var().item(),
    }

    return summary


def parameter_stats(net):
    d = {"min": 0, "max": 0, "mean": 0, "var": 0}
    for parameter in net.parameters():
        for k, v in summary(parameter.data).items():
            d[k] += v
    n = len(list(net.parameters()))
    d = {k: v / n for k, v in d.items()}
    return d
