import time
from datetime import datetime
import wandb


def get_runs(after=0, before=time.time(), **filters):
    api = wandb.Api()

    gte = datetime.fromtimestamp(after).isoformat()
    lte = datetime.fromtimestamp(before).isoformat()

    filters.update({"created_at": {"$gte": gte, "$lte": lte}})

    runs = api.runs(
        path="jmaen-team/zero-shot-das-denoising",
        filters=filters
    )

    return runs


def calculate_relative_ranks(runs, num_data, num_denoisers, denoiser_ids=None, k=1):
    if denoiser_ids is None:
        denoiser_ids = list(range(num_denoisers))

    grouped_runs = {i: [] for i in range(num_data)}
    for run in runs:
        if run.config["denoiser_id"] in denoiser_ids:
            id = run.config["data_id"]
            grouped_runs[id].append(run)

    results = {i: {f"first_{k}_count": {"psnr": 0, "ssim": 0, "lpips": 0}, "average_rank": {"psnr": 0, "ssim": 0, "lpips": 0}} for i in range(num_denoisers)}
    for _, group in grouped_runs.items():
        psnr_order = sorted(group, key=lambda x: x.summary["psnr"], reverse=True)
        ssim_order = sorted(group, key=lambda x: x.summary["ssim"], reverse=True)
        # lpips_order = sorted(group, key=lambda x: x.summary["lpips"])

        for i in range(k):
            results[psnr_order[i].config["denoiser_id"]][f"first_{k}_count"]["psnr"] += 1
            results[ssim_order[i].config["denoiser_id"]][f"first_{k}_count"]["ssim"] += 1
            # results[lpips_order[i].config["denoiser_id"]][f"first_{k}_count"]["lpips"] += 1

        for i, (psnr_run, ssim_run) in enumerate(zip(psnr_order, ssim_order)):
            results[psnr_run.config["denoiser_id"]]["average_rank"]["psnr"] += (i + 1)/num_data
            results[ssim_run.config["denoiser_id"]]["average_rank"]["ssim"] += (i + 1)/num_data
            # results[lpips_run.config["denoiser_id"]]["average_rank"]["lpips"] += (i + 1)/num_data

    for denoiser, ranks in results.items():
        print(f"{denoiser}: {ranks}")


def update_config(runs, **data):
    if type(runs) is not list:
        runs = [runs]

    for run in runs:
        for key, value in data.item():
            run.config[key] = value
            run.update()
