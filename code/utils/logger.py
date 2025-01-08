from typing import Any, Dict
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import wandb
from IPython.display import Video, HTML, display


# TODO: add method to plot line / marker at specific step (e.g. stopping point, skip fadein, ...)
class Logger():
    def __init__(self):
        self.colors = [
            '#D66060',
            '#FFAD69',
            '#7C88E5',
            '#5AB3A7',
            '#D991D1',
            '#F5E663',
            '#A3B9C9',
            '#F09EAC',
            '#91B9ED',
        ]

    def init_run(self, mode: str, options: Dict[str, Any]):
        self.mode = mode

        self.name = f"{options["id"]} - {options["variant"]} - {options["architecture"]}"
        print(f"Running: {self.name}")

        if self.mode == "local":
            self.data = {}
            self.step = 0
        elif self.mode == "wandb":
            wandb.init(
                project=options.pop("project"),
                entity=options.pop("entity"),
                group=options.pop("group"),
                name=self.name,
                config=options,
                # settings=wandb.Settings(init_timeout=120),
            )

    def log(self, data: Dict[str, Any]):
        if self.mode == "local":
            self.log_local(data)
        elif self.mode == "wandb":
            for key, value in data.items():
                if type(value) is list:
                    data[key] = value[0]
            wandb.log(data)

    def log_local(self, data: Dict[str, Any]):
        for key, value in data.items():
            if key in self.data:
                self.data[key].append(value)
            else:
                if type(value) is list:
                    self.data[key] = [[math.nan for _ in value] for _ in range(self.step)]
                else:
                    self.data[key] = [math.nan for _ in range(self.step)]
                self.data[key].append(value)

        self.step += 1

    def finish(self, summary: Dict[str, Any]):
        if self.mode == "local":
            self.display()
            print(f"Summary: {summary}\n")
        elif self.mode == "wandb":
            wandb.run.summary.update(summary)
            wandb.finish()

    def display(self):
        metrics = {key: value for key, value in self.data.items() if isinstance(value[0], (float, list))}
        self.plot_metrics(metrics)

        images = {key: value for key, value in self.data.items() if isinstance(value[0], torch.Tensor)}
        self.generate_videos(images)

    def plot_metrics(self, metrics):
        num_metrics = len(metrics)
        rows = math.ceil(num_metrics / 3)

        _, axes = plt.subplots(rows, 3, figsize=(20, rows * 4))
        axes = axes.flatten()

        for ax, (key, y_values), color in zip(axes, metrics.items(), self.colors):
            x_values = list(range(len(y_values)))

            if type(y_values[0]) is list:
                y_values = [list(row) for row in zip(*y_values)]
                for y in y_values:
                    ax.plot(x_values, y, color=color)
            else:
                ax.plot(x_values, y_values, color=color)
            ax.set_title(key, fontsize=10)

            ax.set_xlabel("Step") 
            ax.set_ylabel("Value") 

        for i in range(num_metrics, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    def generate_videos(self, images):
        for key, values in images.items():
            _, height, width = values[0].squeeze().shape

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(f"output/videos/{self.name}_{key}.mp4", fourcc, 120, (width, height))

            for value in values:
                value = value.squeeze()
                frame = (value.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            video_writer.release()
