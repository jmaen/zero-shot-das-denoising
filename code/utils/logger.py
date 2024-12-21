from typing import Any, Dict
import math
import matplotlib.pyplot as plt
import wandb


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

        name = f"{options["id"]} | {options["variant"]} | {options["architecture"]}"
        print(f"Running: {name}")

        if self.mode == "local":
            self.data = {}
            self.step = 0
        elif self.mode == "wandb":
            wandb.init(
                project=options.pop("project"),
                entity=options.pop("entity"),
                group=options.pop("group"),
                name=name,
                config=options,
                settings=wandb.Settings(init_timeout=120),
            )

    def log(self, data: Dict[str, Any]):
        if self.mode == "local":
            self.log_local(data)
        elif self.mode == "wandb":
            wandb.log(data)

    def log_local(self, data: Dict[str, Any]):
        for key, value in data.items():
            if key in self.data:
                self.data[key].append(value)
            else:
                self.data[key] = [math.nan for _ in range(self.step)]
                self.data[key].append(value)

        self.step += 1

    def finish(self, summary: Dict[str, Any]):
        if self.mode == "local":
            self.plot()
            print(f"Summary: {summary}\n")
        elif self.mode == "wandb":
            wandb.run.summary.update(summary)
            wandb.finish()

    def plot(self):
        num_metrics = len(self.data)
        rows = math.ceil(num_metrics / 3)

        _, axes = plt.subplots(rows, 3, figsize=(20, rows * 4))
        axes = axes.flatten()

        for ax, (key, y_values), color in zip(axes, self.data.items(), self.colors):
            x_values = list(range(len(y_values)))

            ax.plot(x_values, y_values, color=color)
            ax.set_title(key, fontsize=10)

            ax.set_xlabel("Step") 
            ax.set_ylabel("Value") 

        for i in range(num_metrics, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
