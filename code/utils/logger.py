from typing import Any, Dict
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import wandb
from IPython.display import Video, HTML, display


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

        self.name = f"{options["data_id"]} - {options["variant"]} - {options["loss"]} - {options["architecture"]}"
        print(f"Running: {self.name}")

        if self.mode == "local":
            self.data = {}
            self.markers = []
            self.step = 0
        elif self.mode == "wandb":
            wandb.init(
                project=options.pop("project"),
                entity=options.pop("entity"),
                group=options.pop("group"),
                name=self.name,
                config=options,
                settings=wandb.Settings(init_timeout=120),
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
                    self.data[key] = [[None for _ in value] for _ in range(self.step)]
                else:
                    self.data[key] = [None for _ in range(self.step)]
                self.data[key].append(value)

        self.step += 1

    def mark(self, step=None):
        if self.mode != "local":
            return
        
        if step is None:
            step = self.step

        self.markers.append(step)

    def finish(self, summary: Dict[str, Any]):
        if self.mode == "local":
            print(f"Summary: {summary}\n")
            if self.step > 0:
                self.display()
        elif self.mode == "wandb":
            wandb.run.summary.update(summary)
            wandb.finish()

    # TODO: cleanup

    def display(self):
        metrics = self.filter_data(self.data, (int, float))
        if len(metrics) > 0:
            self.visualize_metrics(metrics)

        tensors = self.filter_data(self.data, torch.Tensor)
        if len(tensors) > 0:
            self.visualize_tensors(tensors)

    def filter_data(self, data, type):
        filtered_data = {}
        for key, values in data.items():
            for value in values:
                if value is None:
                    continue

                if isinstance(value, list):
                    value = value[0]

                if isinstance(value, type):
                    filtered_data[key] = values

                break

        return filtered_data

    def visualize_metrics(self, metrics):
        num_metrics = len(metrics)
        rows = math.ceil(num_metrics / 3)

        _, axes = plt.subplots(rows, 3, figsize=(20, rows * 4))
        axes = axes.flatten()

        for ax, (key, y_values), color in zip(axes, metrics.items(), self.colors):
            x_values = list(range(len(y_values)))

            if type(y_values[0]) is list:
                y_values = [[math.nan if elem is None else elem for elem in row] for row in zip(*y_values)]
                for y in y_values:
                    ax.plot(x_values, y, color=color)
            else:
                y_values = [math.nan if elem is None else elem for elem in y_values]
                ax.plot(x_values, y_values, color=color)

            for marker in self.markers:
                ax.axvline(x=marker, color="#DB324D")

            ax.set_title(key, fontsize=10)
            ax.set_xlabel("Step") 
            ax.set_ylabel("Value") 

        for i in range(num_metrics, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    def visualize_tensors(self, tensors):
        for key, values in tensors.items():
            frames = [self.tensor_to_frame(value, i) for i, value in enumerate(values)]

            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            video_writer = cv2.VideoWriter(f"output/audio/{self.name}_{key}.mp4", fourcc, 60, (width, height))

            for frame in frames:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            video_writer.release()

    def tensor_to_frame(self, tensor, step):
        if isinstance(tensor, list):
            frame = self.concatenate_tensors(tensor)
        else:
            tensor = tensor.squeeze()
            tensor = tensor / 2 + 0.5
            tensor = tensor.clamp(0, 1)
            frame = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        height, width, _ = frame.shape
        text_height = 30
        new_frame = np.full((height + text_height, width, 3), 255, dtype=np.uint8)
        new_frame[text_height:, :, :] = frame
        
        text = f"Step {step}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (text_height + text_size[1]) // 2
        
        cv2.putText(new_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return new_frame
    
    def concatenate_tensors(self, tensors, target_height=256, separator_width=10):
        resized_frames = []
        for tensor in tensors:
            tensor = tensor.squeeze()
            tensor = tensor / 2 + 0.5
            tensor = tensor.clamp(0, 1)
            if tensor.dim() != 3:
                tensor = tensor.unsqueeze(0)
            if len(tensor) != 3:
                tensor = tensor[:1].expand((3, -1, -1))
            frame = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            resized_frame = self.resize_frame(frame, target_height)
            resized_frames.append(resized_frame)
        
        separator = np.full((target_height, separator_width, 3), 255, dtype=np.uint8)
        
        concatenated_frame = resized_frames[0]
        for frame in resized_frames[1:]:
            concatenated_frame = np.hstack((concatenated_frame, separator, frame))
        
        return concatenated_frame

    def resize_frame(self, frame, target_height):
        h, w, _ = frame.shape
        scale = target_height / h
        new_width = int(w * scale)
        return cv2.resize(frame, (new_width, target_height))
