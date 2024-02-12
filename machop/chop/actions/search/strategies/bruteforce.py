import torch
import pandas as pd
import logging
import time
import random

import matplotlib.pyplot as plt

from .base import SearchStrategyBase
from torchmetrics.classification import MulticlassAccuracy


logger = logging.getLogger(__name__)


def min_max_scaling(values):
    min_val = min(values)
    max_val = max(values)
    scaled_values = [(x - min_val) / (max_val - min_val) for x in values]
    return scaled_values


class SearchStrategyBruteforce(SearchStrategyBase):
    is_iterative = False

    def _post_init_setup(self):
        self.sum_scaled_metrics = self.config["setup"]["sum_scaled_metrics"]
        self.metric_names = list(sorted(self.config["metrics"].keys()))
        if not self.sum_scaled_metrics:
            self.directions = [
                self.config["metrics"][k]["direction"] for k in self.metric_names
            ]
        else:
            self.direction = self.config["setup"]["direction"]


    def search(self, search_space):

        data_module = self.data_module
        data_module.prepare_data()
        data_module.setup()

        sampled_indexes = {}
        for name, length in search_space.choice_lengths_flattened.items():
            sampled_indexes[name] = random.randint(0, length - 1)
        sampled_config = search_space.flattened_indexes_to_config(sampled_indexes)
        print(sampled_config)

        dummy_in = search_space.dummy_input

        metric = MulticlassAccuracy(num_classes=5)
        num_batchs = 5
        # This first loop is basically our search strategy,
        # in this case, it is a simple brute force search

        acc_avg, loss_avg, lat_avg, size_avg = [], [], [], []
        for _, config in enumerate(search_space):

            is_eval_mode = self.config.get("eval_mode", True)
            mg = search_space.rebuild_model(sampled_config, is_eval_mode)
            j = 0

            # this is the inner loop, where we also call it as a runner.
            accs, losses, latencies, sizes = [], [], [], []
            for inputs in data_module.train_dataloader():
                xs, ys = inputs
                preds = mg.model(xs)

                # Compute loss
                loss = torch.nn.functional.cross_entropy(preds, ys)

                # Compute accuracy
                acc = metric(preds, ys)

                # Measure Latency
                with torch.no_grad():
                    start_time = time.time()
                    _ = mg.model(**dummy_in)
                    end_time = time.time()
                latency = end_time - start_time

                # Measure Model Size (Memory)
                num_params = sum(p.numel() for p in mg.model.parameters())
                precision = config["linear"]["config"]["weight_width"]
                size = num_params * precision

                accs.append(acc)
                losses.append(loss)
                latencies.append(latency)
                sizes.append(size)


                if j > num_batchs:
                    break
                j += 1

            acc_avg.append((sum(accs) / len(accs)).item())
            loss_avg.append((sum(losses) / len(losses)).item())
            lat_avg.append(sum(latencies) / len(latencies))
            size_avg.append(sum(sizes) / len(sizes))
        
        self.print_best(acc_avg, loss_avg, lat_avg, size_avg)


    @staticmethod
    def print_best(acc_avg, loss_avg, lat_avg, size_avg):
        combined = [i + j + k for i, j, k in zip(min_max_scaling(loss_avg), min_max_scaling(lat_avg), min_max_scaling(size_avg))]

        plt.plot(min_max_scaling(combined), label='combined metrics')

        plt.xlabel('Search number')
        plt.ylabel('Values')
        plt.title('Combined Metrics')
        plt.legend()

        plt.show()