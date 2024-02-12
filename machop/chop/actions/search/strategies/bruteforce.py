import torch
import pandas as pd
import logging
import time
import copy

from tqdm import tqdm
from tabulate import tabulate

import matplotlib.pyplot as plt

from .base import SearchStrategyBase
from torchmetrics.classification import MulticlassAccuracy


logger = logging.getLogger(__name__)

name_type = {
    "type": "linear",
    "name": "seq_blocks_2",
}

def min_max_scaling(values):
    min_val = min(values)
    max_val = max(values)
    scaled_values = [(x - min_val) / (max_val - min_val) for x in values]
    return scaled_values

def generate_short_config(config, layers):
    str_config = ""
    str_config += str(config[layers]['config']['data_in_width'])
    str_config += str(config[layers]['config']['data_in_frac_width'])
    str_config += "_"
    str_config += str(config[layers]['config']['weight_width'])
    str_config += str(config[layers]['config']['weight_frac_width'])
    str_config += "_"
    str_config += str(config[layers]['config']['bias_width'])
    str_config += str(config[layers]['config']['bias_frac_width'])
    return str_config



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


    def create_config_list(self, search_space):

        layers = name_type[search_space.config['setup']['by']]
    
        pass_args = {
        "by": search_space.config['setup']['by'],
        "default": {"config": {"name": None}},
        layers: {
                "config": {
                    "name": "integer",
                    # data
                    "data_in_width": 8,
                    "data_in_frac_width": 4,
                    # weight
                    "weight_width": 8,
                    "weight_frac_width": 4,
                    # bias
                    "bias_width": 8,
                    "bias_frac_width": 4,
                }
        },}

        configs = []
        # TO DO : Implement what to do if fracs aren't NA 
        for d_config in search_space.config['seed'][layers]['config']['data_in_width']:
            for w_config in search_space.config['seed'][layers]['config']['weight_width']:
                for b_config in search_space.config['seed'][layers]['config']['bias_width'] :
                    pass_args[layers]['config']['data_in_width'] = d_config
                    pass_args[layers]['config']['data_in_frac_width'] = d_config // 2
                    pass_args[layers]['config']['weight_width'] = w_config
                    pass_args[layers]['config']['weight_frac_width'] = w_config // 2
                    pass_args[layers]['config']['bias_width'] = b_config
                    pass_args[layers]['config']['bias_frac_width'] = b_config // 2
                    # dict.copy() and dict(dict) only perform shallow copies
                    # in fact, only primitive data types in python are doing implicit copy when a = b happens
                    configs.append(copy.deepcopy(pass_args))

        return configs, layers
    
    def compute_metrics_for_config(self, config, search_space, data_module, num_batchs, layers) :

        metric = MulticlassAccuracy(num_classes=5)

        is_eval_mode = self.config.get("eval_mode", True)
        mg = search_space.rebuild_model(config, is_eval_mode)

        j = 0
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
                _ = mg.model(**search_space.dummy_input)
                end_time = time.time()
            latency = end_time - start_time

            # Measure Model Size (Memory)
            num_params = sum(p.numel() for p in mg.model.parameters())
            precision = config[layers]["config"]["weight_width"]
            size = num_params * precision

            accs.append(acc)
            losses.append(loss)
            latencies.append(latency)
            sizes.append(size)

            if j > num_batchs:
                break
            j += 1

        return accs, losses, latencies, sizes


    def search(self, search_space):

        data_module = self.data_module
        data_module.prepare_data()
        data_module.setup()

        config_list, layers = self.create_config_list(search_space)
        
        num_batchs = 5
        acc_avg, loss_avg, lat_avg, size_avg = [], [], [], []
        for config in tqdm(config_list, desc="Processing"):
            accs, losses, latencies, sizes = self.compute_metrics_for_config(config, search_space, data_module, num_batchs, layers)

            acc_avg.append((sum(accs) / len(accs)).item())
            loss_avg.append((sum(losses) / len(losses)).item())
            lat_avg.append(sum(latencies) / len(latencies))
            size_avg.append(sum(sizes) / len(sizes))
        
        self.print_best(acc_avg, loss_avg, lat_avg, size_avg, config_list, layers)


    @staticmethod
    def print_best(acc_avg, loss_avg, lat_avg, size_avg, configs, layers):
        combined = [i + j + k for i, j, k in zip(min_max_scaling(loss_avg), min_max_scaling(lat_avg), min_max_scaling(size_avg))]

        # Get top 3 minimum values and their indices
        sorted_indices_and_values = sorted(zip(combined, acc_avg, loss_avg, lat_avg, size_avg, configs), key=lambda x: x[0])[:3]

        # Extract top 3 metrics
        top_3_acc = [acc for _, acc, loss, _, _, _ in sorted_indices_and_values]
        top_3_loss = [loss for _, _, loss, _, _, _ in sorted_indices_and_values]        
        top_3_latencies = [latency for _, _, _, latency, _, _ in sorted_indices_and_values] 
        top_3_sizes = [size for _, _, _, _, size, _ in sorted_indices_and_values] 
        top_3_configs = [generate_short_config(config, layers) for _, _, _, _, _, config in sorted_indices_and_values] 

        lists_combined = [top_3_configs, top_3_acc, top_3_loss, top_3_latencies, top_3_sizes]

        # Transpose the list of lists for better tabulation
        transposed_lists = list(map(list, zip(*lists_combined)))

        # Create a table using tabulate
        table = tabulate(transposed_lists, headers=['Configuration (short code)', 'Accuracy', 'Loss', 'Latency (s)', 'Memory size (bits)'], tablefmt='grid')

        # Print the table
        print("\n3 best results :")
        print(table)