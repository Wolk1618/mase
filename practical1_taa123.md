# Practical 1

## Lab 1

### 1. What is the impact of varying batch sizes and why?

I got these accuracy results for various batch sizes :

- Batch Size 64: Accuracy: 55.0%
- Batch Size 128: Accuracy: 53.1%
- Batch Size 256: Accuracy: 51.2%
- Batch Size 512: Accuracy: 47.0%
- Batch Size 1024: Accuracy: 33.4%

During the training of the networks, I noticed that larger batch sizes lead to faster training time. This is due to parallelization. In the given scenarii, the model trained with a batch size of 64 achieved the highest accuracy. The results show a higher accuracy for smaller batch size. Thus, smaller batch sizes can offer better generalization but with slower convergence. A tradeoff has to be found. 

### 2. What is the impact of varying maximum epoch number?

I got these accuracy results for various epochs number :

- 1 epoch: Accuracy: 28.7% ; Loss : 1.53
- 5 epochs: Accuracy: 46.7% ; Loss : 1.41
- 10 epochs: Accuracy: 49.8% ; Loss : 1.33
- 50 epochs: Accuracy: 61.7% ; Loss : 1.10
- 100 epochs: Accuracy: 67.9% ; Loss : 0.996

As we can see on these results, increasing the maximum epoch number improves accuracy. It allows the model to learn more from the data. However, there's a risk of overfitting if the model learns noise in the training data. In this case, as the maximum epochs increase, both accuracy and loss improve, indicating that the model benefits from more training.

### 3. What is happening with a large learning rate and what is happening with a small learning rate and why? What is the relationship between learning rates and batch sizes?

I got these accuracy results for various learning rates :

- 1e-01: Accuracy: 57.0%
- 1e-03: Accuracy: 71.7%
- 1e-04: Accuracy: 67.9%
- 1e-05: Accuracy: 51.2%
- 1e-07: Accuracy: 26.4%

Here, the small learning rate entails slow convergence and don't reach the minimum, resulting in poor accuracy. On the other hand, the high learning rates overshoot the minimum and fail to converge, resulting in poor accuracy as well. Larger batch sizes allow for more stable training with larger learning rates.

### 4. Implement a network that has in total around 10x more parameters than the tiny network.

The tiny network has 127 parameters. I thus want to create a model with roughly 1K parameters.
I took the toy network and added some layers to it. Here is my architecture :
```{python}
            # 1st LogicNets Layer
            nn.BatchNorm1d(16),
            nn.ReLU(16),
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(16),

            # 2nd LogicNets Layer
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(32),

            # 3rd LogicNets Layer
            nn.Linear(32, 5),
            nn.BatchNorm1d(5),
            nn.ReLU(5),
```

This network contains 1.1K parameters.

### 5. Test your implementation and evaluate its performance.

To be able to compare the results with tiny and toy networks, I used the same training parameters, namely 128 batch size, 1e-03 as learning rate and 10 epochs.The tiny network achieves 71.7% accuracy and 0.852 loss.

After testing my implementation, I got 74.1% accuracy and 0.723 loss, which is better than the tiny network but not tremendous. There is still some room for improvements (using more or larger layers, using optimizers or dropout).

## Lab 2

### 1. Explain the functionality of `report_graph_analysis_pass` and its printed jargons such as `placeholder`, `get_attr`.

The function `report_graph_analysis_pass` is used to print the details of the architecture of a network. It first prints the graph that corresponds to the network. Then it prints an overview of the different types of nodes in the graph, among which :
- placeholder : represents an input of the graph
- get_attr : represents a parameter we use at a given point in the hierarchy
- call_function : represents a function
- call_method : represents a method called on an object (e.g. a tensor)
- call_module : represents a module in the hierarchy
- output : an output of the graph

In our example, we only have 4 modules (1 batch normalisation, 2 ReLU and 1 linear), which is consistent with the output we got from this pass.

### 2. What are the functionalities of `profile_statistics_analysis_pass` and `report_node_meta_param_analysis_pass` respectively?

The `profile_statistics_analysis_pass` pass is used to collect statistics about some nodes of the graph. It can collect statistics on 2 kinds of nodes : the activation nodes (`target_activation_nodes`) and the layer nodes (`target_weight_nodes`). However, this pass doesn't print the results (it stores the results in the metadata) so we need to use the `report_node_meta_param_analysis_pass` pass to print the data. 

### 3. Explain why only 1 OP is changed after the `quantize_transform_pass`.

In the arguments of the quantize pass `pass_args`, the target is set to `linear`, and all the other types of nodes should be ignored (owing to the default configuration parameter set to `{"name": None}`). As the tiny network only has 4 modules and only 1 linear, only one OP is changed after the pass is applied.

### 4. Write some code to traverse both `mg` and `ori_mg`, check and comment on the nodes in these two graphs. You might find the source code for the implementation of `summarize_quantization_analysis_pass` useful.

I wrote some code to traverse these mase graph. As expected, the Linear nodes have been changed. As we can notice, they changed from type "Linear" to type "LinearInteger" due to quantization.

Output of my code :
```
[['linear', 'Linear']]
[['linear', 'LinearInteger']]
```

### 5. Perform the same quantisation flow to the bigger JSC network that you have trained in lab1. You must be aware that now the `pass_args` for your custom network might be different if you have used more than the `Linear` layer in your network.

Since I only used some bigger Linear layers, I didn't need to change the `pass_args`. The quantization process worked well and all 3 Linear layers have been changed.

Output of my code :
```
[['linear', 'Linear'], ['linear', 'Linear'], ['linear', 'Linear']]
[['linear', 'LinearInteger'], ['linear', 'LinearInteger'], ['linear', 'LinearInteger']]
```

### 6. Write code to show and verify that the weights of these layers are indeed quantised. You might need to go through the source code of the implementation of the quantisation pass and also the implementation of the Quantized Layers

I wrote the following piece of code to print the weights in the different modules :
```
for node, ori_node in zip(mg.fx_graph.nodes, ori_mg.fx_graph.nodes) :
    if get_mase_op(ori_node) == "linear":
        print("\noriginal nodes :\n")
        print("precision : " + str(ori_node.meta["mase"].parameters["common"]["args"]["weight"]["precision"]))
        print("weights : " + str(ori_node.meta["mase"].parameters["common"]["args"]["weight"]["value"]))
        print("bias : " + str(ori_node.meta["mase"].parameters["common"]["args"]["bias"]["value"]))
        
        print("quantized nodes :\n")
        print("precision : " + str(node.meta["mase"].parameters["common"]["args"]["weight"]["precision"]))
        print("weights and bias : ")
        tnode = get_node_actual_target(node)
        tnode.quantize_weights()
```
Where `quantize_weights` is a method I have written in the _LinearBase class to compute the quantized weights and print them (as in forward method).

I tested my code and I thus noticed the quantization of weights. Here is a comparison of the first 10 weights of original and quantized nodes :
`-0.2631,  0.2186, -0.1505,  0.0112, -0.0510,  0.1241,  0.0753,  0.3352, 0.0506,  0.3139`
`-0.2500,  0.1875, -0.1250,  0.0000, -0.0625,  0.1250,  0.0625,  0.3125, 0.0625,  0.3125`

We can clearly notice that each value has been scaled down or up to the nearest multiple of 0.0625 (the quantized value) to reduce the amount of bits necessary to code the weights information and then reduce the memory size of the network.

### 7. Load your own pre-trained JSC network, and perform the quantisation using the command line interface.

I wrote a new config file where I replaced the jsc-tiny network with the name of my bigger network and then launched the quantization pass using this command :
`./ch transform --config configs/examples/jsc_thomas.toml`

I got the same output as in the notebook with the same layers change.