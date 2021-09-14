# Example of HGCN using curvlearn

In this example, we implement HGCN (Ines Chami, et al., Hyperbolic Graph Convolutional Neural Networks, NeurIPS'19) by curvlearn over the OpenFlights airport dataset.
3,188 nodes are kept in the dataset. The adjacency matrix is recorded in ```adj.pkl```, and the numeric features are collected in ```features.pkl```.

The configurations of training are listed in ```config.py```, leading to the following performance.

| Manifold      | AUC   |
| ------------- | ----- |
| Euclidean     | 93.68 |
| PoincareBall  | 94.51 |
| Stereographic | 95.13 |


The entry of the training is ```train.py```. Launch the training by 

```
python examples/hgcn/train.py
```
and have fun!

The code has been tested under the following environment settings:

```
Hardware:
Tesla P100 - 16GB (Actual consumption: 1.4GB)
Intel Xeon E5-2682 v4 @ 2.50GHz

Python dependencies:
tensorflow-gpu==1.15.0
numpy==1.16.5
```

