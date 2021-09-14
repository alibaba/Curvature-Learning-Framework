# Example of HyperML using curvlearn

In this example, we implement HyperML (Lucas V. Tran, et al., HyperML: A Boosting Metric Learning Approach in Hyperbolic Space for Recommender Systems, WSDM'20) by curvlearn over the Kindle-Store dataset from Amazon recommender [datasets](https://nijianmo.github.io/amazon/index.html).

We process the data to extract the 5+ cores, such that each of the remaining users and items have at least 5 ratings each.
The data is then re-indexed and formatted by users into list of python list, i.e.,

```
[
  [1, 2, 3, ..., 30], # Item list of user 1
  [31, 32, 33, ..., 37], # Item list of user 2
  ...
  [136372, 31588, ..., 29315] # Item list of the last user
]
```

The configurations of training are listed in ```config.py```, leading to the following performance.

| Manifold      | HR@10 |
| ------------- | ----- |
| Euclidean     | 69.89 |
| PoincareBall  | 69.21 |
| Stereographic | 77.18 |


The entry of the training is ```train.py```. Launch the training by 

```
python examples/hyperml/train.py
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

