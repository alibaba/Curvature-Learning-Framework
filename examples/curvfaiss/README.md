# CurvFaiss

## Introduction

```CurvFaiss``` is a library for efficient similarity search and clustering of dense vectors in non-Euclidean manifolds. 

Based on [*Faiss*](https://github.com/facebookresearch/faiss), ```CurvFaiss``` develops a new Index ```IndexFlatStereographic``` to support nearest neighbors searching with stereographic distance metric. Together with ```CurvLearn```, non-Euclidean model training and efficient inference are feasible.

Currently ```CurvFaiss``` supports retrieving neighbors in Hyperbolic, Euclidean, Spherical space. The indexing method is based on exact searching. Due to the parallelism in both the data level and instruction level, the indices can be built in less than two hours for 100 million nodes.

To those who want to apply on their own customized metric or optimize the indexing method, a hands-on [*tutorial*](customized.md) is also provided.

## Installation

```CurvFaiss``` requires curvlearn and python3.

The preferred way for installing is via `pip`.

```bash
pip install curvfaiss
```

Since the source codes are compiled under CentOS, as for other platforms, we recommend users follow the [*tutorial*](customized.md) to solve the code dependency.

## Usage

A frequent problem is the runtime dependency.
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python -c "from os.path import abspath,dirname,join; import curvlearn as cl; print(join(dirname((dirname(cl.__file__))),'curvfaiss'))"`
```

Since ```IndexFlatStereographic``` is inherited from ```IndexFlat```, the usgae is the same with ```IndexFlatL2``` in faiss except with an additional parameter ```curvature```.

```python
import curvfaiss

# build index, retrievaling in hyperbolic, euclidean, spherical metric with respect to curvature < 0, = 0, > 0
index = curvfaiss.IndexFlatStereographic(dim, curvature)
index.add(embedding)

# knn search
knn_distance, knn_index = index.search(query, topk)

print(knn_distance, knn_index)
```

See the [*full demo*](knn.py) here!