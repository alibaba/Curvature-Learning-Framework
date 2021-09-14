# CurvLearn For Developers

## How to build a new manifold

Inherit from ```Manifold``` class and implement all astract method in ```Geometry``` class.

Refer [Euclidean manifold](manifolds/euclidean.py) as a template.

## How to build a new tensor operation

Implement the operation method in ```Manifold``` class。

If the operation cannot broadcast to other manifold, just implement it in proper manifold.

Refer [Product manifold](manifolds/product.py) as a template.

## How to build a new riemannian optimizer

Same with tensorflow.

Notice riemannian parameters are identified by their name, you might preprocess them in initialization stage.

Refer [RSGD optimizer](optimizers/rsgd.py) as a template.

