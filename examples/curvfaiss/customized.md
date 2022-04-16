# Tutorial on developing customized distance metric in Faiss

## Build the environment
1) build the docker from [*Dockerfile*](https://github.com/facebookresearch/faiss/blob/main/Dockerfile).
2) install swig4, cmake, anaconda with py=3.7/numpy.
3) follow the installation [*wiki*](https://github.com/facebookresearch/faiss/wiki/Installing-Faiss#compiling-the-python-interface-within-an-anaconda-install), [*readme*](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md#building-from-source) to verify the correctness.


## Develop the new metric
Here is an [*example*](https://github.com/XuZhirong/faiss/commit/a7d5466884d924226aa1d43dd5a41dd44606817c).

The main modifications include: declaring the index in ```IndexFlat```, implementing the metric in ```utils```.

## Compile and Publish
1) remove the build files and rebuild.
2) put  *.so files to the proper location.
3) write [*setup.py*](https://github.com/facebookresearch/faiss/blob/main/faiss/python/setup.py), where the ```package_data``` field should include the *.so files.
4) upload to pypi/docker. [*Instructions*](https://zhuanlan.zhihu.com/p/61174349) are here.