# Relational-AutoEncoders

This package includes the implementation of my ICML 2020 work **"Learning Autoencoders with Relational Regularization"** [https://arxiv.org/pdf/2002.02913.pdf]

# Main Dependencies
* argparse
* matplotlib
* numpy
* pickle
* pytorch
* sklearn

# Platform:
We test this example in a conda environment on Windows 10, with cuda 10.1 and one 1080Ti GPU


# Test our method:
1. Open a terminal and go to the folder of the example.
2. python test_rae.py --model-type deterministic --source-data **DATANAME**
(Learning a deterministic RAE for a dataset.)
3. python test_rae.py --model-type probabilistic --source-data **DATANAME**
(Learning a probabilistic RAE for a dataset.)

* The **DATANAME** can be **MNIST** and **CelebA**

# Test baselines:
1. Open a terminal and go to the folder of the example.
2. python test_**MODEL**.py --source-data **DATANAME**

* The **MODEL** can be **vae**, **wae**, **swae**, **gmvae**, and **vampprior**

All the results are in the folder "Results".
