# polynomial_gelu

A tiny, fast, crude approximation of the gelu activation function using piecewise polynomials. This code is written in Jax and works both on CPU, TPU and GPU.

The approximation is not close enough to be used in already trained models. The polynomial gelu is rather intended to be used from scratch for pretraining models with a faster alternative of the relatively expensive gelu activation function.

The first derivate of the polynomial gelu is continuous, while the second derivate has small jumps at the two border positions where we switch the polynomials.



