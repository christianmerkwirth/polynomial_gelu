import jax
import jax.numpy as jnp
import numpy as np

from matplotlib import pyplot as plt


def fast_gelu(x):
    x = x - 0.2279539373530299
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2

    y = 0.09631819 + 0.50 * x + 0.17 * x2 + 1.8e-10 * x3 - 3.2e-3 * x4
    y = jnp.where(x > -4.13791809, y, 0.0)
    y = jnp.where(x < 4.13964606264697, y, x)

    return y


# To make it compatible with JIT compilation (if needed)
fast_gelu_jit = jax.jit(fast_gelu)
# ALso compute the derivate.
fast_gelu_derivative = jax.jit(jax.vmap(jax.grad(fast_gelu)))

x = jnp.linspace(-5.0, 6.0, 1000)
y = fast_gelu(x)

# Plot the data
plt.subplot(2, 1, 1)
plt.plot(x, y, "b")
plt.plot(x, jax.nn.gelu(x), "r")
plt.grid("xy")
plt.legend(["Polynomial gelu", "gelu"])
plt.title("Polynomial gelu vs actual gelu")

plt.subplot(2, 1, 2)
plt.plot(x, fast_gelu_derivative(x), "r")
plt.plot(x[1:], (y[1:] - y[:-1]) / (x[1:] - x[:-1]), "b")
plt.grid("xy")
plt.title("First derivate of polynomial gelu")
plt.show()
