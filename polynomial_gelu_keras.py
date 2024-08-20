from keras import ops
from keras import layers


class FastGELUActivation(layers.Layer):
    def __init__(self, **kwargs):
        super(FastGELUActivation, self).__init__(**kwargs)

    def call(self, x):
        x = x - 0.2279539373530299
        x2 = x * x
        x3 = x2 * x
        x4 = x2 * x2

        y = 0.09631819 + 0.50 * x + 0.17 * x2 + 1.8e-10 * x3 - 3.2e-3 * x4
        y = ops.where(x > -4.13791809, y, 0.0)
        y = ops.where(x < 4.13964606264697, y, x)

        return y
