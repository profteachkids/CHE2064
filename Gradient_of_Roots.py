from dotmap import DotMap
import pandas as pd

import jax
from jax.lax import stop_gradient
import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)  # JAX default is 32bit single precision
from tools.tree_array_transform2 import VSC, Comp, Range
import tools.che as che

R = 8.314  # J/(mol K)

from scipy.optimize import root


def func(x, c):
    return jnp.array(
        [
            jnp.sin(x[0]) + c[0] * (x[0] - x[1]) ** 3 + c[1],
            c[2] * (x[1] - x[0]) ** 3 + x[1],
        ]
    )


def func2(x, c):
    return jnp.sum(func(x, c) ** 2)


c = jnp.array([0.5, -1.0, 0.5])

res = root(lambda x: func(x, c), (0.1, 0.1))


def dx(f, x, c):
    g = jax.grad(f, 0)
    H = jax.hessian(f, 0)
    return -jnp.linalg.inv(H(x, c)) @ g(x, c)


print(res.x)
dxdc = jax.jacobian(dx, 2)(func2, res.x, c)
dc = jnp.array([0.01, -0.01, 0.01])
print(res.x + dxdc @ dc)

c = c + dc
res = root(lambda x: func(x, c), (0.1, 0.1))
print(res.x)
