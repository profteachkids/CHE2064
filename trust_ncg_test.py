from trust_ncg import minimize
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

@jax.jit
def rosenbrock(x):
    res = jnp.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:1])**2)
    return res

def mccormick(v):
    res = jnp.sin(v[0]+v[1]) + (v[0]-v[1])**2 - 1.5*v[0] + 2.5*v[1] +1
    return res

jnp.set_printoptions(precision=8, linewidth=200)

rng = jax.random.PRNGKey(12)
guess = jax.random.uniform(rng,shape=(6,), minval=0., maxval=3.)
x_min, f_min = minimize(rosenbrock, guess, max_iter=500, rel_tol=1e-9, verbose=True)

x_min, f_min = minimize(mccormick, jnp.array([0., 0.]), max_iter=500, rel_tol=1e-10, verbose=True)
print(x_min, f_min)