import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from jax.config import config
config.update("jax_enable_x64", True)
from functools import partial

@jax.jit
def akima_coeff(x, y):
    dy = jnp.diff(y)
    dx= jnp.diff(x)
    m=dy/dx
    m1 = jnp.atleast_1d(2.*m[0]-m[1])
    m0 = jnp.atleast_1d(2.*m1 - m[0])
    m_2 = jnp.atleast_1d(2*m[-1] - m[-2])
    m_1 = jnp.atleast_1d(2*m_2 - m[-1])

    key1 = jax.random.PRNGKey(1234)
    m=jnp.concatenate([m0, m1, m, m_2, m_1])+1e-12*jax.random.uniform(key1, shape=(m.size+4,))
    dm = jnp.sqrt(jnp.diff(m)**2)
    f1 = dm[2:]
    f2 = dm[:-2]
    f12 = f1 + f2
    t = (f1*m[1:-2] + f2*m[2:-1])/f12
    p2 = (3*(dy/dx) - 2*t[:-1] - t[1:])/dx
    p3 = (t[:-1] + t[1:] - 2*dy/dx)/dx**2
    areas = jnp.cumsum(jnp.concatenate([jnp.zeros(1), y[:-1]*dx + t[:-1]*dx**2/2 + p2*dx**3/3 + p3*dx**4/4]))

    return t, p2, p3,areas


@jax.jit
def interpolate(xx, x, y, akima):
    t, p2, p3, *_ = akima
    idx = jnp.searchsorted(x, xx, side='right')-1
    xmx1 = xx-x[idx]
    return y[idx] + t[idx]*xmx1 + p2[idx]*xmx1**2 + p3[idx]*xmx1**3

@jax.jit
def integrate(xx, x, y, akima):
    t, p2, p3, areas = akima
    idx = jnp.searchsorted(x, xx, side='right')-1
    xmx1 = xx-x[idx]
    return areas[idx]+y[idx]*xmx1 + t[idx]*xmx1**2/2 + p2[idx]*xmx1**3/3 + p3[idx]*xmx1**4/4

interpolate_v = jax.vmap(interpolate, (0, None, None, None))
integrate_v = jax.vmap(integrate, (0, None, None, None))

x=jnp.array([1., 1.5, 2.3, 4.])
y=jnp.array([2., 3., 3.5, 4.])

akima = akima_coeff(x,y)
print(interpolate(1., x, y, akima))
print(interpolate_v(jnp.array([1., 1.5, 2.3,  4.]), x, y, akima))
print(akima[3])
print(integrate(2.2, x, y, akima))
print(integrate_v(jnp.array([1., 1.5, 2.31, 3.99]), x,y,akima))