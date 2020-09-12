
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
from scipy.integrate import solve_ivp
from plotly.subplots import make_subplots
import plotly.io as pio
from dotmap import DotMap
import tools.tree_array_transform as tat

pio.templates.default='plotly_dark'

def switch(f_orig, start=0., end=jnp.inf, sharp=100):
    def f(t,*args, **kwargs):
        return (jax.nn.sigmoid(sharp*(t-start)) - jax.nn.sigmoid(sharp*(t-end)))*f_orig(t-start, *args, **kwargs)
    return f

def model(t, v, s, dv):
    def q1in():
        return jnp.array([0.1 + switch(lambda t: 0.1, 100, 200)(t),
                          0.2 + switch(lambda t: 0.1, 10, 100)(t),
                          0.3 + switch(lambda t: 0.1, 50, 150)(t)])


    V1 = jnp.sum(v.m1/s.rho)
    rho1 = jnp.sum(v.m1)/ V1
    q1out=s.Cv1*jnp.sqrt(V1/s.A1)

    w1 = v.m1/jnp.sum(v.m1)
    dv.m1=jnp.dot(s.rho, q1in()) - rho1*q1out * w1
    return dv

s=DotMap()
s.A1 = 0.5
s.Cv1 = 0.1
s.rho = jnp.array([1000., 900., 800.])

v=DotMap()
v.m1 = jnp.array([10., 10., 10.])


m = tat.VX(v, s)
tend=300.
model_f = m.transform(model)

res = solve_ivp(model_f, (0.,tend), m.x, method='Radau', dense_output=True, jac=jax.jacfwd(model_f,1))
t=jnp.linspace(0,tend,100)
h1 = res.sol(t)[0]
h2 = res.sol(t)[1]

# fig=make_subplots()
# fig.add_scatter(x=t, y=h1, mode='lines', name='h1')
# fig.add_scatter(x=t, y=h2, mode='lines', name='h2')
# fig.show()
