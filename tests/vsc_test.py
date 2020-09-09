import vsc
from dotmap import DotMap
import che_tools as che
import jax
import jax.numpy as jnp
import numpy as np

s=DotMap()
s.state_list_values=[1, 2., 3.]
s.state_list_DotMap = [DotMap(a=1, b=2), DotMap(c=1., d=4.)]

v=DotMap()
v.var_value=45.
v.var_list_values=[4., 5.]
v.var_list_DotMap = [DotMap(e=4, f=1), -2.]
v.comp=vsc.Comp([0.1, 0.2, 0.3, 0.4])
v.range=vsc.Range(20.,0.,100.)

c=vsc.VSC(v,s)

DotMap(c.c).pprint()

x=c.x.at[:3].set(che.xtoq([0.2,0.4,0.3,0.1]))
c.c = c.xtoc(x)


def f(c):
    return c.range**2

fx = c.transform(f)
print(fx(c.x))
print(jax.grad(fx)(c.x))
