from tools.che import vsc
from dotmap import DotMap
import jax.numpy as jnp

s=DotMap()
s.state_list_values=[1, 2., 3.]
s.state_list_DotMap = [DotMap(a=1, b=2), DotMap(c=1., d=4.)]

v=DotMap()
v.a=[jnp.array([[1., 2.],[3.,4.]])]*2

c= vsc.VSC(v, s)
print(c.x)
print(c.xtoc(c.x))
