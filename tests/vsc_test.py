from tools.tree_array_transform import flatten, unflatten, todf, VSC, Range, Comp
from dotmap import DotMap
import jax.numpy as jnp

s=DotMap()
s.state_list_values=[1, 2., 3.]
s.state_list_DotMap = [DotMap(a=1, b=2), DotMap(c=jnp.array([ [3., 4.] ,[1., 3.5]]), d=3)]

v=DotMap()
v.a=[jnp.array([[1., 2.],[3.,4.]])]*2

c = VSC(v,s)

def test_vsc():
    cc = c.xtoc(c.x)
    assert(jnp.all(cc.a[1] == v.a[1]))
    assert(cc.state_list_DotMap[1].c[1,1]==3.5)

