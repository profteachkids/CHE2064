from tools.tree_array_transform import flatten, unflatten, todf, VSC, Range, Comp, nan_like, replace_not_nan
from dotmap import DotMap
import numpy as np
import jax.numpy as jnp
import jax


s=DotMap()
s.state_list_values=[1, 2., 3.]
s.list_DotMap = [DotMap(a=1, b=2), DotMap(c=jnp.array([ [3., 4.] ,[1., 3.5]]), d=3)]

v=DotMap()
v.a=[jnp.array([[1., 2.],[3.,4.]])]*2

c = VSC(v,s)

def test_vsc():
    cc = c.xtoc(c.x)
    assert(jnp.all(cc.a[1] == v.a[1]))
    assert(cc.list_DotMap[1].c[1,1]==3.5)

def test_array_element_variable():
    v.list_DotMap=[]



class Custom1():
    def __init__(self, x, name=None, arr=None, dict=None, dotmap=None, alist=None, f=None):
        self.x = x
        self.name = name
        self.arr = arr
        self.dict = dict
        self.dotmap = dotmap
        self.alist= alist
        self.f = f
        
    def process(self, y):
        return y* self.f(self.x, self.name, self.arr, self.dict, self.dotmap, self.alist)
        
    @staticmethod
    def flatten(c):
        return (c.x,), (c.name, c.arr, c.dict, c.dotmap, c.alist, c.f)

    @staticmethod
    def unflatten(aux, x):
        return Custom1(x[0], *aux)

jax.tree_util.register_pytree_node(Custom1, Custom1.flatten, Custom1.unflatten)

def my_f(x, name, arr, dict, dotmap, alist):
    return x

# s=DotMap()
# s.a = Custom1(3, 'statone', jnp.array([1.,2.]), dict(a=1, b=3), DotMap(c=4, f=5), ['hello', jnp.array([4,5])], my_f)
# s.a=Custom1(3)
# s.a = Comp([1/3, 1/3, 1/3])
#
# v=DotMap()
# v.b = Custom1(8, 'varone', jnp.array([3.,1.]), dict(a=3, b=2), DotMap(c=7, f=5), ['hi', jnp.array([41,52])], my_f)
# # v.b = Range( 50., 0., 100.)
# v.a=Custom1(8,name='varone', f=my_f)
# v.c=Comp([1/3, 1/3, 1/3])

# v_flat, idx, shapes, tree=  flatten(v.toDict())
# cc = DotMap(unflatten(v_flat, idx, shapes, tree))

# c = VSC(v,s)

# def f(val):
#     return type(val)
#
# cc = nan_like(c.c, f)

v_nan = DotMap(nan_like(v))
v_nan.a[1]=v_nan.a[1].tolist()
v_nan.a[1][1][1]=Range(10.,0.,20.)
v_replaced = replace_not_nan(v,v_nan)
print(v_replaced)

v=DotMap()
v.a = Range(100,0,200)
v=v.toDict()
flat, idx, shapes, tree = flatten(v)
vv = unflatten(flat, idx, shapes, tree)