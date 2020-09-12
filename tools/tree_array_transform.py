import jax.numpy as jnp

import jax
from dotmap import DotMap
import pandas as pd
from jax.config import config
config.update("jax_enable_x64", True)

class VSC():
    def __init__(self,v,s):
        self.v = v.toDict() if isinstance(v,DotMap) else v
        self.s = s.toDict() if isinstance(s,DotMap) else s
        self.c = {}
        merge(self.c,s)
        merge(self.c,v)
        self.c_flat, self.idx, self.shapes, self.tree = flatten(self.c)

        self.v_tree = nan_like(self.c)
        merge(self.v_tree,v)
        self.v_flat, *_ = flatten(self.v_tree)
        self.update_idx = jnp.where(jnp.logical_not(jnp.isnan(self.v_flat)))

        self.x = self.v_flat[self.update_idx]

    def xtoc(self,x):
        c = self.c_flat.at[self.update_idx].set(x)
        return DotMap(unflatten(c, self.idx, self.shapes, self.tree))

    def xtov(self,x):
        v_c = self.v_flat.at[self.update_idx].set(self.x)
        v_tree= unflatten(v_c, self.idx, self.shapes, self.tree)
        return DotMap(remove_nan(v_tree))

    def transform(self,model):
        def model_f(x):
            return jnp.squeeze(model(self.xtoc(x)))
        return model_f

def todf(tree):
    res={}
    tuple_keys(tree, res)
    return pd.DataFrame.from_dict(res).transpose().fillna('')

__sizes=[[(f'vector{i}', f'{j}') for j in range(1,i+1)] for i in range(1,10)]
# __sizes[0]=('','Value')
def tuple_keys(orig, flat={}, path=(), sizes=__sizes):

    def process(v, label):
        if type(v) in (tuple,list,dict):
            tuple_keys(v, flat, tuple(path) + (label,))
        else:
            v=jnp.atleast_1d(v)
            if not(jnp.all(jnp.isnan(v))):
                size = v.size
                flat[tuple(path) + (label,)]={sizes[size-1][i]:value for i,value in enumerate(v)}

    t = type(orig)
    if t in (dict, DotMap):
        orig = orig.toDict() if isinstance(orig,DotMap) else orig
        for k,v in orig.items():
            process(v,k)
    elif t in (tuple,list):
        for count,v in enumerate(orig):
            process(v,count)
    return flat

def remove_nan(orig):
    t = type(orig)
    clean=t()
    if t is dict:
        for k,v in orig.items():
            if type(v) in (tuple,list,dict):
                cleaned = remove_nan(v)
                if len(cleaned) > 0:
                    clean[k]=remove_nan(v)
            elif not(jnp.all(jnp.isnan(jnp.atleast_1d(v)))):
                clean[k]=v
    elif t in (tuple,list):
        clean=[]
        for v in orig:
            if type(v) in (tuple,list,dict):
                cleaned = remove_nan(v)
                if len(cleaned) > 0:
                    clean.append(remove_nan(v))
            elif not(jnp.all(jnp.isnan(jnp.atleast_1d(v)))):
                clean.append(v)

    return clean

def nan_like(x):
    values, treedef = jax.tree_flatten(x)
    def none(val):
        if isinstance(val,jnp.ndarray):
            return jnp.nan*jnp.empty_like(val)
        return float('nan')
    val_none = map(none,values)
    return jax.tree_unflatten(treedef,val_none)

def flatten(pytree):

    def val_shape_type(val):
        return jnp.atleast_1d(val).reshape([-1]), jnp.atleast_1d(val).shape, type(val)

    vals, tree = jax.tree_flatten(pytree)
    shapes = [jnp.atleast_1d(val).shape for val in vals]
    vals2 = [jnp.atleast_1d(val).reshape([-1,]) for val in vals] # convert scalars to array to allow concatenation
    v_flat = jnp.concatenate(vals2)
    idx = jnp.cumsum(jnp.array([val.size for val in vals2]))
    return v_flat, idx, shapes, tree

def unflatten(x, idx, shapes, tree):
    return jax.tree_unflatten(tree, [item.reshape(shape) for item,shape in zip(jnp.split(x,idx[:-1]), shapes)])

def merge(a, b):
    a = a.toDict() if isinstance(a,DotMap) else a
    b = b.toDict() if isinstance(b,DotMap) else a
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key])
        a[key] = b[key]

class Comp():
    def __init__(self,x):
        self.x=jnp.asarray(x).reshape(-1)
        if self.x.size<2:
            raise ValueError('At least 2 components required')

    def __repr__(self):
        return f'{self.x}'

    @staticmethod
    def flatten(c):
        return jnp.log(c.x[:-1]) + jnp.log(1.+ (1. - c.x[-1])/c.x[-1]), None


    @staticmethod
    def unflatten(aux, q):
        q=jnp.squeeze(jnp.asarray(q)) #q may be a tuple that can't be squeezed
        xm1 = jnp.exp(q)/(1+jnp.sum(jnp.exp(q)))
        return jnp.concatenate((xm1, jnp.atleast_1d(1.-jnp.sum(xm1))))


jax.tree_util.register_pytree_node(Comp, Comp.flatten, Comp.unflatten)

class Range():
    def __init__(self,x, lo, hi):
        self.x=x
        self.lo = lo
        self.diff = hi-lo
        if self.diff <= 0. or self.x<lo or self.x>hi:
            raise ValueError('Hi > x > Lo is required')

    def __repr__(self):
        return f'{self.x}, lo={self.lo}, diff={self.diff}'

    @staticmethod
    def flatten(v):
        p = (v.x-v.lo)/v.diff
        return (jnp.log(p)-jnp.log(1.-p),), (v.lo,v.diff)

    @staticmethod
    def unflatten(aux, f):
        return jax.nn.sigmoid(f[0])*aux[1]+aux[0]

jax.tree_util.register_pytree_node(Range, Range.flatten, Range.unflatten)