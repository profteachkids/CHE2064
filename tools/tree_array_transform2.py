import jax.numpy as jnp
import jax
from dotmap import DotMap
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint
from jax.config import config
from copy import deepcopy
from scipy.sparse import coo_matrix
config.update("jax_enable_x64", True)
EPS = jnp.finfo(jnp.float64).resolution


class VSC():
    def __init__(self, c, model):
        self.model = model
        self.c = c.toDict() if isinstance(c,DotMap) else c
        self.c_flat, self.idx, self.shapes, self.tree = flatten(self.c)
        self.r = DotMap()
        self.rdf = None
        self.v = None
        self.vdf = None
        self.sdf = None
        self.cdf = None
        self.v_id, self.nan_var = make_nan_variables(self.c)
        self.nan_var_flat, *_ = flatten(self.nan_var)
        self.update_idx = jnp.where(jnp.isnan(self.nan_var_flat))
        self.x = self.c_flat[self.update_idx]
        self.v_flat = jnp.nan * self.c_flat
        # self.test=self.model(self.xtoc(self.x))

    def xtoc(self,x):
        c = self.c_flat.at[self.update_idx].set(x)
        return DotMap(unflatten(c, self.idx, self.shapes, self.tree))

    def xtov(self,x):
        v_c = self.v_flat.at[self.update_idx].set(x)
        v_tree= unflatten(v_c, self.idx, self.shapes, self.tree)
        return DotMap(remove_nan(v_tree))

    def transform(self, model):
        def model_f(x):
            res = model(DotMap(self.xtoc(x)))
            if isinstance(res,tuple):
                res=res[0]
            return jnp.squeeze(res)
        return model_f

    def solve(self, jit=True, verbosity=1, sparse=False):
        def constraints(x):
            res = self.model(DotMap(self.xtoc(x)))
            eq = jnp.array([])
            if type(res) is tuple:
                if type(res[0]) is list:
                    for i in range(len(res[0])):
                        eq=jnp.append(eq,jnp.atleast_1d(res[0][i]))
                else:
                    eq=jnp.append(eq,jnp.atleast_1d(res[0]))
            else:
                if type(res) is list:
                    for i in range(len(res)):
                        eq=jnp.append(eq,jnp.atleast_1d(res[i]))
                else:
                    eq=jnp.append(eq,jnp.atleast_1d(res))
            return eq

        if jit:
            constraints = jax.jit(constraints)

        jac = jax.jacobian(constraints)
        if jit:
            jac= jax.jit(jac)

        def sparse_jac(x):
            jac_x = jac(x)
            idx = jnp.where(jac_x>1e-11)
            return coo_matrix((jac_x[idx], idx))


        if sparse:
            nlc = NonlinearConstraint(constraints, 0.,0., jac=sparse_jac)
        else:
            nlc = NonlinearConstraint(constraints, 0.,0., jac=jac)


        def cb(xk):
            if verbosity > 0:
                print (constraints(xk))

        bounds = [(-25.,25.)]*self.x.size

        res = minimize(lambda x: 0., self.x, method='SLSQP', bounds=bounds, constraints=nlc, callback=cb,
                       tol=1e-8, options=dict(maxiter=1000))

        if verbosity > 1:
            print(res)
            print(self.model(DotMap(self.xtoc(res.x))))
        self.x = res.x
        self.v = self.xtov(self.x)
        self.generate_reports(res.x)

    def generate_reports(self,x):
        self.vdf=todf(self.xtov(x))
        c=self.xtoc(x)
        res = self.model(c)
        if type(res) is tuple:
            self.r = res[1]
            self.rdf= todf(self.r).fillna('')
        self.cdf=todf(c)
        self.sdf=self.cdf.loc[list(set(self.cdf.index) - set(self.vdf.index))]
        return

def make_nan_variables(d):
    d = d.toDict() if isinstance(d,DotMap) else d
    dd = deepcopy(d)
    v_list = []
    for (k,v), (dk, dv) in zip(dd.items(), d.items()):
        if isinstance(v,dict):
            make_nan_variables(v)
        elif isinstance(v,Comp):
            dd[k]=Comp(jnp.nan*jnp.ones_like(v.x))
            v_list.append(id(dv))
        elif isinstance(v,Range):
            dd[k]=Range(jnp.nan, 0.,1.)
            v_list.append(id(dv))
        elif isinstance(v,RangeArray):
            dd[k]=RangeArray(jnp.nan * jnp.ones_like(v.x), 0.,1.)
            v_list.append(id(dv))
    return v_list, dd



def todf(tree):
    res={}
    tuple_keys(tree, res)
    return pd.DataFrame.from_dict(res).transpose().fillna('')

__sizes=[[(f'vector{i}', f'{j}') for j in range(1,i+1)] for i in range(1,100)]
# __sizes[0]=('','Value')
def tuple_keys(orig, flat={}, path=(), sizes=__sizes):

    def process(v, label):
        if type(v) in (tuple,list,dict, DotMap):
            tuple_keys(v, flat, tuple(path) + (label,))
        else:
            v = v.val if isinstance(v, jax.interpreters.ad.JVPTracer) else v
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

def nan_like(x, f=None):
    x = x.toDict() if isinstance(x,DotMap) else x
    values, treedef = jax.tree_flatten(x)
    def none(val):
        if isinstance(val,jnp.ndarray):
            return jnp.nan*jnp.empty_like(val)
        return float('nan')

    if f is None:
        val_none = map(none,values)
    else:
        val_none = map(f, values)
    return jax.tree_unflatten(treedef,val_none)

def flatten(pytree):
    vals, tree = jax.tree_flatten(pytree)
    shapes = [jnp.atleast_1d(val).shape for val in vals]
    vals2 = [jnp.atleast_1d(val).reshape([-1,]) for val in vals] # convert scalars to array to allow concatenation
    v_flat = jnp.concatenate(vals2)
    idx = list(jnp.cumsum(jnp.array([val.size for val in vals2])))
    return v_flat, idx, shapes, tree

def unflatten(x, idx, shapes, tree):
    return jax.tree_unflatten(tree, [(lambda item, shape: jnp.squeeze(item) if shape==(1,) else item.reshape(shape))(item,shape)
                                     for item,shape in zip(jnp.split(x,idx[:-1]), shapes)])

def replace_not_nan(a,b):
    a = a.toDict() if isinstance(a,DotMap) else a
    b = b.toDict() if isinstance(b,DotMap) else b
    a_flat, idx, shapes, tree = flatten(a)
    b_flat, *_ = flatten(b)
    c_flat = jnp.where(jnp.logical_not(jnp.isnan(b_flat)), b_flat, a_flat)
    return DotMap(unflatten(c_flat, idx, shapes, tree))

def merge(a, b, all = True):
    a = a.toDict() if isinstance(a,DotMap) else a
    b = b.toDict() if isinstance(b,DotMap) else a
    for key in b:
        b_value = b[key]
        if key not in a:
            a[key] = b_value
        elif isinstance(a[key], dict) and isinstance(b_value, dict):
            merge(a[key], b_value, all)
        elif all:
            a[key] = b_value

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
        return jnp.concatenate((jnp.atleast_1d(xm1), jnp.atleast_1d(1.-jnp.sum(xm1))))


jax.tree_util.register_pytree_node(Comp, Comp.flatten, Comp.unflatten)

class RangeArray():
    def __init__(self,x, lo, hi):
        self.x=x
        self.lo = lo
        self.diff = hi-lo
        if jnp.any(self.diff <= 0.) or jnp.any(self.x<lo) or jnp.any(self.x>hi):
            raise ValueError('Hi > x > Lo is required')

    def __repr__(self):
        return f'{self.x}, lo={self.lo}, diff={self.diff}'

    @staticmethod
    def flatten(v):
        p = (v.x-v.lo)/v.diff
        return (jnp.log(p)-jnp.log(1.-p),), (v.lo,v.diff)

    @staticmethod
    def unflatten(aux, f):
        f=jnp.squeeze(jnp.asarray(f))
        return jax.nn.sigmoid(f)*aux[1]+aux[0]

jax.tree_util.register_pytree_node(RangeArray, RangeArray.flatten, RangeArray.unflatten)


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
