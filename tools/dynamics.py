import jax
import jax.numpy as jnp
from dotmap import DotMap
import copy
from tools.tree_array_transform import flatten, unflatten
from jax.config import config
config.update("jax_enable_x64", True)

class VX():
    def __init__(self,v, s=None):
        self.v = v.toDict() if isinstance(v,DotMap) else v
        self.dv = DotMap(copy.deepcopy(self.v))
        self.x, self.idx, self.shapes, self.tree = flatten(self.v)
        self.s = s

    def xtov(self,x):
        return DotMap(unflatten(x, self.idx, self.shapes, self.tree))

    def vtox(self,v):
        self.x, *_ = flatten(v)
        return(self.x)

    def soltov(self, sol):
        nt = sol.shape[-1]
        shapes = [shape + (nt,) for shape in self.shapes]
        return DotMap(unflatten(sol, self.idx, shapes, self.tree))

    def transform(self, model):
        if self.s is None:
            def model_f(t, x):
                return jnp.squeeze(self.vtox(model(t, self.xtov(x), self.dv).toDict()))
        else:
            def model_f(t, x):
                dx = self.vtox( model(t, self.xtov(x), self.s, self.dv).toDict())
                return jnp.squeeze(dx)

        return model_f


def onoff(f_orig, start=0., end=jnp.inf, sharp=100):
    def f(t,*args, **kwargs):
        return (jax.nn.sigmoid(sharp*(t-start)) - jax.nn.sigmoid(sharp*(t-end)))*f_orig(t-start, *args, **kwargs)
    return f

def onoff_val(value, start=0., end=jnp.inf, sharp=100):
    return onoff(lambda t: value, start, end, sharp)

def switchhold(f_orig, start, hold, sharp=100):
    def f(t, *args, **kwargs):
        return (onoff(f_orig, start=start, sharp=sharp)(t, *args, **kwargs) -
                onoff(f_orig, start=start+hold, sharp=sharp)(t, *args, **kwargs))
    return f

def ramp(delta, t_start, t_end):
    return switchhold(lambda t: delta*t/(t_end-t_start), t_start, t_end-t_start)
