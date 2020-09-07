import jax.numpy as jnp
import jax
from dotmap import DotMap
import pandas as pd
import copy
from jax.config import config
config.update("jax_enable_x64", True)

def tuple_keys(d,flat={},path=(),sizes=None):
    if sizes is None:
        sizes=[[(f'v{i}', f'x{j}') for j in range(1,i+1)] for i in range(1,10)]
    d = d.toDict() if isinstance(d,DotMap) else d
    for k,v in d.items():
        if isinstance(v,dict):
            tuple_keys(v, flat, tuple(path) + (k,))
        else:
            if not(jnp.all(jnp.isnan(v))):
                size = v.size
                flat[tuple(path) + (k,)]={sizes[size-1][i]:value for i,value in enumerate(v)}
    return

def remove_nan(orig):
    clean={}
    for k,v in orig.items():
        if isinstance(v,dict):
            cleaned = remove_nan(v)
            if len(cleaned) > 0:
                clean[k]=remove_nan(v)
        else:
            if not(jnp.all(jnp.isnan(v))):
                clean[k]=v
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
    vals, tree = jax.tree_flatten(pytree)
    vals2 = [jnp.atleast_1d(val).astype(jnp.float64) for val in vals]
    v_flat = jnp.concatenate(vals2)
    idx = jnp.cumsum(jnp.array([val.size for val in vals2]))
    return v_flat, idx, tree

def unflatten(x, idx, tree):
    return jax.tree_unflatten(tree, jnp.split(x,idx[:-1]))


def merge(a, b):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key])
        a[key] = b[key]


def transform(model, v, s):
    v = v.toDict() if isinstance(v,DotMap) else v
    s = s.toDict() if isinstance(s,DotMap) else s
    c = {}
    merge(c,s)
    merge(c,v)
    c_flat, idx, tree = flatten(c)
    c_flat = jnp.array(c_flat)

    v_tree = nan_like(c)
    merge(v_tree,v)
    v_flat, _, _ = flatten(v_tree)
    update_idx = jnp.where(jnp.logical_not(jnp.isnan(v_flat)))

    def model_f(x):
        c = c_flat.at[update_idx].set(x)
        c = unflatten(c,idx,tree)
        return jnp.squeeze(model(c))

    def transform_sol(x_min_array):
        c = c_flat.at[update_idx].set(x_min_array)
        c = unflatten(c,idx,tree)

        v_c = v_flat.at[update_idx].set(x_min_array)
        x_tree= unflatten(v_c, idx,tree)

        x_min = remove_nan(x_tree)
        res={}
        tuple_keys(x_min,res)
        df_x_min = pd.DataFrame(res).transpose().fillna('')

        res={}
        tuple_keys(c,res)
        df_c = pd.DataFrame(res).transpose().fillna('')

        return x_min, df_x_min, c, df_c, x_tree

    return model_f, v_flat[update_idx], transform_sol

@jax.jit
def get_boundaries_intersections(z, d, trust_radius):
    a = jnp.vdot(d, d)
    b = 2 * jnp.vdot(z, d)
    c = jnp.vdot(z, z) - trust_radius**2
    sqrt_discriminant = jnp.sqrt(b*b - 4*a*c)
    ta = (-b - sqrt_discriminant) / (2*a)
    tb = (-b + sqrt_discriminant) / (2*a)
    return jnp.sort(jnp.stack([ta, tb]))

def minimize(func, guess, trust_radius = 1., max_trust_radius=100., grad_tol=1e-6, abs_tol=1e-10, rel_tol=1e-6,
             max_iter = 100, max_cg_iter = 100, verbose=False):

    grad_f = jax.grad(func)
    x = guess
    p_boundary = jnp.zeros_like(x)
    hits_boundary = True
    for trust_iter in range(max_iter):
        z=jnp.zeros_like(x)
        f = func(x)
        grad = grad_f(x)
        if jnp.linalg.norm(grad) < grad_tol:
            break
        r = grad
        d = -r
        for cg_iter in range(max_cg_iter):
            Bd = jax.grad(lambda x: jnp.vdot(grad_f(x),d))(x)
            dBd = jnp.vdot(d, Bd)
            r_squared = jnp.vdot(r, r)
            alpha = r_squared / dBd
            z_next = z + alpha * d
            if jnp.linalg.norm(z_next) >= trust_radius:
                t = get_boundaries_intersections(z, d, trust_radius)
                p_boundary = z + t[1] * d
                hits_boundary = True
                break
            r_next = r + alpha * Bd
            if jnp.linalg.norm(r_next) < 1e-10:
                hits_boundary=False
                p_boundary = z_next
                break
            d_next = -r_next + d * jnp.vdot(r_next, r_next) / r_squared
            z=z_next
            r=r_next
            d=d_next

        x_proposed = x + p_boundary
        actual_reduction = f - func(x_proposed)

        grad_p_boundary = jnp.vdot(grad, p_boundary)
        Bp_boundary = jax.grad(lambda x: jnp.vdot(grad_f(x), p_boundary))(x)

        approx = f + grad_p_boundary + 0.5 * jnp.vdot(p_boundary, Bp_boundary)
        predicted_reduction = f - approx
        rho = actual_reduction / predicted_reduction

        if rho < 0.25:
            trust_radius = trust_radius*0.25
        elif rho > 0.75 and hits_boundary:
            trust_radius = min(2*trust_radius, max_trust_radius)
        if rho > 0.15:
            x = x_proposed

        if verbose:
            print(f'{trust_iter}:{cg_iter}, f: {f}')
            print(f'x: {x}')
            print(f'grad: {grad}')
            print(f'dx: {p_boundary}')
            print()
        if (jnp.max(jnp.abs(p_boundary/x)) < rel_tol) or (jnp.max(jnp.abs(p_boundary)) < abs_tol):
            break

    if verbose:
        print('Final results:')
        print(f'f: {f}')
        print(f'x: {x}')
        print(f'grad: {grad}')
        print(f'dx: {p_boundary}')
        print()
    return x, f

