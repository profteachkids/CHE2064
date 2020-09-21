import jax.numpy as jnp
import jax
from jax.config import config
from scipy.optimize import minimize as scipy_minimize
config.update("jax_enable_x64", True)

def minimize(func, guess, verbosity=1):

    @jax.jit
    def hvp(x,p):
        return jax.grad(lambda x: jnp.vdot(jax.grad(func)(x),p))(x)

    def cb(xk, state):
        if verbosity > 0:
            print (state.fun)

    bounds = [(-25.,25.)]*guess.size

    res = scipy_minimize(func, guess, method='trust-constr', bounds=bounds,jac=jax.grad(func), hessp=hvp, callback=cb)
    return res.x, res.fun
@jax.jit
def get_boundaries_intersections(z, d, trust_radius):
    a = jnp.vdot(d, d)
    b = 2 * jnp.vdot(z, d)
    c = jnp.vdot(z, z) - trust_radius**2
    sqrt_discriminant = jnp.sqrt(b*b - 4*a*c)
    ta = (-b - sqrt_discriminant) / (2*a)
    tb = (-b + sqrt_discriminant) / (2*a)
    return jnp.sort(jnp.stack([ta, tb]))

def minimize_custom(func, guess, trust_radius = 1., max_trust_radius=100., grad_tol=1e-6, abs_tol=1e-10, rel_tol=1e-6,
             max_iter = 100, max_cg_iter = 100, verbosity=1):

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

        if verbosity>0:
            print(f'{trust_iter}:{cg_iter}, f: {f}')
        if verbosity>1:
            print(f'x: {x}')
            print(f'grad: {grad}')
            print(f'dx: {p_boundary}')
            print()
        if (jnp.max(jnp.abs(p_boundary/x)) < rel_tol) or (jnp.max(jnp.abs(p_boundary)) < abs_tol):
            break

    if verbosity>0:
        print('Final results:')
        print(f'f: {f}')
        print(f'x: {x}')
    if verbosity>1:
        print(f'grad: {grad}')
        print(f'dx: {p_boundary}')
    return x, f

