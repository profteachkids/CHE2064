
from dotmap import DotMap
import pandas as pd
import jax
from jax.lax import stop_gradient
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True) #JAX default is 32bit single precision
from tools.tree_array_transform import VSC, Comp, Range
import tools.che as che
R=8.314 # J/(mol K)

#%%

p = che.Props(['Nitrogen','Oxygen', 'Argon', 'Water'])


def model2(c,r):
    r.Pw = p.Pvap(c.T)[3]
    r.V_vap = c.V_tot - c.Vw_i # Approximation - water in the vapor phase is negligible
    r.air_n = c.P_i * r.V_vap / (R * c.T_i)

    r.W_n_vap = r.Pw * r.V_vap / (R * c.T)
    r.P = r.air_n  * R * c.T / r.V_vap + r.Pw

    # tuples from left and right side of equations
    P_constraint = (c.P_f, r.P)
    W_n_constraint = (r.W_n_vap, c.W_n_vap_desired)
    return (P_constraint, W_n_constraint)

#%%

c=DotMap()
c.W_tot = 1. # 1 kg
c.V_tot = 0.01 # 10 Liters
c.P_i = 1e5 # Pa air pressure
c.P_f = 2e5 # Pa final pressure
c.T_i = 298.
c.W_n_vap_desired = 0.3

c.Vw_i = c.W_tot/p.rhol(c.T_i)[3]
c.T = Range(350, 300, 400) #final temperature
c.V_tot = Range(0.015, 0., 0.03) # total container volume

vsc=VSC(c,model2)
vsc.solve(verbosity=0, jit=True)