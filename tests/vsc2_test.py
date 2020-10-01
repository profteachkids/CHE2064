from tools.tree_array_transform import flatten, unflatten, VSC, Range, Comp, nan_like, replace_not_nan
from dotmap import DotMap
import numpy as np
import jax.numpy as jnp
import jax

def model(c,r):
    pass

c=DotMap()
c.Ftot=10 # Total Feed moes
c.Fz = jnp.array([1/3, 1/3, 1/3]) # Equimolar feed composition
c.FT = 450 # Feed temperature
c.flashP= 101325 # Flash drum pressure

c.Vy = Comp([1/4,1/4,1/2]) # Guess vapor/liquid composition equal to feed
c.Lx = Comp([1/4,1/2,1/4]) # Comp - constrains mole fractions to behave like mole fractions!
c.flashT = Range(360, 273.15, c.FT)  # Guess and bounds for flash temperature
c.Vtot = Range(c.Ftot/1.5, 0., c.Ftot)  # Guess half of feed in vapor

vsc = VSC(c,model)
print(vsc.xtoc(vsc.x))
print(vsc.xtov(vsc.x))