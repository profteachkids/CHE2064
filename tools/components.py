from tools.tree_array_transform import flatten, unflatten, todf, VSC, Range, Comp
from dotmap import DotMap
import jax.numpy as jnp
from tools.trust_ncg import minimize
import tools.che as che

p = che.Props(['Ethanol','Water'])
class Model():

    def __init__(self):
        self.left=[]
        self.right=[]
        self.r = DotMap()
        self.DOF = 0

    def stream(self,u, total, frac):
        u.total = total
        u.frac = frac
        self.DOF+=p.N_comps


    def unit(self, instreams, outstreams):
        def total_in():
            return sum(stream.total*stream.frac for stream in instreams)

        def total_out():
            return sum(stream.total*stream.frac for stream in outstreams)

        self.left.append(total_in)
        self.right.append(total_out)
        self.DOF -= p.N_comps


    def sumsqerr(self):
        return jnp.linalg.norm([(l()-r())/(l()+r()) for l,r in zip(self.left, self.right)])

s=DotMap()
m=Model()
for i in range(1,6):
    m.stream(s[f's{i}'], 1.5, jnp.array([1/3,1/3,1/3]))

m.unit([s.s1, s.s2, s.s3], [s.s4, s.s5])

for i in range(10,50,10):
    s.s1.total=1.*i
    print(m.sumsqerr())

