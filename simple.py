import sys

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from numpy.random import default_rng
from itertools import combinations

from dao import er_dag, corr, simulate
from vtet import Wishart, Bollen_Ting, CCA, ARK


reps = 1000
n = 500

sp = 0.5
resample = True
frac = 0.9

rng = default_rng()

err = lambda *x: rng.normal(0, np.sqrt(x[0]), x[1])
# err = lambda *x: rng.gumbel(0, np.sqrt(6.0) * np.sqrt(x[0]) / np.pi, x[1])
# err = lambda *x: rng.exponential(np.sqrt(x[0]), x[1])
# err = lambda *x: rng.laplace(0, np.sqrt(x[0]) / np.sqrt(2), x[1])
# err = lambda *x: rng.uniform(-np.sqrt(3) * np.sqrt(x[0]), np.sqrt(3) * np.sqrt(x[0]), x[1])

g = np.zeros([6, 6], dtype=np.uint8)

g[0, 4] = 1
g[1, 4] = 1
g[2, 4] = 1
g[3, 4] = 1

alt = int(sys.argv[1])
if alt: g = er_dag(p=4, d=1)

obs = (0, 1, 2, 3)

# names = ["Wishart", "Bollen-Ting", "Bollen-Ting-F", "CCA", "ARK-1", f"ARK-{sp}"]
names = ["Wishart", "Bollen-Ting", "CCA", "ARK-1", f"ARK-{sp}"]
p_values = {name: [] for name in names}

for _ in range(reps):

    _, B, O = corr(g)
    df = pd.DataFrame(simulate(B, O, n, err))
    # tests = [Wishart(df), Bollen_Ting(df), Bollen_Ting(df), CCA(df), ARK(df, 1), ARK(df, sp)]
    tests = [Wishart(df), Bollen_Ting(df), CCA(df), ARK(df, 1), ARK(df, sp)]

    tets = []
    for iX in combinations(obs, 2):
        if 0 not in iX: continue
        iY = tuple([b for b in obs if b not in iX])
        tets.append((iX, iY))

    for i, test in enumerate(tests):
        # if i == 1: p_value = test.pure_triple(S=obs[:-1], v=obs[-1], fisher=False, resample=resample, frac=frac)
        # else: p_value = test.pure_triple(S=obs[:-1], v=obs[-1], fisher=True, resample=resample, frac=frac)
        p_value = test.tetrad(tet=tets[0], resample=False)
        p_values[names[i]].append(p_value)

df = pd.DataFrame.from_dict(p_values)

fig = px.ecdf(p_values, title=f"P-Values: Simple Model {'HA' if alt else 'H0'} --- n = {n}")
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="green", dash="dash"), name="Uniform"))
fig.add_vline(x=0.05, line_width=2, line_dash="dot", line_color="red")
fig.show()
