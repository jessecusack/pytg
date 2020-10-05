# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: pytg
#     language: python
#     name: pytg
# ---

# %% [markdown]
# # Solve the TG equation step by step

# %%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import findiff as fd

# load data
dat = np.loadtxt("Nash_data.txt", comments="%")

v = dat[:, 0]
rho = 1000.*dat[:, 1]
z = dat[:, 2]
g = 9.81
rho0 = np.mean(rho)
b = -g * (rho/rho0 - 1)

# %%
fig, axs = plt.subplots(1, 3, sharey=True)
axs[0].plot(v, z)
axs[1].plot(rho, z)
axs[0].set_ylabel("$z$ (m)")
axs[0].set_xlabel("$v$ (m s$^{-1}$)")
axs[1].set_xlabel(r"$\rho$ (kg m$^{-3}$)")
axs[2].plot(b, z)
axs[2].set_xlabel("$b$ (m s$^{-2}$)")

# %% [markdown]
# Try the module.

# %%
import pytg.TG as TG

dat = np.loadtxt("Nash_data.txt", comments="%")
u = dat[:, 0]
rho = 1000.*dat[:, 1]
z = dat[:, 2]
g = 9.81
rho0 = np.mean(rho)
b = -g * (rho/rho0 - 1)
Kv = 1.0e-3
Kb = Kv/7
# Wavenumber
k = 1e-4
l = 0.

om, wvec, bvec = TG.vTG(z, u, u*0, b, k, l, Kv, Kb, BCv_upper="rigid", BCv_lower="rigid", BCb_upper="constant", BCb_lower="constant")
cp = -om.imag/k

fig, ax = plt.subplots(1, 1)
ax.plot(cp, '.')
ax.set_xlabel("Mode")

fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].plot(wvec[:, -1], z)
axs[1].plot(bvec[:, -1], z)
axs[0].plot(wvec[:, 0], z)
axs[1].plot(bvec[:, 0], z)
