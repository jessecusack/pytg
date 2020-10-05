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

# %%
# Fake profile

vf0 = 0.05
zmax = 0
zmin = -100
ndata = 101

zf = np.linspace(zmin, zmax, ndata)
vf = vf0*np.ones_like(zf)
bf = np.linspace(0.00, 0.02, ndata)


fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].plot(vf, zf)
axs[1].plot(bf, zf)

# %%
# Constants and extras
Kv = 1.0e-6
Kb = Kv/7
# Wavenumber
k = 1e-5
l = 0.

bi = b
vi = v
zi = z

# function [sig,w,b]=SSF(z,U,B,k,l,nu,kappa,iBC1,iBCN,imode)
# %
# % USAGE: [sig,w]=SSF(z,U,B,nu,kappa,k,l,iBC1,iBCN,imode)
# %
# % Stability analysis for a viscous, diffusive, stratified, parallel shear flow
# % INPUTS:
# % z = vertical coordinate vector (evenly spaced)
# % U = velocity profile
# % B = buoyancy profile (Bz=squared BV frequency)
# % k,l = wave vector 
# % nu, kappa = viscosity, diffusivity
# % iBC1 = boundary conditions at z=z(0)
# %    (1) velocity: 1=rigid (default), 0=frictionless
# %    (2) buoyancy: 1=insulating, 0=fixed-buoyancy (default)
# % iBCN = boundary conditions at z=z(N+1).
# %        Definitions as for iBC1. Default: iBCN=iBC1.
# % imode = mode choice (default imode=1)
# %         imode=0: output all modes, sorted by growth rate
# %
# % OUTPUTS:
# % sig = growth rate of FGM
# % w = vertical velocity eigenfunction
# % b = buoyancy eigenfunction
# %
# % CALLS:
# % ddz, ddz2, ddz4
# %
# % W. Smyth, OSU, Nov04


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Stage 1: Preliminaries
# %

# % check for equal spacing

# % defaults
# if nargin<8;iBC1=[1 0];end
# if nargin<9;iBCN=iBC1;end
# if nargin<10;imode=1;end

# % define constants
# ii=complex(0.,1.);
# del=mean(diff(z));
# N=length(z);
# kt=sqrt(k^2+l^2);

dz = zi[1] - zi[0]
N = vi.size
kh = np.sqrt(k**2 + l**2)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Stage 2: Derivative matrices and BCs
# %
# D1=ddz(z); %1st derivative matrix with 1-sided boundary terms
# Bz=D1*B;

# D2=ddz2(z); %2nd derivative matrix with 1-sided boundary terms
# Uzz=D2*U;

# First derivative
ddz = fd.FinDiff(0, dz, 1).matrix(vi.shape).toarray()
# Second derivative
d2dz2 = fd.FinDiff(0, dz, 2).matrix(vi.shape).toarray()

dbdz = ddz @ bi
d2udz2 = d2dz2 @ vi

fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].plot(dbdz, zi)
axs[1].plot(bi, zi)

# % Impermeable boundary
d2dz2[0, :] = 0
d2dz2[0, 0] = -2/dz**2
d2dz2[0, 1] = 1/dz**2
d2dz2[-1, :] = 0
d2dz2[-1, -1] = -2/dz**2
d2dz2[-1, -2] = 1/dz**2

# % Asymptotic boundary
# % D2(1,:)=0;
# % D2(1,1)=2*(-del*kt-1)/del^2;
# % D2(1,2)=2/del^2;
# % D2(N,:)=0;
# % D2(N,N)=2*(-del*kt-1)/del^2;
# % D2(N,N-1)=2/del^2;

# % Fourth derivative
d4dz4 = fd.FinDiff(0, dz, 4).matrix(vi.shape).toarray()

iBC1 = [0, 0]
iBCN = [0, 0]

# % Rigid or frictionless BCs for 4th derivative
d4dz4[0, :] = 0
d4dz4[0, 0] = (5+2*iBC1[0])/dz**4
d4dz4[0, 1] = -4/dz**4
d4dz4[0, 2] = 1/dz**4
d4dz4[1, :] = 0
d4dz4[1, 0] = -4/dz**4
d4dz4[1, 1] = 6/dz**4
d4dz4[1, 2] = -4/dz**4
d4dz4[1, 3] = 1/dz**4
d4dz4[-1, :] = 0
d4dz4[-1, -1] = (5+2*iBCN[0])/dz**4
d4dz4[-1, -2] = -4/dz**4
d4dz4[-1, -3] = 1/dz**4
d4dz4[-2, :] = 0
d4dz4[-2, -1] = -4/dz**4
d4dz4[-2, -2] = 6/dz**4
d4dz4[-2, -3] = -4/dz**4
d4dz4[-2, -4] = 1/dz**4

# % Derivative matrix for buoyancy
# D2b=ddz2(z); %2nd derivative matrix with 1-sided boundary terms
# % Fixed-buoyancy boundary
# D2b(1,:)=0;
# D2b(1,1)=-2/del^2;
# D2b(1,2)=1/del^2;
# D2b(N,:)=0;
# D2b(N,N)=-2/del^2;
# D2b(N,N-1)=1/del^2;

d2dz2b = d2dz2.copy()

# % Insulating boundaries
# if iBC1(2)==1
#     D2b(1,:)=0;
#     D2b(1,1)=-2/(3*del^2);
#     D2b(1,2)=2/(3*del^2);
# end

# if iBCN(2)==1
#     D2b(N,:)=0;
#     D2b(N,N)=-2/(3*del^2);
#     D2b(N,N-1)=2/(3*del^2);
# end

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Stage 3: Assemble stability matrices
# %
# % Laplacian and squared Laplacian matrices
# Id=eye(N);
# L=D2-kt^2*Id;
# Lb=D2b-kt^2*Id;
# LL=D4-2*kt^2*D2+kt^4*Id;

Id = np.eye(N)
L = d2dz2 - Id*kh**2  # Laplacian
Lb = d2dz2b - Id*kh**2  # Laplacian for buoyancy
LL = d4dz4 - 2*d2dz2*kh**2 + Id*kh**4  # Laplacian of laplacian

# N2=2*N;
# NP=N+1;
# % A=zeros(N2,N2);
# % B=zeros(N2,N2);

# A = np.zeros((2*N, 2*N))
# B = np.zeros((2*N, 2*N))

A = np.block([[L, np.zeros_like(L)], [np.zeros_like(L), Id]])

# % assemble matrix A
# A=[L Id*0 ; Id*0 Id];

# % Compute submatrices of B using Levi's syntax
# b11=-ii*k*diag(U)*L+ii*k*diag(Uzz)+nu*LL;
# b21=-diag(Bz);
# b12=-kt^2*Id;
# b22=-ii*k*diag(U)+kappa*Lb;

b11 = -1j*k*np.diag(vi)@L + 1j*k*np.diag(d2udz2) + Kv*LL
b21 = -np.diag(dbdz)
b12 = -Id*kh**2
b22 = -1j*k*np.diag(vi) + Kb*Lb

# % assemble matrix B
# B=[b11 b12 ; b21 b22];

B = np.block([[b11, b12], [b21, b22]])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Stage 4: Solve eigenvalue problem and extract results
# %
# % Solve generalized eigenvalue problem
# [v,e]=eig(B,A);
# sigma=diag(e);

evals, evecs = sp.linalg.eig(B, A)

# % Sort eigvals
# [~,ind]=sort(real(sigma),1,'descend');
# sigma=sigma(ind);
# v=v(:,ind);


cp = -evals.imag/k
idxs = np.argsort(cp)
cp = cp[idxs]
evals = evals[idxs]
evecs = evecs[:, idxs]

wevec = evecs[:N, :]
bevec = evecs[N:, :]

fig, ax = plt.subplots(1, 1)
ax.plot(cp, '.')

fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].plot(wevec[:, -1], zi)
axs[1].plot(bevec[:, -1], zi)
axs[0].plot(wevec[:, 0], zi)
axs[1].plot(bevec[:, 0], zi)


# % Extract the selected mode(s)
# if imode==0
#     sig=sigma;
#     w=v(1:N,:);
#     b=v(NP:N2,:);
# elseif imode>0
#     sig=sigma(imode);
#     w=v(1:N,imode);
#     b=v(NP:N2,imode);
# end

# return


# %% [markdown]
# Try my own module.

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

fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].plot(wvec[:, -1], z)
axs[1].plot(bvec[:, -1], z)
axs[0].plot(wvec[:, 0], z)
axs[1].plot(bvec[:, 0], z)
