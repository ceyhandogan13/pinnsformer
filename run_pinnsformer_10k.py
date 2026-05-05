"""
PINNsFormer-FD: 10 000 epoch training with best λ=(1,50,50).
clip_grad=1.0 + ReduceLROnPlateau.
"""
import sys, os, time
sys.path.insert(0, 'bs_pinn')
sys.path.insert(0, 'pinnsformer')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from black_scholes_data import (
    generate_dataset, bs_call,
    sample_residual_points, sample_terminal_points, sample_boundary_points
)
from bs_pinn_model import BS_PINN, BS_PINNsFormer, rMAE, rRMSE

DEVICE = 'cpu'
K, r, sigma = 100.0, 0.05, 0.20
S_MIN, S_MAX, TAU_MAX = 20.0, 200.0, 1.0

# ── Collocation points (same seed as Week 3-4) ────────────────────────────────
N_RES, N_TC, N_BC = 500, 200, 200
S_res,  tau_res          = sample_residual_points(N_RES, S_MIN, S_MAX, 1e-3, TAU_MAX, seed=0)
S_tc,   tau_tc,  V_tc   = sample_terminal_points(N_TC, S_MIN, S_MAX, K, r, sigma, seed=1)
(S_bl, tau_bl, V_bl), (S_br, tau_br, V_br) = sample_boundary_points(
    N_BC, 1e-3, TAU_MAX, S_MIN, S_MAX, K, r, sigma, seed=2
)
TRAIN_ARGS = (S_res, tau_res, S_tc, tau_tc, V_tc, S_bl, tau_bl, V_bl, S_br, tau_br, V_br)

# Evaluation grid
ds_call  = generate_dataset(K=K, r=r, sigma=sigma, n_S=100, n_tau=100, option_type='call')
S_flat   = ds_call['flat']['S']
tau_flat = ds_call['flat']['tau']
V_true   = ds_call['flat']['V']
mask_eval = tau_flat > 1e-4

os.makedirs('bs_pinn', exist_ok=True)

# ── PINNsFormer-FD: 10 000 epochs ─────────────────────────────────────────────
EPOCHS   = 10000
LOG_EVERY = 1000

print(f'PINNsFormer-FD | d_model=64, N=3, heads=4')
print(f'λ=(1,50,50) | clip_grad=1.0 | ReduceLROnPlateau | lr=5e-4')
print(f'Epochs: {EPOCHS}')
print('-'*60)

pf = BS_PINNsFormer(d_model=64, d_hidden=128, N=3, heads=4,
                     num_step=5, step_size=1e-4,
                     lam1=1.0, lam2=50.0, lam3=50.0,
                     lr=5e-4, device=DEVICE)
pf.precompute(*TRAIN_ARGS)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    pf.optimizer, mode='min', factor=0.5, patience=400, min_lr=1e-6
)

hist = []
t0 = time.time()

for ep in range(1, EPOCHS + 1):
    losses = pf.train_step(clip_grad=1.0)
    hist.append(losses)
    scheduler.step(losses[0])
    if ep % LOG_EVERY == 0:
        lr_now = pf.optimizer.param_groups[0]['lr']
        print(f'Ep {ep:6d} | total={losses[0]:.4e} | pde={losses[1]:.4e} '
              f'| tc={losses[2]:.4e} | bc={losses[3]:.4e} | lr={lr_now:.2e}')

elapsed = time.time() - t0
hist = np.array(hist)

# ── Evaluation ────────────────────────────────────────────────────────────────
V_pf = pf.predict(S_flat, tau_flat)
rm   = rMAE(V_pf[mask_eval],  V_true[mask_eval])
rr   = rRMSE(V_pf[mask_eval], V_true[mask_eval])
print(f'\nPINNsFormer-FD (10 000 ep) → rMAE={rm*100:.2f}%  rRMSE={rr*100:.2f}%  time={elapsed:.0f}s ({elapsed/60:.1f}min)')

# ── PINN reference (same points) ─────────────────────────────────────────────
print('\nPINN reference (10 000 ep)...')
pinn = BS_PINN(hidden_dim=64, num_layer=6, sigma=sigma, r=r,
               lam1=1.0, lam2=10.0, lam3=10.0, lr=1e-3, device=DEVICE)
t0p = time.time()
for ep in range(1, EPOCHS + 1):
    pinn.train_step(*TRAIN_ARGS)
pinn_time = time.time() - t0p
V_pinn  = pinn.predict(S_flat, tau_flat)
rm_p    = rMAE(V_pinn[mask_eval],  V_true[mask_eval])
rr_p    = rRMSE(V_pinn[mask_eval], V_true[mask_eval])
print(f'PINN (10 000 ep) → rMAE={rm_p*100:.2f}%  rRMSE={rr_p*100:.2f}%  time={pinn_time:.0f}s')

# ── Comparison table ──────────────────────────────────────────────────────────
print('\n' + '='*65)
print(f'{"Model":<35} | {"rMAE":>8} | {"rRMSE":>8} | {"Time":>8}')
print('-'*65)
print(f'{"PINN (10 000 ep)":<35} | {rm_p*100:>7.2f}% | {rr_p*100:>7.2f}% | {pinn_time:>6.0f}s')
print(f'{"PINNsFormer-FD (10 000 ep)":<35} | {rm*100:>7.2f}% | {rr*100:>7.2f}% | {elapsed:>6.0f}s')
print('='*65)

# ── Loss curves ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].semilogy(hist[:, 0], color='purple', label='Total', linewidth=1.2)
axes[0].semilogy(hist[:, 1], color='blue',   label='L_pde', linestyle='--', linewidth=1)
axes[0].semilogy(hist[:, 2], color='green',  label='L_tc',  linestyle=':', linewidth=1)
axes[0].semilogy(hist[:, 3], color='orange', label='L_bc',  linestyle='-.', linewidth=1)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss (log)')
axes[0].set_title('PINNsFormer-FD — 10 000 epochs\n(λ=(1,50,50), clip_grad=1.0, ReduceLROnPlateau)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

half = EPOCHS // 2
axes[1].semilogy(range(half, EPOCHS), hist[half:, 0], color='purple', linewidth=1.2)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Total Loss (log)')
axes[1].set_title('Second Half (convergence)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bs_pinn/fig_pf_10k_loss.png', dpi=130)
plt.close()
print('Loss curve saved → bs_pinn/fig_pf_10k_loss.png')

# ── Slice comparison ──────────────────────────────────────────────────────────
tau_vals = [0.1, 0.3, 0.5, 1.0]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, tv in zip(axes, tau_vals):
    tau_arr = np.full_like(ds_call['S_grid'], tv)
    V_ana   = bs_call(ds_call['S_grid'], K, r, sigma, tv)
    V_nn    = pinn.predict(ds_call['S_grid'], tau_arr)
    V_pf_s  = pf.predict(ds_call['S_grid'], tau_arr)
    ax.plot(ds_call['S_grid'], V_ana,  'k-',  label='Analytical',        linewidth=2.5)
    ax.plot(ds_call['S_grid'], V_nn,   'b--', label='PINN (10k)',         linewidth=1.5)
    ax.plot(ds_call['S_grid'], V_pf_s, 'r:',  label='PINNsFormer-FD (10k)', linewidth=2)
    ax.axvline(K, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('S'); ax.set_ylabel('V'); ax.set_title(f'τ={tv} yr')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle('Analytical vs PINN vs PINNsFormer-FD — 10 000 epochs', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('bs_pinn/fig_pf_10k_slices.png', dpi=130)
plt.close()
print('Slice comparison saved → bs_pinn/fig_pf_10k_slices.png')

# ── Error heatmaps ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, V_pred, title in [
    (axes[0], V_pinn, f'PINN |Error|  (rMAE={rm_p*100:.2f}%)'),
    (axes[1], V_pf,   f'PINNsFormer-FD |Error|  (rMAE={rm*100:.2f}%)'),
]:
    err = np.abs(V_pred - V_true).reshape(100, 100)
    im  = ax.contourf(ds_call['S_mesh'], ds_call['tau_mesh'], err, levels=20, cmap='hot_r')
    plt.colorbar(im, ax=ax, label='|Error|')
    ax.axvline(K, color='cyan', linewidth=1.5, linestyle='--', label=f'K={K}')
    ax.set_xlabel('S'); ax.set_ylabel('τ'); ax.set_title(title); ax.legend()
plt.suptitle('Absolute Error Heatmap — 10 000 epoch comparison', fontsize=12)
plt.tight_layout()
plt.savefig('bs_pinn/fig_pf_10k_heatmap.png', dpi=130)
plt.close()
print('Heatmap saved → bs_pinn/fig_pf_10k_heatmap.png')
