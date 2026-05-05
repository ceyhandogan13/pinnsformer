"""
Reproducible final comparison: PINN vs PINNsFormer-FD
-------------------------------------------------------
• torch.manual_seed(42) sabitlendi  → her çalıştırmada aynı sonuç
• N_RES=2000, N_TC=500, N_BC=500    → Report 1 ile tutarlı
• Her iki model aynı collocation noktalarında, aynı epoch sayısıyla
"""
import sys, os, time
sys.path.insert(0, 'bs_pinn')
sys.path.insert(0, 'pinnsformer')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Sabit seed (reproducibility) ──────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

from black_scholes_data import (
    generate_dataset, bs_call,
    sample_residual_points, sample_terminal_points, sample_boundary_points
)
from bs_pinn_model import BS_PINN, BS_PINNsFormer, rMAE, rRMSE

DEVICE = 'cpu'
K, r, sigma = 100.0, 0.05, 0.20
S_MIN, S_MAX, TAU_MAX = 20.0, 200.0, 1.0

# ── Collocation points (Report 1 ile aynı) ────────────────────────────────────
N_RES, N_TC, N_BC = 2000, 500, 500
S_res,  tau_res          = sample_residual_points(N_RES, S_MIN, S_MAX, 1e-3, TAU_MAX, seed=0)
S_tc,   tau_tc,  V_tc   = sample_terminal_points(N_TC, S_MIN, S_MAX, K, r, sigma, seed=1)
(S_bl, tau_bl, V_bl), (S_br, tau_br, V_br) = sample_boundary_points(
    N_BC, 1e-3, TAU_MAX, S_MIN, S_MAX, K, r, sigma, seed=2
)
TRAIN_ARGS = (S_res, tau_res, S_tc, tau_tc, V_tc, S_bl, tau_bl, V_bl, S_br, tau_br, V_br)

# Değerlendirme grid
ds_call  = generate_dataset(K=K, r=r, sigma=sigma, n_S=100, n_tau=100, option_type='call')
S_flat   = ds_call['flat']['S']
tau_flat = ds_call['flat']['tau']
V_true   = ds_call['flat']['V']
mask_eval = tau_flat > 1e-4

os.makedirs('bs_pinn', exist_ok=True)
EPOCHS = 10000

print('='*65)
print(f'Sabit seed: torch.manual_seed({SEED}), np.random.seed({SEED})')
print(f'Collocation: N_RES={N_RES}, N_TC={N_TC}, N_BC={N_BC}')
print(f'Epochs: {EPOCHS}')
print('='*65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PINN (Report 1 ile birebir aynı ayarlar)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1/2] PINN (hidden=64, layers=6, tanh, λ=(1,10,10), lr=1e-3)...')
torch.manual_seed(SEED)

pinn = BS_PINN(hidden_dim=64, num_layer=6, sigma=sigma, r=r,
               lam1=1.0, lam2=10.0, lam3=10.0, lr=1e-3, device=DEVICE)

hist_pinn = []
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    losses = pinn.train_step(*TRAIN_ARGS)
    hist_pinn.append(losses)
    if ep % 1000 == 0:
        print(f'  Ep {ep:6d} | total={losses[0]:.4e}')

pinn_time = time.time() - t0
hist_pinn = np.array(hist_pinn)

V_pinn   = pinn.predict(S_flat, tau_flat)
rm_pinn  = rMAE(V_pinn[mask_eval],  V_true[mask_eval])
rr_pinn  = rRMSE(V_pinn[mask_eval], V_true[mask_eval])
print(f'  → rMAE={rm_pinn*100:.2f}%  rRMSE={rr_pinn*100:.2f}%  süre={pinn_time:.0f}s')

# ─────────────────────────────────────────────────────────────────────────────
# 2. PINNsFormer-FD (best λ, clip_grad, scheduler)
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n[2/2] PINNsFormer-FD (d_model=64, N=3, heads=4, λ=(1,50,50), clip=1.0)...')
torch.manual_seed(SEED)

pf = BS_PINNsFormer(d_model=64, d_hidden=128, N=3, heads=4,
                     num_step=5, step_size=1e-4,
                     lam1=1.0, lam2=50.0, lam3=50.0,
                     lr=5e-4, device=DEVICE)
pf.precompute(*TRAIN_ARGS)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    pf.optimizer, mode='min', factor=0.5, patience=400, min_lr=1e-6
)

hist_pf = []
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    losses = pf.train_step(clip_grad=1.0)
    hist_pf.append(losses)
    scheduler.step(losses[0])
    if ep % 1000 == 0:
        lr_now = pf.optimizer.param_groups[0]['lr']
        print(f'  Ep {ep:6d} | total={losses[0]:.4e} | lr={lr_now:.2e}')

pf_time = time.time() - t0
hist_pf = np.array(hist_pf)

V_pf  = pf.predict(S_flat, tau_flat)
rm_pf = rMAE(V_pf[mask_eval],  V_true[mask_eval])
rr_pf = rRMSE(V_pf[mask_eval], V_true[mask_eval])
print(f'  → rMAE={rm_pf*100:.2f}%  rRMSE={rr_pf*100:.2f}%  süre={pf_time:.0f}s')

# ─────────────────────────────────────────────────────────────────────────────
# Sonuç tablosu
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print(f'{"Model":<35} | {"rMAE":>8} | {"rRMSE":>8} | {"Süre":>8}')
print('-'*65)
print(f'{"PINN (10 000 ep)":<35} | {rm_pinn*100:>7.2f}% | {rr_pinn*100:>7.2f}% | {pinn_time:>5.0f}s')
print(f'{"PINNsFormer-FD (10 000 ep)":<35} | {rm_pf*100:>7.2f}% | {rr_pf*100:>7.2f}% | {pf_time:>5.0f}s')
print('='*65)
print(f'\nSeed: {SEED}  |  N_RES={N_RES}  |  N_TC={N_TC}  |  N_BC={N_BC}')

# ─────────────────────────────────────────────────────────────────────────────
# Figürler
# ─────────────────────────────────────────────────────────────────────────────

# 1) Loss eğrileri
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].semilogy(hist_pinn[:, 0], 'b-',  label='PINN',            linewidth=1.5)
axes[0].semilogy(hist_pf[:, 0],   'r--', label='PINNsFormer-FD',  linewidth=1.5)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Total Loss (log)')
axes[0].set_title('Loss Karşılaştırması (10 000 epoch, seed=42)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

half = EPOCHS // 2
axes[1].semilogy(range(half, EPOCHS), hist_pinn[half:, 0], 'b-',  label='PINN',           linewidth=1.5)
axes[1].semilogy(range(half, EPOCHS), hist_pf[half:, 0],   'r--', label='PINNsFormer-FD', linewidth=1.5)
axes[1].set_xlabel('Epoch'); axes[1].set_title('İkinci Yarı (yakınsama)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bs_pinn/fig_final_loss.png', dpi=130)
plt.close()

# 2) Bar chart
fig, ax = plt.subplots(figsize=(7, 5))
names = ['PINN\n(10k ep)', 'PINNsFormer-FD\n(10k ep)']
rmaes_c  = [rm_pinn*100, rm_pf*100]
rrmses_c = [rr_pinn*100, rr_pf*100]
x = np.arange(2); w = 0.35
b1 = ax.bar(x - w/2, rmaes_c,  w, label='rMAE (%)',  color=['steelblue', 'crimson'], alpha=0.85)
b2 = ax.bar(x + w/2, rrmses_c, w, label='rRMSE (%)', color=['steelblue', 'crimson'], alpha=0.50)
for bar, val in zip(list(b1) + list(b2), rmaes_c + rrmses_c):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12)
ax.set_ylabel('Hata (%)'); ax.set_title(f'PINN vs PINNsFormer — Reproducible\n(seed={SEED}, N_res={N_RES})')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('bs_pinn/fig_final_comparison.png', dpi=130)
plt.close()

# 3) Hata ısı haritaları
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, V_pred, rm, title in [
    (axes[0], V_pinn, rm_pinn, f'PINN  rMAE={rm_pinn*100:.2f}%'),
    (axes[1], V_pf,   rm_pf,   f'PINNsFormer-FD  rMAE={rm_pf*100:.2f}%'),
]:
    err = np.abs(V_pred - V_true).reshape(100, 100)
    im  = ax.contourf(ds_call['S_mesh'], ds_call['tau_mesh'], err, levels=20, cmap='hot_r')
    plt.colorbar(im, ax=ax, label='|Error|')
    ax.axvline(K, color='cyan', linewidth=1.5, linestyle='--', label=f'K={K}')
    ax.set_xlabel('Varlık Fiyatı S'); ax.set_ylabel('Vadeye Kalan Süre τ')
    ax.set_title(title); ax.legend()
plt.suptitle('Mutlak Hata Isı Haritası — 10 000 Epoch (seed=42)', fontsize=12)
plt.tight_layout()
plt.savefig('bs_pinn/fig_final_heatmap.png', dpi=130)
plt.close()

# 4) Dilim karşılaştırması
tau_vals = [0.1, 0.3, 0.5, 1.0]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, tv in zip(axes, tau_vals):
    tau_arr = np.full_like(ds_call['S_grid'], tv)
    V_ana   = bs_call(ds_call['S_grid'], K, r, sigma, tv)
    V_nn    = pinn.predict(ds_call['S_grid'], tau_arr)
    V_pf_s  = pf.predict(ds_call['S_grid'], tau_arr)
    ax.plot(ds_call['S_grid'], V_ana,  'k-',  label='Analitik',         linewidth=2.5)
    ax.plot(ds_call['S_grid'], V_nn,   'b--', label='PINN',              linewidth=1.5)
    ax.plot(ds_call['S_grid'], V_pf_s, 'r:',  label='PINNsFormer-FD',   linewidth=2)
    ax.axvline(K, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('S'); ax.set_ylabel('V'); ax.set_title(f'τ={tv} yr')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle(f'Analitik vs PINN vs PINNsFormer-FD — 10 000 Epoch (seed={SEED}, N_res={N_RES})', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('bs_pinn/fig_final_slices.png', dpi=130)
plt.close()

print('\nTüm figürler kaydedildi → bs_pinn/fig_final_*.png')
