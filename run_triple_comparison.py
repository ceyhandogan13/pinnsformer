"""
Reproducible 3-way comparison: PINN vs PINNsFormer-FD vs PINNsFormer-AD
------------------------------------------------------------------------
• torch.manual_seed(42) + np.random.seed(42)
• N_RES=2000, N_TC=500, N_BC=500
• EPOCHS=5000 (her üç model)
"""
import sys, os, time
sys.path.insert(0, 'bs_pinn')
sys.path.insert(0, 'pinnsformer')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

from black_scholes_data import (
    generate_dataset, bs_call,
    sample_residual_points, sample_terminal_points, sample_boundary_points
)
from bs_pinn_model import BS_PINN, BS_PINNsFormer, BS_PINNsFormer_AD, rMAE, rRMSE

DEVICE = 'cpu'
K, r, sigma = 100.0, 0.05, 0.20
S_MIN, S_MAX, TAU_MAX = 20.0, 200.0, 1.0

N_RES, N_TC, N_BC = 2000, 500, 500
S_res,  tau_res         = sample_residual_points(N_RES, S_MIN, S_MAX, 1e-3, TAU_MAX, seed=0)
S_tc,   tau_tc,  V_tc  = sample_terminal_points(N_TC, S_MIN, S_MAX, K, r, sigma, seed=1)
(S_bl, tau_bl, V_bl), (S_br, tau_br, V_br) = sample_boundary_points(
    N_BC, 1e-3, TAU_MAX, S_MIN, S_MAX, K, r, sigma, seed=2
)
TRAIN_ARGS = (S_res, tau_res, S_tc, tau_tc, V_tc, S_bl, tau_bl, V_bl, S_br, tau_br, V_br)

ds_call  = generate_dataset(K=K, r=r, sigma=sigma, n_S=100, n_tau=100, option_type='call')
S_flat   = ds_call['flat']['S']
tau_flat = ds_call['flat']['tau']
V_true   = ds_call['flat']['V']
mask_eval = tau_flat > 1e-4

os.makedirs('bs_pinn', exist_ok=True)
EPOCHS = 5000

print('='*65)
print(f'Seed: torch.manual_seed({SEED}), np.random.seed({SEED})')
print(f'Collocation: N_RES={N_RES}, N_TC={N_TC}, N_BC={N_BC}')
print(f'Epochs: {EPOCHS}')
print('='*65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PINN
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1/3] PINN (hidden=64, layers=6, tanh, λ=(1,10,10), lr=1e-3)...')
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

V_pinn  = pinn.predict(S_flat, tau_flat)
rm_pinn = rMAE(V_pinn[mask_eval],  V_true[mask_eval])
rr_pinn = rRMSE(V_pinn[mask_eval], V_true[mask_eval])
print(f'  → rMAE={rm_pinn*100:.2f}%  rRMSE={rr_pinn*100:.2f}%  süre={pinn_time:.0f}s')

# ─────────────────────────────────────────────────────────────────────────────
# 2. PINNsFormer-FD
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n[2/3] PINNsFormer-FD (d_model=64, N=3, heads=4, λ=(1,50,50), clip=1.0)...')
torch.manual_seed(SEED)

pf = BS_PINNsFormer(d_model=64, d_hidden=128, N=3, heads=4,
                     num_step=5, step_size=1e-4,
                     lam1=1.0, lam2=50.0, lam3=50.0,
                     lr=5e-4, device=DEVICE)
pf.precompute(*TRAIN_ARGS)

scheduler_fd = torch.optim.lr_scheduler.ReduceLROnPlateau(
    pf.optimizer, mode='min', factor=0.5, patience=400, min_lr=1e-6
)

hist_pf = []
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    losses = pf.train_step(clip_grad=1.0)
    hist_pf.append(losses)
    scheduler_fd.step(losses[0])
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
# 3. PINNsFormer-AD
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n[3/3] PINNsFormer-AD (d_model=64, N=3, heads=4, λ=(1,50,50), clip=1.0)...')
torch.manual_seed(SEED)

pad = BS_PINNsFormer_AD(d_model=64, d_hidden=128, N=3, heads=4,
                         num_step=5, step_size=1e-4,
                         sigma=sigma, r=r,
                         lam1=1.0, lam2=50.0, lam3=50.0,
                         lr=5e-4, device=DEVICE)

scheduler_ad = torch.optim.lr_scheduler.ReduceLROnPlateau(
    pad.optimizer, mode='min', factor=0.5, patience=400, min_lr=1e-6
)

hist_ad = []
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    losses = pad.train_step(*TRAIN_ARGS)
    hist_ad.append(losses)
    scheduler_ad.step(losses[0])
    if ep % 1000 == 0:
        lr_now = pad.optimizer.param_groups[0]['lr']
        print(f'  Ep {ep:6d} | total={losses[0]:.4e} | lr={lr_now:.2e}')

ad_time = time.time() - t0
hist_ad = np.array(hist_ad)

V_ad  = pad.predict(S_flat, tau_flat)
rm_ad = rMAE(V_ad[mask_eval],  V_true[mask_eval])
rr_ad = rRMSE(V_ad[mask_eval], V_true[mask_eval])
print(f'  → rMAE={rm_ad*100:.2f}%  rRMSE={rr_ad*100:.2f}%  süre={ad_time:.0f}s')

# ─────────────────────────────────────────────────────────────────────────────
# Sonuç tablosu
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print(f'{"Model":<38} | {"rMAE":>8} | {"rRMSE":>8} | {"Süre":>8}')
print('-'*70)
print(f'{"PINN (5 000 ep)":<38} | {rm_pinn*100:>7.2f}% | {rr_pinn*100:>7.2f}% | {pinn_time:>5.0f}s')
print(f'{"PINNsFormer-FD (5 000 ep)":<38} | {rm_pf*100:>7.2f}% | {rr_pf*100:>7.2f}% | {pf_time:>5.0f}s')
print(f'{"PINNsFormer-AD (5 000 ep)":<38} | {rm_ad*100:>7.2f}% | {rr_ad*100:>7.2f}% | {ad_time:>5.0f}s')
print('='*70)
print(f'Seed: {SEED}  |  N_RES={N_RES}  |  N_TC={N_TC}  |  N_BC={N_BC}')

# ─────────────────────────────────────────────────────────────────────────────
# Figürler
# ─────────────────────────────────────────────────────────────────────────────

# 1) Loss eğrileri
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].semilogy(hist_pinn[:, 0], 'b-',   label='PINN',            linewidth=1.5)
axes[0].semilogy(hist_pf[:, 0],   'r--',  label='PINNsFormer-FD',  linewidth=1.5)
axes[0].semilogy(hist_ad[:, 0],   'g:',   label='PINNsFormer-AD',  linewidth=2.0)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Total Loss (log)')
axes[0].set_title(f'Loss Karşılaştırması — 5 000 Epoch (seed={SEED})')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

half = EPOCHS // 2
axes[1].semilogy(range(half, EPOCHS), hist_pinn[half:, 0], 'b-',  label='PINN',           linewidth=1.5)
axes[1].semilogy(range(half, EPOCHS), hist_pf[half:, 0],   'r--', label='PINNsFormer-FD', linewidth=1.5)
axes[1].semilogy(range(half, EPOCHS), hist_ad[half:, 0],   'g:',  label='PINNsFormer-AD', linewidth=2.0)
axes[1].set_xlabel('Epoch'); axes[1].set_title('İkinci Yarı (yakınsama)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bs_pinn/fig_triple_loss.png', dpi=130)
plt.close()

# 2) Bar chart (rMAE + rRMSE)
fig, ax = plt.subplots(figsize=(9, 5))
names = ['PINN\n(5k ep)', 'PINNsFormer-FD\n(5k ep)', 'PINNsFormer-AD\n(5k ep)']
rmaes_c  = [rm_pinn*100, rm_pf*100, rm_ad*100]
rrmses_c = [rr_pinn*100, rr_pf*100, rr_ad*100]
colors   = ['steelblue', 'crimson', 'seagreen']
x = np.arange(3); w = 0.35
b1 = ax.bar(x - w/2, rmaes_c,  w, label='rMAE (%)',  color=colors, alpha=0.85)
b2 = ax.bar(x + w/2, rrmses_c, w, label='rRMSE (%)', color=colors, alpha=0.50)
for bar, val in zip(list(b1) + list(b2), rmaes_c + rrmses_c):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel('Hata (%)'); ax.set_title(f'PINN vs PINNsFormer-FD vs PINNsFormer-AD\n(seed={SEED}, N_res={N_RES}, 5 000 epoch)')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('bs_pinn/fig_triple_comparison.png', dpi=130)
plt.close()

# 3) Hata ısı haritaları (3 panel)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, V_pred, rm, title in [
    (axes[0], V_pinn, rm_pinn, f'PINN  rMAE={rm_pinn*100:.2f}%'),
    (axes[1], V_pf,   rm_pf,   f'PINNsFormer-FD  rMAE={rm_pf*100:.2f}%'),
    (axes[2], V_ad,   rm_ad,   f'PINNsFormer-AD  rMAE={rm_ad*100:.2f}%'),
]:
    err = np.abs(V_pred - V_true).reshape(100, 100)
    im  = ax.contourf(ds_call['S_mesh'], ds_call['tau_mesh'], err, levels=20, cmap='hot_r')
    plt.colorbar(im, ax=ax, label='|Error|')
    ax.axvline(K, color='cyan', linewidth=1.5, linestyle='--', label=f'K={K}')
    ax.set_xlabel('Varlık Fiyatı S'); ax.set_ylabel('Vadeye Kalan Süre τ')
    ax.set_title(title); ax.legend()
plt.suptitle(f'Mutlak Hata Isı Haritası — 5 000 Epoch (seed={SEED})', fontsize=12)
plt.tight_layout()
plt.savefig('bs_pinn/fig_triple_heatmap.png', dpi=130)
plt.close()

# 4) Dilim karşılaştırması
tau_vals = [0.1, 0.3, 0.5, 1.0]
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, tv in zip(axes, tau_vals):
    tau_arr = np.full_like(ds_call['S_grid'], tv)
    V_ana   = bs_call(ds_call['S_grid'], K, r, sigma, tv)
    V_nn    = pinn.predict(ds_call['S_grid'], tau_arr)
    V_pf_s  = pf.predict(ds_call['S_grid'], tau_arr)
    V_ad_s  = pad.predict(ds_call['S_grid'], tau_arr)
    ax.plot(ds_call['S_grid'], V_ana,   'k-',  label='Analitik',         linewidth=2.5)
    ax.plot(ds_call['S_grid'], V_nn,    'b--', label='PINN',              linewidth=1.5)
    ax.plot(ds_call['S_grid'], V_pf_s,  'r:',  label='PINNsFormer-FD',   linewidth=2.0)
    ax.plot(ds_call['S_grid'], V_ad_s,  'g-.', label='PINNsFormer-AD',   linewidth=1.8)
    ax.axvline(K, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('S'); ax.set_ylabel('V'); ax.set_title(f'τ={tv} yr')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.suptitle(f'Analitik vs PINN vs PINNsFormer-FD vs PINNsFormer-AD\n(seed={SEED}, N_res={N_RES}, 5 000 epoch)',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('bs_pinn/fig_triple_slices.png', dpi=130)
plt.close()

# 5) Süre karşılaştırması (bar)
fig, ax = plt.subplots(figsize=(7, 4))
models_t = ['PINN', 'PINNsFormer-FD', 'PINNsFormer-AD']
times_t  = [pinn_time, pf_time, ad_time]
bars = ax.bar(models_t, times_t, color=['steelblue', 'crimson', 'seagreen'], alpha=0.85)
for bar, val in zip(bars, times_t):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{val:.0f}s\n({val/60:.1f}dk)', ha='center', va='bottom', fontsize=10)
ax.set_ylabel('Eğitim Süresi (s)'); ax.set_title(f'Eğitim Süresi Karşılaştırması — 5 000 Epoch\n(seed={SEED}, N_res={N_RES})')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('bs_pinn/fig_triple_time.png', dpi=130)
plt.close()

print('\nTüm figürler kaydedildi → bs_pinn/fig_triple_*.png')
