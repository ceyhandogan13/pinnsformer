"""
Week 3-4 experiments: λ sweep + final training + PINN vs PINNsFormer comparison.
Saves all figures and prints results table.
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
from bs_pinn_model import BS_PINN, BS_PINNsFormer, BS_PINNsFormer_AD, rMAE, rRMSE

DEVICE = 'cpu'
K, r, sigma = 100.0, 0.05, 0.20
S_MIN, S_MAX, TAU_MAX = 20.0, 200.0, 1.0

# ── Shared collocation points ─────────────────────────────────────────────────
N_RES, N_TC, N_BC = 500, 200, 200
S_res,  tau_res            = sample_residual_points(N_RES, S_MIN, S_MAX, 1e-3, TAU_MAX, seed=0)
S_tc,   tau_tc,   V_tc    = sample_terminal_points(N_TC, S_MIN, S_MAX, K, r, sigma, seed=1)
(S_bl, tau_bl, V_bl), (S_br, tau_br, V_br) = sample_boundary_points(
    N_BC, 1e-3, TAU_MAX, S_MIN, S_MAX, K, r, sigma, seed=2
)
TRAIN_ARGS = (S_res, tau_res, S_tc, tau_tc, V_tc, S_bl, tau_bl, V_bl, S_br, tau_br, V_br)

# Evaluation grid
ds_call   = generate_dataset(K=K, r=r, sigma=sigma, n_S=100, n_tau=100, option_type='call')
S_flat    = ds_call['flat']['S']
tau_flat  = ds_call['flat']['tau']
V_true    = ds_call['flat']['V']
mask_eval = tau_flat > 1e-4

os.makedirs('bs_pinn', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. WaveAct vs tanh plot
# ─────────────────────────────────────────────────────────────────────────────
x = np.linspace(-4, 4, 400)
wave = np.sin(x) + np.cos(x)
tanh = np.tanh(x)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(x, tanh, 'b-', label='tanh (PINN)', linewidth=2)
axes[0].plot(x, wave, 'r--', label='WaveAct (PINNsFormer)', linewidth=2)
axes[0].set_title('Activation Functions'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
for sig, label, color in [(tanh,'tanh','b'), (wave,'WaveAct','r')]:
    fft = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(x))
    axes[1].semilogy(freqs[:60], fft[:60]+1e-10, color=color, label=label, linewidth=1.5)
axes[1].set_title('Frequency Spectrum'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bs_pinn/fig_wavelet_vs_tanh.png', dpi=120)
plt.close()
print('[1/6] WaveAct plot saved.')

# ─────────────────────────────────────────────────────────────────────────────
# 2. Speed benchmark
# ─────────────────────────────────────────────────────────────────────────────
pf_ad_b = BS_PINNsFormer_AD(d_model=32, d_hidden=64, N=2, heads=2)
for _ in range(3): pf_ad_b.train_step(*TRAIN_ARGS)
t0 = time.time()
for _ in range(20): pf_ad_b.train_step(*TRAIN_ARGS)
time_ad = (time.time()-t0)/20

pf_fd_b = BS_PINNsFormer(d_model=32, d_hidden=64, N=2, heads=2)
pf_fd_b.precompute(*TRAIN_ARGS)
for _ in range(3): pf_fd_b.train_step()
t0 = time.time()
for _ in range(20): pf_fd_b.train_step()
time_fd = (time.time()-t0)/20

print(f'[2/6] Speed — AD: {time_ad*1000:.1f}ms/ep  FD: {time_fd*1000:.1f}ms/ep  speedup: {time_ad/time_fd:.1f}x')

# ─────────────────────────────────────────────────────────────────────────────
# 3. λ sweep (with gradient clipping)
# ─────────────────────────────────────────────────────────────────────────────
SWEEP_EPOCHS = 1000
CONFIGS = [
    ('(1,  1,  1)', 1.0,  1.0,  1.0),
    ('(1,  5,  5)', 1.0,  5.0,  5.0),
    ('(1, 10, 10)', 1.0, 10.0, 10.0),
    ('(1, 20, 20)', 1.0, 20.0, 20.0),
    ('(1, 50, 50)', 1.0, 50.0, 50.0),
    ('(1, 10, 20)', 1.0, 10.0, 20.0),
    ('(1, 20, 10)', 1.0, 20.0, 10.0),
    ('(2, 10, 10)', 2.0, 10.0, 10.0),
]
sweep_results = []
print('[3/6] λ sweep...')
for label, l1, l2, l3 in CONFIGS:
    pf = BS_PINNsFormer(d_model=32, d_hidden=64, N=2, heads=2,
                         lam1=l1, lam2=l2, lam3=l3, lr=5e-4)
    pf.precompute(*TRAIN_ARGS)
    t0 = time.time()
    for _ in range(SWEEP_EPOCHS):
        pf.train_step(clip_grad=1.0)
    elapsed = time.time()-t0
    V_pred = pf.predict(S_flat, tau_flat)
    rm  = rMAE(V_pred[mask_eval],  V_true[mask_eval])
    rr  = rRMSE(V_pred[mask_eval], V_true[mask_eval])
    sweep_results.append((label, l1, l2, l3, rm, rr, elapsed))
    print(f'  {label:14s} | rMAE={rm:.4f} | rRMSE={rr:.4f} | {elapsed:.0f}s')

best = min(sweep_results, key=lambda x: x[4])
print(f'  Best: λ={best[0]}  rMAE={best[4]:.4f}  rRMSE={best[5]:.4f}')

# Sweep bar chart
labels  = [r[0] for r in sweep_results]
rmaes   = [r[4]*100 for r in sweep_results]
rrmses  = [r[5]*100 for r in sweep_results]
best_i  = rmaes.index(min(rmaes))
xb = np.arange(len(labels)); w = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(xb-w/2, rmaes,  w, label='rMAE (%)',  color='steelblue', alpha=0.8)
b2 = ax.bar(xb+w/2, rrmses, w, label='rRMSE (%)', color='coral',    alpha=0.8)
b1[best_i].set_edgecolor('black'); b1[best_i].set_linewidth(2)
b2[best_i].set_edgecolor('black'); b2[best_i].set_linewidth(2)
for bar, val in zip(b1, rmaes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
ax.set_xticks(xb); ax.set_xticklabels([f'λ={l}' for l in labels], rotation=30, ha='right')
ax.set_ylabel('Error (%)'); ax.set_title(f'λ Sweep — PINNsFormer-FD ({SWEEP_EPOCHS} ep, clip_grad=1.0)')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('bs_pinn/fig_lambda_sweep.png', dpi=120)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Final PINNsFormer-FD (gradient clipping + LR scheduler)
# ─────────────────────────────────────────────────────────────────────────────
PF_EPOCHS = 5000
print(f'\n[4/6] Final PINNsFormer-FD ({PF_EPOCHS} epochs, clip_grad=1.0, ReduceLROnPlateau)...')

pf_final = BS_PINNsFormer(d_model=64, d_hidden=128, N=3, heads=4,
                            lam1=1.0, lam2=20.0, lam3=20.0,
                            lr=5e-4, device=DEVICE)
pf_final.precompute(*TRAIN_ARGS)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    pf_final.optimizer, mode='min', factor=0.5, patience=300, min_lr=1e-6
)

hist_pf = []
t0_pf = time.time()
for ep in range(1, PF_EPOCHS+1):
    losses = pf_final.train_step(clip_grad=1.0)
    hist_pf.append(losses)
    scheduler.step(losses[0])
    if ep % 500 == 0:
        lr_now = pf_final.optimizer.param_groups[0]['lr']
        print(f'  Ep {ep:5d} | total={losses[0]:.4e} | pde={losses[1]:.4e} '
              f'| tc={losses[2]:.4e} | bc={losses[3]:.4e} | lr={lr_now:.2e}')

pf_train_time = time.time()-t0_pf
hist_pf = np.array(hist_pf)

V_pf = pf_final.predict(S_flat, tau_flat)
rm_pf  = rMAE(V_pf[mask_eval],  V_true[mask_eval])
rr_pf  = rRMSE(V_pf[mask_eval], V_true[mask_eval])
print(f'  → rMAE={rm_pf:.4f} ({rm_pf*100:.2f}%)  rRMSE={rr_pf:.4f}  time={pf_train_time:.0f}s')

# Loss curve
fig, axes = plt.subplots(1,2,figsize=(12,4))
axes[0].semilogy(hist_pf[:,0], label='Total', color='purple')
axes[0].semilogy(hist_pf[:,1], label='L_pde', linestyle='--', color='blue')
axes[0].semilogy(hist_pf[:,2], label='L_tc',  linestyle=':', color='green')
axes[0].semilogy(hist_pf[:,3], label='L_bc',  linestyle='-.', color='orange')
axes[0].set_xlabel('Epoch'); axes[0].set_title('PINNsFormer-FD Loss (clip+scheduler)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
half = len(hist_pf)//2
axes[1].semilogy(range(half,PF_EPOCHS), hist_pf[half:,0], color='purple')
axes[1].set_xlabel('Epoch'); axes[1].set_title('PINNsFormer-FD — Second Half')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bs_pinn/fig_pf_loss.png', dpi=120)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 5. PINN baseline
# ─────────────────────────────────────────────────────────────────────────────
PINN_EPOCHS = 10000
print(f'\n[5/6] PINN baseline ({PINN_EPOCHS} epochs)...')

pinn = BS_PINN(hidden_dim=64, num_layer=6, sigma=sigma, r=r,
               lam1=1.0, lam2=10.0, lam3=10.0, lr=1e-3, device=DEVICE)

hist_pinn = []
t0_pinn = time.time()
for ep in range(1, PINN_EPOCHS+1):
    losses = pinn.train_step(*TRAIN_ARGS)
    hist_pinn.append(losses)
    if ep % 2000 == 0:
        print(f'  Ep {ep:5d} | total={losses[0]:.4e}')

pinn_train_time = time.time()-t0_pinn
hist_pinn = np.array(hist_pinn)

V_pinn = pinn.predict(S_flat, tau_flat)
rm_pinn  = rMAE(V_pinn[mask_eval],  V_true[mask_eval])
rr_pinn  = rRMSE(V_pinn[mask_eval], V_true[mask_eval])
print(f'  → rMAE={rm_pinn:.4f} ({rm_pinn*100:.2f}%)  rRMSE={rr_pinn:.4f}  time={pinn_train_time:.0f}s')

# ─────────────────────────────────────────────────────────────────────────────
# 6. Comparison plots
# ─────────────────────────────────────────────────────────────────────────────
print('\n[6/6] Generating comparison plots...')

# Bar chart
names   = ['PINN\n(10k ep)', 'PINNsFormer-FD\n(5k ep, clip+sched)']
rmaes_c = [rm_pinn*100, rm_pf*100]
rrmses_c = [rr_pinn*100, rr_pf*100]
x = np.arange(len(names)); w = 0.35
fig, ax = plt.subplots(figsize=(8,5))
b1 = ax.bar(x-w/2, rmaes_c,  w, label='rMAE (%)',  color=['steelblue','purple'], alpha=0.85)
b2 = ax.bar(x+w/2, rrmses_c, w, label='rRMSE (%)', color=['steelblue','purple'], alpha=0.55)
for bar, val in zip(list(b1)+list(b2), rmaes_c+rrmses_c):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel('Error (%)'); ax.set_title('PINN vs PINNsFormer — European Call')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('bs_pinn/fig_model_comparison.png', dpi=120)
plt.close()

# Error heatmaps
fig, axes = plt.subplots(1,2,figsize=(14,5))
for ax, V_pred, title in [
    (axes[0], V_pinn, 'PINN |Error|'),
    (axes[1], V_pf,   'PINNsFormer-FD |Error|'),
]:
    err = np.abs(V_pred-V_true).reshape(100,100)
    im  = ax.contourf(ds_call['S_mesh'], ds_call['tau_mesh'], err, levels=20, cmap='hot_r')
    plt.colorbar(im, ax=ax, label='|Error|')
    ax.axvline(K, color='cyan', linewidth=1.5, linestyle='--', label=f'K={K}')
    ax.set_xlabel('S'); ax.set_ylabel('τ'); ax.set_title(title); ax.legend()
plt.suptitle('Absolute Error Heatmap — European Call', fontsize=12)
plt.tight_layout()
plt.savefig('bs_pinn/fig_error_heatmap.png', dpi=120)
plt.close()

# Slice comparison
tau_vals = [0.1, 0.3, 0.5, 1.0]
fig, axes = plt.subplots(1,4,figsize=(18,4))
for ax, tau_val in zip(axes, tau_vals):
    tau_arr = np.full_like(ds_call['S_grid'], tau_val)
    V_ana  = bs_call(ds_call['S_grid'], K, r, sigma, tau_val)
    V_nn   = pinn.predict(ds_call['S_grid'], tau_arr)
    V_pf_s = pf_final.predict(ds_call['S_grid'], tau_arr)
    ax.plot(ds_call['S_grid'], V_ana,  'k-',  label='Analytical', linewidth=2.5)
    ax.plot(ds_call['S_grid'], V_nn,   'b--', label='PINN',        linewidth=1.5)
    ax.plot(ds_call['S_grid'], V_pf_s, 'r:',  label='PINNsFormer-FD', linewidth=2)
    ax.axvline(K, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('S'); ax.set_ylabel('V'); ax.set_title(f'τ={tau_val}yr')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle('Analytical vs PINN vs PINNsFormer-FD (τ slices)', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('bs_pinn/fig_slice_comparison.png', dpi=120)
plt.close()

# Loss convergence overlay
fig, ax = plt.subplots(figsize=(9,5))
ax.semilogy(hist_pinn[:,0], 'b-',  label='PINN (10k ep)', linewidth=1.5)
ax.semilogy(np.linspace(0, PINN_EPOCHS, PF_EPOCHS), hist_pf[:,0],
            'r--', label='PINNsFormer-FD (5k ep, normalised axis)', linewidth=1.5)
ax.set_xlabel('Normalised training steps'); ax.set_ylabel('Total Loss (log)')
ax.set_title('Loss Convergence Comparison')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bs_pinn/fig_convergence_and_error.png', dpi=120)
plt.close()

# ── Final summary ─────────────────────────────────────────────────────────────
print('\n' + '='*65)
print(f'{"Model":<35} | {"rMAE":>8} | {"rRMSE":>8} | {"Time":>8}')
print('-'*65)
print(f'{"PINN (10 000 ep)":<35} | {rm_pinn*100:>7.2f}% | {rr_pinn*100:>7.2f}% | {pinn_train_time:>6.0f}s')
print(f'{"PINNsFormer-FD (5 000 ep)":<35} | {rm_pf*100:>7.2f}% | {rr_pf*100:>7.2f}% | {pf_train_time:>6.0f}s')
print('='*65)
print('\nAll figures saved to bs_pinn/ directory.')
