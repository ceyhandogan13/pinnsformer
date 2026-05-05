"""
Ablation study for PINNsFormer on Black-Scholes:
  1. WaveAct vs tanh activation
  2. Sequence length L ∈ {1, 3, 5, 10}
  3. Step size δ ∈ {1e-5, 1e-4, 1e-3}

All runs: d_model=64, N=3, heads=4, λ=(1,20,20),
          10 000 epochs, clip_grad=1.0, ReduceLROnPlateau,
          seed=42, N_res=2000, N_TC=500, N_BC=500, CPU.
"""
import sys, os, time, json
sys.path.insert(0, 'bs_pinn')
sys.path.insert(0, 'pinnsformer')

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from black_scholes_data import (
    generate_dataset, bs_call,
    sample_residual_points, sample_terminal_points, sample_boundary_points
)
from bs_pinn_model import BS_PINNsFormer, rMAE, rRMSE
from model.pinnsformer import WaveAct

# ── Shared setup ──────────────────────────────────────────────────────────────
K, r, sigma   = 100.0, 0.05, 0.20
S_MIN, S_MAX, TAU_MAX = 20.0, 200.0, 1.0
N_RES, N_TC, N_BC = 2000, 500, 500
SEED  = 42
EPOCHS = 10000
LOG_EVERY = 1000

ds_call  = generate_dataset(S_min=S_MIN, S_max=S_MAX, K=K, r=r, sigma=sigma,
                             n_S=100, n_tau=100, option_type='call')
S_flat   = ds_call['flat']['S']
tau_flat = ds_call['flat']['tau']
V_true   = ds_call['flat']['V']
mask_eval = tau_flat > 1e-4

np.random.seed(SEED)
S_res,  tau_res         = sample_residual_points(N_RES, S_MIN, S_MAX, 1e-3, TAU_MAX, seed=0)
S_tc,   tau_tc,  V_tc  = sample_terminal_points(N_TC, S_MIN, S_MAX, K, r, sigma, seed=1)
(S_bl, tau_bl, V_bl), (S_br, tau_br, V_br) = sample_boundary_points(
    N_BC, 1e-3, TAU_MAX, S_MIN, S_MAX, K, r, sigma, seed=2)
TRAIN_ARGS = (S_res, tau_res, S_tc, tau_tc, V_tc, S_bl, tau_bl, V_bl, S_br, tau_br, V_br)

OUT = 'bs_pinn'

# ── Helper: replace WaveAct with tanh ────────────────────────────────────────
def replace_waveact(model, act_cls):
    for name, module in list(model.named_children()):
        if isinstance(module, WaveAct):
            setattr(model, name, act_cls())
        else:
            replace_waveact(module, act_cls)

# ── Training helper ───────────────────────────────────────────────────────────
def train_model(label, pf):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    pf.precompute(*TRAIN_ARGS)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        pf.optimizer, mode='min', factor=0.5, patience=300, min_lr=1e-6)
    hist = []
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        losses = pf.train_step(clip_grad=1.0)
        hist.append(losses)
        scheduler.step(losses[0])
        if ep % LOG_EVERY == 0:
            lr = pf.optimizer.param_groups[0]['lr']
            print(f'  [{label}] ep {ep:5d} | total={losses[0]:.4e} | lr={lr:.2e}')
    elapsed = time.time() - t0
    V_pred = pf.predict(S_flat, tau_flat)
    rm  = rMAE(V_pred[mask_eval],  V_true[mask_eval])
    rr  = rRMSE(V_pred[mask_eval], V_true[mask_eval])
    print(f'  [{label}] DONE — rMAE={rm*100:.2f}%  rRMSE={rr*100:.2f}%  time={elapsed:.0f}s')
    return rm, rr, elapsed, np.array(hist), V_pred

results = {}

# ══════════════════════════════════════════════════════════════════════════════
# 1. Activation ablation: WaveAct vs tanh
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== 1. Activation Ablation ===')

for act_name, act_cls in [('WaveAct', None), ('tanh', nn.Tanh)]:
    torch.manual_seed(SEED)
    pf = BS_PINNsFormer(d_model=64, d_hidden=128, N=3, heads=4,
                         num_step=5, step_size=1e-4,
                         lam1=1.0, lam2=20.0, lam3=20.0,
                         lr=5e-4, device='cpu')
    if act_cls is not None:
        replace_waveact(pf.model, act_cls)
    rm, rr, elapsed, hist, V_pred = train_model(f'act={act_name}', pf)
    results[f'act_{act_name}'] = dict(rMAE=rm, rRMSE=rr, time=elapsed,
                                       hist=hist.tolist(), V_pred=V_pred.tolist())

# ══════════════════════════════════════════════════════════════════════════════
# 2. Sequence length ablation: L ∈ {1, 3, 5, 10}
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== 2. Sequence Length Ablation ===')

for L in [1, 3, 5, 10]:
    torch.manual_seed(SEED)
    pf = BS_PINNsFormer(d_model=64, d_hidden=128, N=3, heads=4,
                         num_step=L, step_size=1e-4,
                         lam1=1.0, lam2=20.0, lam3=20.0,
                         lr=5e-4, device='cpu')
    rm, rr, elapsed, hist, V_pred = train_model(f'L={L}', pf)
    results[f'L_{L}'] = dict(rMAE=rm, rRMSE=rr, time=elapsed,
                              hist=hist.tolist(), V_pred=V_pred.tolist())

# ══════════════════════════════════════════════════════════════════════════════
# 3. Step size ablation: δ ∈ {1e-5, 1e-4, 1e-3}
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== 3. Step Size Ablation ===')

for delta in [1e-5, 1e-4, 1e-3]:
    torch.manual_seed(SEED)
    pf = BS_PINNsFormer(d_model=64, d_hidden=128, N=3, heads=4,
                         num_step=5, step_size=delta,
                         lam1=1.0, lam2=20.0, lam3=20.0,
                         lr=5e-4, device='cpu')
    rm, rr, elapsed, hist, V_pred = train_model(f'delta={delta:.0e}', pf)
    results[f'delta_{delta:.0e}'] = dict(rMAE=rm, rRMSE=rr, time=elapsed,
                                          hist=hist.tolist(), V_pred=V_pred.tolist())

# ── Save results ──────────────────────────────────────────────────────────────
with open(f'{OUT}/ablation_results.json', 'w') as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ('hist','V_pred')}
               for k, v in results.items()}, f, indent=2)
print('\nResults saved to bs_pinn/ablation_results.json')

# ── Figure 1: Activation ablation bar ────────────────────────────────────────
act_names = ['WaveAct', 'tanh']
act_rmaes  = [results[f'act_{n}']['rMAE']*100 for n in act_names]
act_rrmses = [results[f'act_{n}']['rRMSE']*100 for n in act_names]

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(2); w = 0.35
b1 = ax.bar(x-w/2, act_rmaes,  w, label='rMAE (%)',  color=['purple','steelblue'], alpha=0.9)
b2 = ax.bar(x+w/2, act_rrmses, w, label='rRMSE (%)', color=['purple','steelblue'], alpha=0.5)
for bar, val in zip(list(b1)+list(b2), act_rmaes+act_rrmses):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(act_names)
ax.set_ylabel('Error (%)'); ax.set_title('Activation Ablation: WaveAct vs tanh')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_ablation_activation.png', dpi=120, bbox_inches='tight')
plt.close()

# ── Figure 2: Sequence length ─────────────────────────────────────────────────
Ls      = [1, 3, 5, 10]
L_rmaes  = [results[f'L_{L}']['rMAE']*100 for L in Ls]
L_rrmses = [results[f'L_{L}']['rRMSE']*100 for L in Ls]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(Ls, L_rmaes,  'o-', label='rMAE (%)',  color='purple', linewidth=2)
ax.plot(Ls, L_rrmses, 's--', label='rRMSE (%)', color='coral',  linewidth=2)
ax.set_xlabel('Sequence Length L'); ax.set_ylabel('Error (%)')
ax.set_title('Sequence Length Ablation')
ax.set_xticks(Ls); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_ablation_seqlen.png', dpi=120, bbox_inches='tight')
plt.close()

# ── Figure 3: Step size ───────────────────────────────────────────────────────
deltas      = [1e-5, 1e-4, 1e-3]
delta_rmaes  = [results[f'delta_{d:.0e}']['rMAE']*100 for d in deltas]
delta_rrmses = [results[f'delta_{d:.0e}']['rRMSE']*100 for d in deltas]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(3), delta_rmaes,  'o-',  label='rMAE (%)',  color='purple', linewidth=2)
ax.plot(range(3), delta_rrmses, 's--', label='rRMSE (%)', color='coral',  linewidth=2)
ax.set_xticks(range(3)); ax.set_xticklabels([f'{d:.0e}' for d in deltas])
ax.set_xlabel('Step Size δ'); ax.set_ylabel('Error (%)')
ax.set_title('Step Size Ablation')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_ablation_stepsize.png', dpi=120, bbox_inches='tight')
plt.close()

print('\nAll figures saved.')
print('\n=== Final Summary ===')
print(f'{"Config":<20} | {"rMAE":>8} | {"rRMSE":>8} | {"Time":>8}')
print('-'*55)
for k, v in results.items():
    print(f'{k:<20} | {v["rMAE"]*100:>7.2f}% | {v["rRMSE"]*100:>7.2f}% | {v["time"]:>6.0f}s')
