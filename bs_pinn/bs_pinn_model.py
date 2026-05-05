"""
Black-Scholes PINN and PINNsFormer wrappers.

Domain mapping:
    x  →  S  (asset price,    normalised to [0,1])
    t  →  τ  (time-to-maturity, normalised to [0,1])

Both models receive (S_norm, tau_norm) tensors of shape (N, 1)
and return V_pred of the same shape.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pinnsformer'))

import torch
import torch.nn as nn
import numpy as np

from model.pinn import PINNs
from model.pinnsformer import PINNsformer
from util import make_time_sequence


# ── normalisation helpers ─────────────────────────────────────────────────────

class BSNormalizer:
    """Normalise S and tau to [0,1] for network input."""

    def __init__(self, S_min, S_max, tau_min, tau_max):
        self.S_min, self.S_max = S_min, S_max
        self.tau_min, self.tau_max = tau_min, tau_max

    def norm_S(self, S):
        return (S - self.S_min) / (self.S_max - self.S_min)

    def norm_tau(self, tau):
        return (tau - self.tau_min) / (self.tau_max - self.tau_min)

    def to_tensor(self, arr, device='cpu', requires_grad=False):
        t = torch.tensor(arr, dtype=torch.float32, device=device)
        if requires_grad:
            t.requires_grad_(True)
        return t.unsqueeze(-1) if t.dim() == 1 else t


# ── Black-Scholes PDE residual ────────────────────────────────────────────────

def bs_pde_residual(V, S_phys, tau, sigma=0.2, r=0.05):
    """
    Compute the Black-Scholes PDE residual:
        dV/dtau - 0.5*sigma^2*S^2*d2V/dS2 - r*S*dV/dS + r*V = 0

    V, S_phys, tau : tensors with requires_grad=True, shape (N, 1) or (N, L, 1).
    Returns residual tensor of same shape.
    """
    dV_dtau = torch.autograd.grad(
        V, tau,
        grad_outputs=torch.ones_like(V),
        create_graph=True, retain_graph=True
    )[0]

    dV_dS = torch.autograd.grad(
        V, S_phys,
        grad_outputs=torch.ones_like(V),
        create_graph=True, retain_graph=True
    )[0]

    d2V_dS2 = torch.autograd.grad(
        dV_dS, S_phys,
        grad_outputs=torch.ones_like(dV_dS),
        create_graph=True, retain_graph=True
    )[0]

    residual = (dV_dtau
                - 0.5 * sigma**2 * S_phys**2 * d2V_dS2
                - r * S_phys * dV_dS
                + r * V)
    return residual


# ── PINN wrapper for Black-Scholes ────────────────────────────────────────────

class BS_PINN:
    """
    Wraps the baseline PINN model with Black-Scholes physics.

    Loss = lambda1*L_pde + lambda2*L_tc + lambda3*L_bc
    """

    def __init__(self, hidden_dim=64, num_layer=6, sigma=0.2, r=0.05,
                 lam1=1.0, lam2=10.0, lam3=10.0, lr=1e-3, device='cpu'):
        self.model = PINNs(in_dim=2, hidden_dim=hidden_dim, out_dim=1, num_layer=num_layer).to(device)
        self.sigma = sigma
        self.r = r
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.history = []

    def _to(self, arr, grad=False):
        t = torch.tensor(arr, dtype=torch.float32, device=self.device)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if grad:
            t.requires_grad_(True)
        return t

    def loss(self, S_res, tau_res,
             S_tc, tau_tc, V_tc,
             S_bc_l, tau_bc_l, V_bc_l,
             S_bc_r, tau_bc_r, V_bc_r):
        """Compute composite loss. All inputs are numpy arrays (physical units)."""

        # --- residual points ---
        S_r = self._to(S_res, grad=True)
        t_r = self._to(tau_res, grad=True)
        V_r = self.model(S_r, t_r)
        res = bs_pde_residual(V_r, S_r, t_r, self.sigma, self.r)
        L_pde = torch.mean(res**2)

        # --- terminal condition (tau=0) ---
        S_t = self._to(S_tc)
        t_t = self._to(tau_tc)
        V_pred_tc = self.model(S_t, t_t)
        V_true_tc = self._to(V_tc)
        L_tc = torch.mean((V_pred_tc - V_true_tc)**2)

        # --- boundary conditions ---
        S_bl = self._to(S_bc_l)
        t_bl = self._to(tau_bc_l)
        V_bl = self.model(S_bl, t_bl)
        V_true_bl = self._to(V_bc_l)

        S_br = self._to(S_bc_r)
        t_br = self._to(tau_bc_r)
        V_br = self.model(S_br, t_br)
        V_true_br = self._to(V_bc_r)

        L_bc = torch.mean((V_bl - V_true_bl)**2) + torch.mean((V_br - V_true_br)**2)

        total = self.lam1 * L_pde + self.lam2 * L_tc + self.lam3 * L_bc
        return total, L_pde, L_tc, L_bc

    def train_step(self, *args):
        self.optimizer.zero_grad()
        total, L_pde, L_tc, L_bc = self.loss(*args)
        total.backward()
        self.optimizer.step()
        return (total.item(), L_pde.item(), L_tc.item(), L_bc.item())

    def predict(self, S_arr, tau_arr):
        self.model.eval()
        with torch.no_grad():
            S_t = self._to(S_arr)
            t_t = self._to(tau_arr)
            V = self.model(S_t, t_t)
        self.model.train()
        return V.squeeze(-1).cpu().numpy()


# ── PINNsFormer wrapper for Black-Scholes (finite-difference PDE residual) ────
#
# Design rationale
# ----------------
# Computing d²V/dS² via autograd through a Transformer requires
# create_graph=True, which makes the backward pass ~100× slower than PINN.
# Instead we use central finite differences for all PDE partial derivatives.
# Gradients for weight updates still flow correctly through the model outputs.
# All collocation tensors are pre-computed once (numpy→tensor conversion
# is moved out of the inner training loop).

class BS_PINNsFormer:
    """
    PINNsFormer adapted to Black-Scholes with finite-difference PDE residual.
    Call .precompute() once before training to cache tensors.
    """

    def __init__(self, d_model=64, d_hidden=128, N=3, heads=4,
                 num_step=5, step_size=1e-4,
                 sigma=0.2, r=0.05,
                 lam1=1.0, lam2=10.0, lam3=10.0,
                 lr=1e-3, device='cpu',
                 eps_S=1.0, eps_tau=1e-3):
        self.model = PINNsformer(d_out=1, d_model=d_model, d_hidden=d_hidden,
                                  N=N, heads=heads).to(device)
        self.sigma    = sigma
        self.r        = r
        self.lam1     = lam1
        self.lam2     = lam2
        self.lam3     = lam3
        self.num_step = num_step
        self.step_size = step_size
        self.eps_S    = eps_S
        self.eps_tau  = eps_tau
        self.device   = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.history  = []
        # cached tensors (set by precompute)
        self._cache   = {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _t(self, arr):
        """numpy → (N,1) float32 tensor on device."""
        t = torch.tensor(np.asarray(arr, dtype=np.float32), device=self.device)
        return t.unsqueeze(-1) if t.dim() == 1 else t

    def _make_seq_t(self, S_t, tau_t):
        """
        Build pseudo-sequence from already-converted (N,1) tensors.
        Returns S_seq (N,L,1), tau_seq (N,L,1).
        """
        L     = self.num_step
        S_seq = S_t.unsqueeze(1).expand(-1, L, -1)          # (N,L,1)
        steps = (torch.arange(L, dtype=torch.float32, device=self.device)
                 * self.step_size)
        tau_seq = tau_t.unsqueeze(1) + steps.view(1, L, 1)  # (N,L,1)
        return S_seq, tau_seq

    def _fwd(self, S_t, tau_t):
        """Single forward pass → (N,1) option price tensor."""
        S_seq, tau_seq = self._make_seq_t(S_t, tau_t)
        return self.model(S_seq, tau_seq)[:, 0, :]

    # ── finite-difference PDE residual ────────────────────────────────────────

    def _pde_fd(self, S_t, tau_t):
        """
        Black-Scholes residual via central differences.
        ∂V/∂τ - ½σ²S²∂²V/∂S² - rS∂V/∂S + rV = 0
        All 5 forward passes run with grad tracking for weight update.
        """
        eps_S   = self.eps_S
        eps_tau = self.eps_tau

        V_c  = self._fwd(S_t,           tau_t)
        V_sp = self._fwd(S_t + eps_S,   tau_t)
        V_sm = self._fwd(S_t - eps_S,   tau_t)
        V_tp = self._fwd(S_t,           tau_t + eps_tau)

        dV_dS   = (V_sp - V_sm) / (2 * eps_S)
        d2V_dS2 = (V_sp - 2 * V_c + V_sm) / (eps_S ** 2)
        dV_dtau = (V_tp - V_c) / eps_tau

        return (dV_dtau
                - 0.5 * self.sigma**2 * S_t**2 * d2V_dS2
                - self.r * S_t * dV_dS
                + self.r * V_c)

    # ── pre-compute tensors ────────────────────────────────────────────────────

    def precompute(self, S_res, tau_res,
                   S_tc, tau_tc, V_tc,
                   S_bc_l, tau_bc_l, V_bc_l,
                   S_bc_r, tau_bc_r, V_bc_r):
        """Convert all numpy arrays to tensors once; cache for training."""
        c = self._cache
        c['S_res']  = self._t(S_res)
        c['tau_res'] = self._t(tau_res)
        c['S_tc']   = self._t(S_tc)
        c['tau_tc'] = self._t(tau_tc)
        c['V_tc']   = self._t(V_tc)
        c['S_bcl']  = self._t(S_bc_l)
        c['tau_bcl'] = self._t(tau_bc_l)
        c['V_bcl']  = self._t(V_bc_l)
        c['S_bcr']  = self._t(S_bc_r)
        c['tau_bcr'] = self._t(tau_bc_r)
        c['V_bcr']  = self._t(V_bc_r)

    # ── loss ──────────────────────────────────────────────────────────────────

    def loss(self):
        """Compute composite loss from cached tensors."""
        c = self._cache
        if not c:
            raise RuntimeError('Call precompute() before loss().')

        # PDE residual (finite differences, 4 forward passes)
        res   = self._pde_fd(c['S_res'], c['tau_res'])
        L_pde = torch.mean(res ** 2)

        # Terminal condition
        V_pred_tc = self._fwd(c['S_tc'], c['tau_tc'])
        L_tc      = torch.mean((V_pred_tc - c['V_tc']) ** 2)

        # Boundary conditions
        V_bl   = self._fwd(c['S_bcl'], c['tau_bcl'])
        V_br   = self._fwd(c['S_bcr'], c['tau_bcr'])
        L_bc   = torch.mean((V_bl - c['V_bcl']) ** 2) + \
                 torch.mean((V_br - c['V_bcr']) ** 2)

        total  = self.lam1 * L_pde + self.lam2 * L_tc + self.lam3 * L_bc
        return total, L_pde, L_tc, L_bc

    # ── convenience: accept raw numpy (for backward-compat with PINN API) ─────

    def loss_np(self, S_res, tau_res,
                S_tc, tau_tc, V_tc,
                S_bc_l, tau_bc_l, V_bc_l,
                S_bc_r, tau_bc_r, V_bc_r):
        self.precompute(S_res, tau_res, S_tc, tau_tc, V_tc,
                        S_bc_l, tau_bc_l, V_bc_l, S_bc_r, tau_bc_r, V_bc_r)
        return self.loss()

    def train_step(self, clip_grad: float = 1.0):
        """One optimiser step.  Gradient norm is clipped to clip_grad (set None to disable)."""
        self.optimizer.zero_grad()
        total, L_pde, L_tc, L_bc = self.loss()
        total.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
        self.optimizer.step()
        return (total.item(), L_pde.item(), L_tc.item(), L_bc.item())

    def predict(self, S_arr, tau_arr):
        self.model.eval()
        with torch.no_grad():
            V = self._fwd(self._t(S_arr), self._t(tau_arr))
        self.model.train()
        return V.squeeze(-1).cpu().numpy()


# ── PINNsFormer wrapper — Autograd PDE residual (original, slow) ─────────────
#
# Kept for comparison. Uses create_graph=True for d²V/dS² through the
# Transformer, which is ~100× slower than the FD version above.

class BS_PINNsFormer_AD:
    """
    PINNsFormer with full autograd PDE residual (second-order, create_graph).
    Slower but exact derivatives. Use only for short runs / speed benchmarking.
    """

    def __init__(self, d_model=64, d_hidden=128, N=3, heads=4,
                 num_step=5, step_size=1e-4,
                 sigma=0.2, r=0.05,
                 lam1=1.0, lam2=10.0, lam3=10.0,
                 lr=1e-3, device='cpu'):
        self.model = PINNsformer(d_out=1, d_model=d_model, d_hidden=d_hidden,
                                  N=N, heads=heads).to(device)
        self.sigma    = sigma
        self.r        = r
        self.lam1, self.lam2, self.lam3 = lam1, lam2, lam3
        self.num_step = num_step
        self.step_size = step_size
        self.device   = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.history  = []

    def _t(self, arr, grad=False):
        t = torch.tensor(np.asarray(arr, dtype=np.float32), device=self.device)
        if t.dim() == 1: t = t.unsqueeze(-1)
        if grad: t.requires_grad_(True)
        return t

    def _make_seq(self, S_leaf, tau_leaf):
        L = self.num_step
        S_seq = S_leaf.unsqueeze(1).expand(-1, L, -1)
        steps = torch.arange(L, dtype=torch.float32, device=self.device) * self.step_size
        tau_seq = tau_leaf.unsqueeze(1) + steps.view(1, L, 1)
        return S_seq, tau_seq

    def loss(self, S_res, tau_res,
             S_tc, tau_tc, V_tc,
             S_bc_l, tau_bc_l, V_bc_l,
             S_bc_r, tau_bc_r, V_bc_r):

        # PDE residual via autograd (exact, slow)
        S_r   = self._t(S_res,   grad=True)
        tau_r = self._t(tau_res, grad=True)
        S_seq, tau_seq = self._make_seq(S_r, tau_r)
        V_r   = self.model(S_seq, tau_seq)[:, 0, :]
        res   = bs_pde_residual(V_r, S_r, tau_r, self.sigma, self.r)
        L_pde = torch.mean(res ** 2)

        # Terminal condition
        S_t_seq, tau_t_seq = self._make_seq(self._t(S_tc), self._t(tau_tc))
        V_tc_pred = self.model(S_t_seq, tau_t_seq)[:, 0, :]
        L_tc = torch.mean((V_tc_pred - self._t(V_tc)) ** 2)

        # Boundary conditions
        def _bc_loss(S_arr, tau_arr, V_arr):
            S_seq, tau_seq = self._make_seq(self._t(S_arr), self._t(tau_arr))
            V_pred = self.model(S_seq, tau_seq)[:, 0, :]
            return torch.mean((V_pred - self._t(V_arr)) ** 2)

        L_bc = _bc_loss(S_bc_l, tau_bc_l, V_bc_l) + _bc_loss(S_bc_r, tau_bc_r, V_bc_r)

        total = self.lam1 * L_pde + self.lam2 * L_tc + self.lam3 * L_bc
        return total, L_pde, L_tc, L_bc

    def train_step(self, *args):
        self.optimizer.zero_grad()
        total, L_pde, L_tc, L_bc = self.loss(*args)
        total.backward()
        self.optimizer.step()
        return (total.item(), L_pde.item(), L_tc.item(), L_bc.item())

    def predict(self, S_arr, tau_arr):
        self.model.eval()
        with torch.no_grad():
            S_t   = torch.tensor(np.asarray(S_arr, dtype=np.float32), device=self.device).unsqueeze(-1)
            tau_t = torch.tensor(np.asarray(tau_arr, dtype=np.float32), device=self.device).unsqueeze(-1)
            L = self.num_step
            S_seq   = S_t.unsqueeze(1).expand(-1, L, -1)
            steps   = torch.arange(L, dtype=torch.float32, device=self.device) * self.step_size
            tau_seq = tau_t.unsqueeze(1) + steps.view(1, L, 1)
            V = self.model(S_seq, tau_seq)[:, 0, :]
        self.model.train()
        return V.squeeze(-1).cpu().numpy()


# ── evaluation metrics ────────────────────────────────────────────────────────

def rMAE(V_pred, V_true):
    """Relative Mean Absolute Error."""
    return np.mean(np.abs(V_pred - V_true)) / (np.mean(np.abs(V_true)) + 1e-8)


def rRMSE(V_pred, V_true):
    """Relative Root Mean Square Error."""
    return np.sqrt(np.mean((V_pred - V_true)**2)) / (np.sqrt(np.mean(V_true**2)) + 1e-8)
