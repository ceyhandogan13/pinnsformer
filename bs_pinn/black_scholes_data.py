"""
Black-Scholes synthetic dataset generation.

Domain transfer from PINNsFormer notation:
    Physical:  x (space),  t (time)
    Financial: S (asset price), tau (time-to-maturity, tau = T - t)

Analytical Black-Scholes formulas for European call and put options.
"""

import numpy as np
from scipy.stats import norm


# ── analytical solution ──────────────────────────────────────────────────────

def bs_d1(S, K, r, sigma, tau):
    """d1 term of Black-Scholes formula. tau must be > 0."""
    return (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))


def bs_d2(S, K, r, sigma, tau):
    return bs_d1(S, K, r, sigma, tau) - sigma * np.sqrt(tau)


def bs_call(S, K, r, sigma, tau):
    """European call price. Returns 0 at tau=0 (payoff boundary)."""
    # at expiry: intrinsic value
    mask = tau <= 0.0
    d1 = np.where(mask, 0.0, bs_d1(S, K, r, sigma, np.maximum(tau, 1e-10)))
    d2 = np.where(mask, 0.0, bs_d2(S, K, r, sigma, np.maximum(tau, 1e-10)))
    price = np.where(
        mask,
        np.maximum(S - K, 0.0),
        S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2),
    )
    return price


def bs_put(S, K, r, sigma, tau):
    """European put price via put-call parity."""
    return bs_call(S, K, r, sigma, tau) - S + K * np.exp(-r * tau)


# ── dataset builder ──────────────────────────────────────────────────────────

def generate_dataset(
    S_min=20.0,
    S_max=200.0,
    tau_min=0.0,
    tau_max=1.0,
    K=100.0,
    r=0.05,
    sigma=0.2,
    n_S=100,
    n_tau=100,
    option_type="call",
    seed=42,
):
    """
    Returns a dictionary with:
        S_grid, tau_grid : 1-D arrays of the axes
        S_mesh, tau_mesh : 2-D meshgrids (shape n_tau x n_S)
        V_mesh           : analytical option price on the grid (n_tau x n_S)
        flat             : dict with 1-D arrays S, tau, V  (n_S * n_tau,)
        params           : dict of K, r, sigma
    """
    rng = np.random.default_rng(seed)

    S_grid = np.linspace(S_min, S_max, n_S)
    tau_grid = np.linspace(tau_min, tau_max, n_tau)

    S_mesh, tau_mesh = np.meshgrid(S_grid, tau_grid)  # (n_tau, n_S)

    pricer = bs_call if option_type == "call" else bs_put
    V_mesh = pricer(S_mesh, K, r, sigma, tau_mesh)

    return {
        "S_grid": S_grid,
        "tau_grid": tau_grid,
        "S_mesh": S_mesh,
        "tau_mesh": tau_mesh,
        "V_mesh": V_mesh,
        "flat": {
            "S": S_mesh.ravel(),
            "tau": tau_mesh.ravel(),
            "V": V_mesh.ravel(),
        },
        "params": {"K": K, "r": r, "sigma": sigma},
    }


# ── collocation point samplers ────────────────────────────────────────────────

def sample_residual_points(n_res, S_min=20, S_max=200, tau_min=1e-3, tau_max=1.0, seed=0):
    """Interior collocation points for the PDE residual loss."""
    rng = np.random.default_rng(seed)
    S = rng.uniform(S_min, S_max, n_res)
    tau = rng.uniform(tau_min, tau_max, n_res)
    return S, tau


def sample_terminal_points(n_tc, S_min=20, S_max=200, K=100, r=0.05, sigma=0.2, seed=1):
    """
    Terminal condition (tau = 0): V(S, 0) = max(S-K, 0) for call.
    """
    rng = np.random.default_rng(seed)
    S = rng.uniform(S_min, S_max, n_tc)
    tau = np.zeros(n_tc)
    V = np.maximum(S - K, 0.0)   # call payoff
    return S, tau, V


def sample_boundary_points(n_bc, tau_min=1e-3, tau_max=1.0,
                            S_min=20, S_max=200,
                            K=100, r=0.05, sigma=0.2, seed=2):
    """
    Boundary conditions:
        V(S_min, tau) ≈ 0          (deep OTM call)
        V(S_max, tau) ≈ S_max - K*exp(-r*tau)  (deep ITM call)
    """
    rng = np.random.default_rng(seed)
    tau = rng.uniform(tau_min, tau_max, n_bc)

    S_left = np.full(n_bc, S_min)
    V_left = bs_call(S_left, K, r, sigma, tau)

    S_right = np.full(n_bc, S_max)
    V_right = bs_call(S_right, K, r, sigma, tau)

    return (S_left, tau, V_left), (S_right, tau, V_right)


if __name__ == "__main__":
    ds = generate_dataset(option_type="call")
    print("Call dataset V_mesh shape:", ds["V_mesh"].shape)
    print("Sample prices at S=100, tau=0.5:",
          bs_call(100, ds["params"]["K"], ds["params"]["r"], ds["params"]["sigma"], 0.5))
