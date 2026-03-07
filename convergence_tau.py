import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt


# Parameters (part a)
tf = 32.0
g = 0.5
d = 0.01

engine = "engine.exe"
config = "configuration.in.example"

# tau definition: N(tau)/Ninf = Nr
Nr = 0.2

# Time steps to test
dt_list = tf / (2 ** np.arange(6, 10))
nsteps = (tf / dt_list).astype(int)


beta = np.sqrt(g * g + 4 * d)
Ninf = 0.5 * (g + beta)


def read_t_N(path):
    lines = open(path, "r").read().splitlines()
    data = np.loadtxt(lines[:-1])  # last line is "Total steps: ..."
    return data[:, 0], data[:, 1]


def tau_from_curve(t, N):
    ratio = N / Ninf

    # If we never reach Nr, tau is undefined
    if ratio[0] > Nr or ratio[-1] < Nr:
        return np.nan

    return float(np.interp(Nr, ratio, t))


def fit_order(x, y):
    a, b = np.polyfit(np.log(x), np.log(y), 1)
    p = -a
    yfit = np.exp(b) * x ** a
    return p, yfit


# Analytical reference for tau: use exact solution on a very fine time grid + interpolation
t_ref = np.linspace(0, tf, 200000)
N_exact = 2 * d * (1 - np.exp(-beta * t_ref)) / (beta - g + (beta + g) * np.exp(-beta * t_ref))
tau_ref = tau_from_curve(t_ref, N_exact)


methods = [
    ("explicit", 1.0),
    ("semi-implicit", 0.5),
    ("implicit", 0.0),
]

os.makedirs("conv_outputs_tau", exist_ok=True)

plt.figure(figsize=(7, 5))

for name, alpha in methods:
    tau_list = []

    for dt in dt_list:
        out = f"conv_outputs_tau/{name}_dt={dt:.15g}.out"

        cmd = (
            f".\\{engine} {config} "
            f"tf={tf} dt={dt:.15g} N0=0 g={g} d={d} alpha={alpha} "
            f"sampling=1 output={out}"
        )

        subprocess.run(cmd, shell=True, check=True)
        t, N = read_t_N(out)
        tau_list.append(tau_from_curve(t, N))

    tau_list = np.array(tau_list, dtype=float)

    # relative error on tau
    err = np.abs(1 - tau_list / tau_ref)
    ok = np.isfinite(err)
    err = np.maximum(err, 1e-300)

    if ok.sum() >= 2:
        p, yfit = fit_order(nsteps[ok], err[ok])
        p_round = int(round(p))
        plt.loglog(nsteps[ok], err[ok], "o-", label=f"{name}: p={p:.2f} (~1/N^{p_round})")
        plt.loglog(nsteps[ok], yfit, "--", linewidth=1)
    else:
        plt.loglog(nsteps[ok], err[ok], "o-", label=f"{name}: not enough points")


plt.xlabel("nsteps (= tf/dt)")
plt.ylabel("relative error |1 - tau_num/tau_ref|")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("convergence_tau.png", dpi=200)
print("Saved convergence_tau.png")
