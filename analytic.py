import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt


# Parameters (part a)
tf = 32.0
g = 0.5
d = 0.01

engine1 = "engine.exe"
engine2 = "engine2.exe"
config = "configuration.in.example"

# tau definition: N(tau)/Ninf = Nr
Nr = 0.2

# Time steps to test
dt = tf/32

tols = np.logspace(-8, -1, 16)


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
tau_analytic_by_method = {}
tau_iterative_by_method = {}



methods = [
    ("semi-implicit", 0.5),
    ("implicit", 0.0),
]

os.makedirs("ex1e_analytic_tau", exist_ok=True)
os.makedirs("ex1e_analytic_tau_engine1", exist_ok=True)
os.makedirs("ex1e_analytic_tau_engine2", exist_ok=True)

plt.figure(figsize=(7, 5))

for name, alpha in methods:
    tau_iterative = []

    # engine 1 (iterative)
    out = f"ex1e_analytic_tau_engine1/{name}_dt={dt:.15g}.out"

    cmd = (
        f".\\{engine2} {config} "
        f"tf={tf} dt={dt:.15g} N0=0 g={g} d={d} alpha={alpha} "
        f"sampling=1 output={out}"
    )

    subprocess.run(cmd, shell=True, check=True)
    t, N = read_t_N(out)
    tau_analytic = tau_from_curve(t, N)
    tau_analytic_by_method[name] = tau_analytic

    for tol in tols:
        # engine 2 (analytic)
        out = f"ex1e_analytic_tau_engine2/{name}_dt={dt:.15g}.out"

        cmd = (
            f".\\{engine1} {config} "
            f"tf={tf} dt={dt:.15g} N0=0 g={g} d={d} alpha={alpha} "
            f"sampling=1 output={out} tol={tol}"
        )

        subprocess.run(cmd, shell=True, check=True)
        t, N = read_t_N(out)
        tau_it = tau_from_curve(t, N)
        tau_iterative.append(tau_it)

    tau_iterative_by_method[name] = np.array(tau_iterative, dtype=float)


# plot tau_ref, tau_analytic, and tau_iterative:
# for low tol values, tau_iterative should approach tau_analytic and tau_ref

for name, _ in methods:
    y = tau_iterative_by_method[name]
    plt.semilogx(tols, y, "o-", label=f"tau_iterative ({name})")
    plt.axhline(tau_analytic_by_method[name], linestyle="--", linewidth=1,
                label=f"tau_analytic ({name})")

plt.axhline(tau_ref, color="k", linestyle=":", linewidth=2, label="tau_ref")

plt.xlabel("tol")
plt.ylabel("tau")
plt.title("Tau comparison vs tolerance")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("ex1e_analytic_tau/tau_vs_tol.png", dpi=200)
print("Saved ex1e_analytic_tau/tau_vs_tol.png")
plt.show()




    

