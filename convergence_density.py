import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt


# Problem parameters (part a)
tf = 32.0
g = 0.5
d = 0.01

engine = "engine.exe"
config = "configuration.in.example"

# Time steps to test
# (tf/dt will be the x-axis: nsteps)
dt_list = tf / (2 ** np.arange(7, 12))  # 8, 4, 2, ..., 0.25

nsteps = (tf / dt_list).astype(int)


# Exact final value N_th(tf) for dN/dt = d + N(g - N) with N(0)=0
beta = np.sqrt(g * g + 4 * d)
N_th = 2 * d * (1 - np.exp(-beta * tf)) / (beta - g + (beta + g) * np.exp(-beta * tf))


def final_N_from_file(path):
    lines = open(path, "r").read().splitlines()
    data = np.loadtxt(lines[:-1])  # last line is "Total steps: ..."
    return float(data[-1, 1])


def fit_order(x, y):
    # fit log(y) = a*log(x) + b
    a, b = np.polyfit(np.log(x), np.log(y), 1)
    p = -a  # y ~ x^a = 1/x^p
    yfit = np.exp(b) * x ** a
    return p, yfit


methods = [
    ("explicit", 1.0),
    ("semi-implicit", 0.5),
    ("implicit", 0.0),
]

os.makedirs("conv_outputs", exist_ok=True)

plt.figure(figsize=(7, 5))

for name, alpha in methods:
    err = []

    for dt in dt_list:
        out = f"conv_outputs/{name}_dt={dt:.15g}.out"

        cmd = (
            f".\\{engine} {config} "
            f"tf={tf} dt={dt:.15g} N0=0 g={g} d={d} alpha={alpha} "
            f"sampling=1 output={out}"
        )

        subprocess.run(cmd, shell=True, check=True)
        N_num = final_N_from_file(out)
        err.append(abs(N_num - N_th))

    err = np.array(err)
    err = np.maximum(err, 1e-300)  # avoid log(0)

    p, yfit = fit_order(nsteps, err)
    p_round = int(round(p))

    plt.loglog(nsteps, err, "o-", label=f"{name}: p={p:.2f} (~1/N^{p_round})")
    plt.loglog(nsteps, yfit, "--", linewidth=1)


plt.xlabel("nsteps (= tf/dt)")
plt.ylabel("abs error |N_num(tf) - N_th(tf)|")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("convergence_final_density.png", dpi=200)
print("Saved convergence_final_density.png")
