import math
import matplotlib.pyplot as plt
from typing import List

def pt_newton_raphson(rows: List[dict], title: str):
    print(f"\n{title}")
    header = [
        "Iter", "x_i", "f(x_i)", "f'(x_i)", "x_i+1", "Ea(%)"
    ]
    print("{:>4s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s}".format(*header))
    for r in rows:
        def fmt(x):
            return f"{x:.6f}" if x is not None else "".ljust(10)
        def fmt_sci(x):
            return f"{x:.6e}" if x is not None else "".ljust(10)
        print("{:4d} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s}".format(
            r["Iter"],
            fmt(r["x_i"]),
            fmt_sci(r["f_xi"]),
            fmt_sci(r["df_xi"]),
            fmt(r["x_next"]),
            fmt(r["Ea(%)"]),
        ))


def newton_raphson(f, df, x0, max_it=100, rel_er_thr=None):
    results = []
    
    xi = x0
    it = 0

    while it < max_it:
        fxi = f(xi)
        dfxi = df(xi)
        if dfxi == 0:
            raise ZeroDivisionError("Derivative became zero. Method fails.")

        x_next = xi - fxi / dfxi

        if it == 0:
            ea = None
        else:
            ea = abs((x_next - xi) / x_next) * 100

        result = {
            'Iter': it+1,
            'x_i': xi,
            'f_xi': fxi,
            'df_xi': dfxi,
            'x_next': x_next,
            'Ea(%)': ea
        }
        results.append(result)

        if ea is not None and rel_er_thr is not None and ea < rel_er_thr:
            break

        xi = x_next
        it += 1

    pt_newton_raphson(results, "--- Newton-Raphson Method ---")
    return x_next, results, it



def f(m):
    g = 9.81
    c = 15
    t = 10
    target_y = 36
    y = (g * m / c) * (1 - math.exp(-(c/m) * t))
    return y - target_y


def df(m):
    g = 9.81
    c = 15
    t = 10
    exp_term = math.exp(-(c/m) * t)
    return (g/c) * (1 - exp_term) - (g*t/m) * exp_term




r, results, it = newton_raphson(f, df, x0=40, max_it=50, rel_er_thr=0.001)
print(f"\nRoot: {r}\n")


iters = [r['Iter'] for r in results if r['Ea(%)'] is not None]
abs_ers = [r['Ea(%)'] for r in results if r['Ea(%)'] is not None]

plt.figure(figsize=(10, 6))
plt.plot(iters, abs_ers, 'go-', linewidth=2, markersize=8)
plt.xlabel('Iteration Number')
plt.ylabel('Relative Error (%)')
plt.title('Convergence of Newton-Raphson Method')
plt.grid(True, alpha=0.3)
plt.show()
