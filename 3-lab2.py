import math
import matplotlib.pyplot as plt
from typing import List

def pt_secant(rows: List[dict], title: str):
    print(f"\n{title}")
    header = [
        "Iter", "x_i-1", "x_i", "x_i+1", "f(x_i-1)", "f(x_i)", "f(x_i+1)", "Ea(%)"
    ]
    print("{:>4s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s}".format(*header))
    for r in rows:
        def fmt(x):
            return f"{x:.10f}" if x is not None else "".ljust(10)
        print("{:4d} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s}".format(
            r["Iter"],
            fmt(r["x0"]),
            fmt(r["x1"]),
            fmt(r["x2"]),
            fmt(r["f0"]),
            fmt(r["f1"]),
            fmt(r["f2"]),
            fmt(r["Ea(%)"]),
        ))



def secant(f, x0, x1, max_it=100, rel_er_thr=None, tol=1e-12):
    f0, f1 = f(x0), f(x1)
    results = []
    old_x2 = None
    it = 0

    while it < max_it:
        if abs(f1 - f0) < tol:
            print("Division by zero encountered.")
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = f(x2)
        if old_x2 is not None:
            ea = abs((x2 - old_x2) / x2) * 100
        else:
            ea = None
        result = { 'Iter': it+1, 'x0': x0, 'x1': x1, 'x2': x2, 'f0': f0, 'f1': f1, 'f2': f2, 'Ea(%)': ea }
        results.append(result)

        if abs(f2) < tol:
            break
        if ea is not None and rel_er_thr is not None and ea < rel_er_thr:
            break
        x0, f0 = x1, f1
        x1, f1 = x2, f2
        old_x2 = x2
        it += 1
    pt_secant(results, "--- Secant Method ---")
    return x2, results, it+1





def f(m):
    g = 9.81
    c = 15
    t = 10
    target_y = 36
    y = (g * m / c) * (1 - math.exp(-(c/m) * t))
    return y - target_y



r, results, it = secant(f, 30, 40, max_it=50, rel_er_thr=0.001)
print(f"\nRoot: {r}, Iterations: {it}\n")



iters = [r['Iter'] for r in results if r['Ea(%)'] is not None]
abs_ers = [r['Ea(%)'] for r in results if r['Ea(%)'] is not None]


plt.figure(figsize=(10, 6))
plt.plot(iters, abs_ers, 'go-', linewidth=2, markersize=8)
plt.xlabel('Iteration Number')
plt.ylabel('Relative Error (%)')
plt.title('Convergence of Secant Method')
plt.grid(True, alpha=0.3)
plt.show()
