import math
import matplotlib.pyplot as plt
from typing import List

def pt_false_position(rows: List[dict], title: str):
    print(f"\n{title}")
    header = [
        "Iter", "xl", "xu", "xr", "f(xl)", "f(xu)", "f(xr)", "Ea(%)"
    ]
    print("{:>4s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s}".format(*header))
    for r in rows:
        def fmt(x):
            return f"{x:.10f}" if x is not None else "".ljust(10)
        print("{:4d} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s}".format(
            r["Iter"],
            fmt(r["xl"]),
            fmt(r["xu"]),
            fmt(r["xr"]),
            fmt(r["f_xl"]),
            fmt(r["f_xu"]),
            fmt(r["f_xr"]),
            fmt(r["Ea(%)"]),
        ))



def false_position(f, xl, xu, max_it=100, rel_er_thr=None):
    fl, fu = f(xl), f(xu)
    if fl * fu > 0:
        raise ValueError("f(xl) and f(xu) must have opposite signs")

    results = []
    old_xr = None
    it = 0

    while it < max_it:
        xr = (xl * fu - xu * fl) / (fu - fl)
        fr = f(xr)

        if old_xr is not None:
            ea = abs((xr - old_xr) / xr) * 100
        else:
            ea = None

        result = {
            'Iter': it+1,
            'xl': xl,
            'xu': xu,
            'xr': xr,
            'f_xl': fl,
            'f_xu': fu,
            'f_xr': fr,
            'Ea(%)': ea
        }
        results.append(result)

        if fr == 0:
            pt_false_position(results, "--- False Position Method ---")
            return xr, results, it+1

        if fl * fr < 0:
            xu, fu = xr, fr
        else:
            xl, fl = xr, fr

        if ea is not None and rel_er_thr is not None and ea < rel_er_thr:
            break

        old_xr = xr
        it += 1

    pt_false_position(results, "--- False Position Method ---")
    return xr, results, it+1




def f(m):
    g = 9.81
    c = 15
    t = 10
    target_y = 36
    y = (g * m / c) * (1 - math.exp(-(c/m) * t))
    return y - target_y



r, results, it = false_position(f, 40, 80, max_it=50, rel_er_thr=0.001)
print(f"\nRoot: {r}, Iterations: {it}\n")


iters = [r['Iter'] for r in results if r['Ea(%)'] is not None]
abs_ers = [r['Ea(%)'] for r in results if r['Ea(%)'] is not None]


plt.figure(figsize=(10, 6))
plt.plot(iters, abs_ers, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Iteration Number')
plt.ylabel('Relative Error (%)')
plt.title('Convergence of False Position Method')
plt.grid(True, alpha=0.3)
plt.show()
