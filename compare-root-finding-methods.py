import math
import matplotlib.pyplot as plt
from typing import List

def pt_bisection(rows: List[dict], title: str):
    print(f"\n{title}")
    header = [
        "Iter", "xl", "xu", "xr", "f(xl)", "f(xu)", "f(xr)", "Ea(%)"
    ]
    print("{:>4s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s}{:>14s}".format(*header))
    for r in rows:
        def fmt(x):
            return f"{x:.5f}" if x is not None else "".ljust(10)
        print("{:4d} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s}{:>14s}".format(
            r["Iter"],
            fmt(r["a"]),
            fmt(r["b"]),
            fmt(r["c"]),
            fmt(r["f_a"]),
            fmt(r["f_b"]),
            fmt(r["f_c"]),
            fmt(r["Ea(%)"]),
        ))


def pt_false_position(rows: List[dict], title: str):
    print(f"\n{title}")
    header = [
        "Iter", "xl", "xu", "xr", "f(xl)", "f(xu)", "f(xr)", "Ea(%)"
    ]
    print("{:>4s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s}".format(*header))
    for r in rows:
        def fmt(x):
            return f"{x:.5f}" if x is not None else "".ljust(10)
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


def pt_newton_raphson(rows: List[dict], title: str):
    print(f"\n{title}")
    header = [
        "Iter", "x_i", "f(x_i)", "f'(x_i)", "x_i+1", "Ea(%)"
    ]
    print("{:>4s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s}".format(*header))
    for r in rows:
        def fmt(x):
            return f"{x:.5f}" if x is not None else "".ljust(10)
        def fmt_sci(x):
            return f"{x:.5e}" if x is not None else "".ljust(10)
        print("{:4d} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s}".format(
            r["Iter"],
            fmt(r["x_i"]),
            fmt_sci(r["f_xi"]),
            fmt_sci(r["df_xi"]),
            fmt(r["x_next"]),
            fmt(r["Ea(%)"]),
        ))


def pt_secant(rows: List[dict], title: str):
    print(f"\n{title}")
    header = [
        "Iter", "x_i-1", "x_i", "x_i+1", "f(x_i-1)", "f(x_i)", "f(x_i+1)", "Ea(%)"
    ]
    print("{:>4s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s}".format(*header))
    for r in rows:
        def fmt(x):
            return f"{x:.5f}" if x is not None else "".ljust(10)
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




# -------------------------------
# Bisection Method
# -------------------------------
def bisection(f, a, b, max_it=100, rel_er_thr=None):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    results = []
    old_c = None
    it = 0
    
    while it < max_it:
        c = (a + b) / 2
        fc = f(c)
        if old_c is not None:
            ea = abs((c - old_c) / c) * 100
        else:
            ea = None
        result = { 'Iter': it+1, 'a': a, 'b': b, 'c': c, 'f_a': fa, 'f_b': fb, 'f_c': fc, 'Ea(%)': ea }
        results.append(result)
        if fc == 0:
            pt_bisection(results, "--- Bisection Method ---")
            return c, results, it+1
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        if ea is not None and rel_er_thr is not None and ea < rel_er_thr:      
            break
        old_c = c
        it += 1
    pt_bisection(results, "--- Bisection Method ---")
    return c, results, it+1


# -------------------------------
# False Position Method
# -------------------------------
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
        result = { 'Iter': it+1, 'xl': xl, 'xu': xu, 'xr': xr, 'f_xl': fl, 'f_xu': fu, 'f_xr': fr,'Ea(%)': ea}
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


# -------------------------------
# Newton-Raphson Method
# -------------------------------
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
        
        result = { 'Iter': it+1, 'x_i': xi, 'f_xi': fxi, 'df_xi': dfxi, 'x_next': x_next, 'Ea(%)': ea }
        results.append(result)

        if ea is not None and rel_er_thr is not None and ea < rel_er_thr:
            break

        xi = x_next
        it += 1

    pt_newton_raphson(results, "--- Newton-Raphson Method ---")
    return x_next, results, it


# -------------------------------
# Secant Method
# -------------------------------
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


# function and its derivative
def f(h):
    exp_term = math.exp(-h/2)
    return h**3 - 10*h + 5*exp_term - 2


def df(h):
    exp_term = math.exp(-h/2)
    return 3*h**2 - 10 - (5/2)*exp_term

# -------------------------------
# Run All Methods
# -------------------------------
errors = {}

# Bisection
root_b, res_b, _ = bisection(f, 0.1, 0.4, rel_er_thr=0.001)
print(f"\nRoot: {root_b}\n")
errors["Bisection"] = [r['Ea(%)'] for r in res_b if r['Ea(%)'] is not None]

# False Position
root_fp, res_fp, _ = false_position(f, 0.1, 0.4, rel_er_thr=0.001)
print(f"\nRoot: {root_fp}\n")
errors["False Position"] = [r['Ea(%)'] for r in res_fp if r['Ea(%)'] is not None]

# Newton-Raphson
root_nr, res_nr, _ = newton_raphson(f, df, x0=1.5, rel_er_thr=0.001)
print(f"\nRoot: {root_nr}\n")
errors["Newton-Raphson"] = [r['Ea(%)'] for r in res_nr if r['Ea(%)'] is not None]

# Secant
root_sec, res_sec, _ = secant(f, 1.5, 2.0, rel_er_thr=0.001)
print(f"\nRoot: {root_sec}\n")
errors["Secant"] = [r['Ea(%)'] for r in res_sec if r['Ea(%)'] is not None]

# -------------------------------
# Plot All Together
# -------------------------------
plt.figure(figsize=(10, 6))
for method, errs in errors.items():
    plt.plot(range(2, len(errs)+2), errs, marker='o', linewidth=1, label=method)

plt.xlabel("Iteration Number")
plt.ylabel("Relative Error (%)")
plt.title("Comparison of Convergence for Root-Finding Methods")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()