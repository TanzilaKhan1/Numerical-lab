import math
import matplotlib.pyplot as plt


def bisect(f, a, b, max_it=100, rel_er_thr=None):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    
    results = []
    
    print("Iter | a       | b       | c       | f(c)        | Ea (%)       ")
    old_c = None
    it = 0
    
    while it < max_it:
        c = (a + b) / 2
        fc = f(c)
        
        if it == 0:
            ea = "-"
        if old_c is not None:
            ea = abs((c - old_c) / c) * 100
        else:
            ea = None

        print(f"{it+1:3d} | {a:.10f} | {b:.10f} | {c:.10f} | {fc:.10f} | {ea} ")

        result = {
            'Iter': it+1,
            'xl': round(a, 10),
            'xu': round(b, 10),
            'xr': round(c, 10),
            'f(xr)': round(fc, 10),
            'ea(%)': round(ea, 10) if it > 0 else None 
        }
        results.append(result)

        if fc == 0:
            return c, it+1
        
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        
        if ea is not None and ea < rel_er_thr:      
            break
        
        old_c = c
        it += 1
        
    root = c
    return root, results, it+1



def f1(x):
    return 225 + 82*x - 90*x*x + 44*x**3 - 8*x**4 + 0.7*x**5

r, results, it = bisect(f1, -1.2, -1.0, rel_er_thr=0.05)
print(f"Root: {r}, Iterations: {it}\n")


iters = [r['Iter'] for r in results if r['ea(%)'] != "N/A"]
abs_ers = [r['ea(%)'] for r in results if r['ea(%)'] != "N/A"]
    
plt.figure(figsize=(10, 6))
plt.plot(iters, abs_ers, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Iteration Number')
plt.ylabel('Relative Error (%)')
plt.title('Convergence of Bisection Method')
plt.grid(True, alpha=0.3)
plt.show()     
