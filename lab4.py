import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial


def lagrange_interpolation(x_points, y_points, x_eval):
    n = len(x_points)
    
    def evaluate(x):
        result = 0.0
        for i in range(n):
            term = y_points[i]
            for j in range(n):
                if i != j:
                    term *= (x - x_points[j]) / (x_points[i] - x_points[j])
            result += term
        return result
    
    if np.isscalar(x_eval):
        return evaluate(x_eval)
    else:
        return np.array([evaluate(x) for x in x_eval])


def newton_interpolation(x_points, y_points, x_eval):
    n = len(x_points)
    divided_diff = np.zeros((n, n))
    divided_diff[:, 0] = y_points
    
    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i][j] = (divided_diff[i + 1][j - 1] - divided_diff[i][j - 1]) / \
                                  (x_points[i + j] - x_points[i])
    
    def evaluate(x):
        result = divided_diff[0][n - 1]
        for i in range(n - 2, -1, -1):
            result = result * (x - x_points[i]) + divided_diff[0][i]
        return result
    
    if np.isscalar(x_eval):
        return evaluate(x_eval), divided_diff
    else:
        return np.array([evaluate(x) for x in x_eval]), divided_diff


def get_polynomial_coeffs(x_points, y_points):
    result = Polynomial([0])
    for i in range(len(x_points)):
        Li = Polynomial([1])
        for j in range(len(x_points)):
            if i != j:
                Li = Li * Polynomial([-x_points[j], 1]) / (x_points[i] - x_points[j])
        result = result + y_points[i] * Li
    return result.coef


def print_polynomial(coeffs, name):
    terms = []
    for i in range(len(coeffs)-1, -1, -1):
        if abs(coeffs[i]) > 1e-9:
            sign = "+" if coeffs[i] >= 0 and terms else ""
            if i == 0:
                terms.append(f"{sign} {coeffs[i]:.6f}".strip())
            elif i == 1:
                terms.append(f"{sign} {coeffs[i]:.6f}x".strip())
            else:
                terms.append(f"{sign} {coeffs[i]:.6f}x^{i}".strip())
    print(f"{name} = {' '.join(terms)}")


def print_divided_diff_table(x_points, y_points):
    n = len(x_points)
    divided_diff = np.zeros((n, n))
    divided_diff[:, 0] = y_points
    
    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i][j] = (divided_diff[i + 1][j - 1] - divided_diff[i][j - 1]) / \
                                  (x_points[i + j] - x_points[i])
    
    print(f"{'i':>3} {'x_i':>8} {'f[x_i]':>12}", end="")
    for j in range(1, n):
        print(f" {'f[...]':>15}", end="")
    print()
    
    for i in range(n):
        print(f"{i:>3} {x_points[i]:>8.2f} {divided_diff[i][0]:>12.6f}", end="")
        for j in range(1, n - i):
            print(f"{divided_diff[i][j]:>15.8f}", end="")
        print()


def print_newton_form(x_points, divided_diff, name):
    terms = [f"{divided_diff[0][0]:.6f}"]
    for i in range(1, len(x_points)):
        if abs(divided_diff[0][i]) > 1e-10:
            factors = " * ".join([f"(x - {x_points[j]:.2f})" for j in range(i)])
            sign = "+" if divided_diff[0][i] >= 0 else ""
            terms.append(f"{sign} {divided_diff[0][i]:.8f} * {factors}")
    print(f"{name} = {' '.join(terms)}")
                

def plot_all_polynomials(x_data, y_data, all_x, all_y, x_eval):
    x_plot = np.linspace(0, 6, 200)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (k, x_k, y_k) in enumerate(zip(range(2, 7), all_x, all_y)):
        y_plot = lagrange_interpolation(x_k, y_k, x_plot)
        axes[idx].plot(x_k, y_k, 'bo', markersize=8, label='Data Points', zorder=3)
        axes[idx].plot(x_plot, y_plot, 'r-', linewidth=2, label=f'P{k}(x)')
        axes[idx].axvline(x=x_eval, color='green', linestyle='--', linewidth=1, alpha=0.5)
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel(f'P{k}(x)')
        axes[idx].set_title(f'P{k}(x)', fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
    
    axes[5].plot(x_data, y_data, 'ro', markersize=10, label='All Data Points', zorder=10)
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    for k, (x_k, y_k, color) in enumerate(zip(all_x, all_y, colors), start=2):
        y_plot = lagrange_interpolation(x_k, y_k, x_plot)
        axes[5].plot(x_plot, y_plot, linewidth=1.5, label=f'P{k}(x)', color=color, alpha=0.6)
    axes[5].axvline(x=x_eval, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[5].set_xlabel('x')
    axes[5].set_ylabel('f(x)')
    axes[5].set_title('All Polynomials Combined', fontweight='bold')
    axes[5].legend(fontsize=7, loc='upper left')
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lagrange_all_degrees.png', dpi=150, bbox_inches='tight')

def plot_degree_vs_delta(lagrange_degrees, lagrange_deltas, newton_degrees, newton_deltas):
    plt.figure(figsize=(7, 5))
    plt.plot(lagrange_degrees, lagrange_deltas, 'o-', label='Lagrange del_k')
    #plt.plot(newton_degrees, newton_deltas, 's-', label="Newton del_k")
    plt.xlabel('Degree k')
    plt.ylabel('|del_k|')
    plt.title('Degree vs del_k (Lagrange vs Newton)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('degree_vs_delk.png', dpi=150, bbox_inches='tight')

def task1():
    print("Task 1: Lagrange Interpolation\n")
    
    x_data = np.array([0, 1, 2.5, 3, 4.5, 5, 6])
    y_data = np.array([2.00000, 5.43750, 7.35160, 7.56250, 8.44530, 9.18750, 12.00000])
    x_eval = 3.5
    
    print("Given:")
    for i in range(len(x_data)):
        print(f"  x[{i}] = {x_data[i]:4.1f},  y[{i}] = {y_data[i]:.5f}")
    
    print(f"\nEstimate f({x_eval})\n")
    
    x_p2 = x_data[[3, 2, 4]]
    y_p2 = y_data[[3, 2, 4]]
    node_selections = [
        [3, 2, 4],
        [3, 2, 4, 5],
        [3, 2, 4, 5, 1],
        [3, 2, 4, 5, 1, 6],
        [3, 2, 4, 5, 1, 6, 0]
    ]
    
    print("Lagrange Basis Functions for P2(x):\n")
    for i in range(len(x_p2)):
        Li = Polynomial([1])
        for j in range(len(x_p2)):
            if i != j:
                Li = Li * Polynomial([-x_p2[j], 1]) / (x_p2[i] - x_p2[j])
        print_polynomial(Li.coef, f"L{i}(x)")
    
    print("\n\nLagrange Polynomial Equations (Expanded Form):\n")
    
    
    for k, node_idx in enumerate(node_selections, start=2):
        coeffs = get_polynomial_coeffs(x_data[node_idx], y_data[node_idx])
        print_polynomial(coeffs, f"P{k}(x)")
        print()
    
    print("\nEvaluation Results:\n")
    results = []
    all_x = []
    all_y = []
    
    for k, node_idx in enumerate(node_selections, start=2):
        x_k = x_data[node_idx]
        y_k = y_data[node_idx]
        result = lagrange_interpolation(x_k, y_k, x_eval)
        
        print(f"Degree {k}: P{k}(x)")
        print(f"Selected nodes: {x_k}")
        print(f"P{k}({x_eval}) = {result:.8f}")
        
        if k == 2:
            print(f"del{k} = --\n")
        else:
            print(f"del{k} = {abs(result - results[-1]):.8f}\n")
        
        results.append(result)
        all_x.append(x_k)
        all_y.append(y_k)
    
    print("Convergence Table:")
    print(f"{'Degree':>8} {'Pk(3.5)':>15} {'delk':>15}")
    for i, res in enumerate(results, start=2):
        if i == 2:
            print(f"{i:>8} {res:>15.8f} {'-':>15}")
        else:
            print(f"{i:>8} {res:>15.8f} {abs(res - results[i-3]):>15.8e}")
    
    plot_all_polynomials(x_data, y_data, all_x, all_y, x_eval)
    
    degrees = list(range(2, 2 + len(results)))
    delk_degrees = degrees[1:]
    delk_values = [abs(results[i] - results[i - 1]) for i in range(1, len(results))]
    print(f"delk_degrees: {delk_degrees}")
    print(f"delk_values: {delk_values}")
    return delk_degrees, delk_values


def task2():
    print("\n\nTask 2: Newton's Divided Difference Interpolation\n")
    
    x_data = np.array([0, 1, 2.5, 3, 4.5, 5, 6])
    y_data = np.array([2.00000, 5.43750, 7.35160, 7.56250, 8.44530, 9.18750, 12.00000])
    x_eval = 3.5
    
    print(f"Estimate f({x_eval})\n")
    
    distances = np.abs(x_data - x_eval)
    sorted_indices = np.argsort(distances)
    
    print("Nodes ordered by distance from 3.5:")
    for idx in sorted_indices:
        print(f"  x = {x_data[idx]:4.1f},  distance = {distances[idx]:.1f}")
    
    results = []
    all_divided_diffs = []
    all_x_newton = []
    
    for k in range(2, len(x_data) + 1):
        indices = sorted_indices[:k+1]
        x_k = x_data[indices]
        y_k = y_data[indices]
        
        print(f"\nDegree {k}: N{k}(x) using {k+1} nodes {x_k}")
        
        if k == 2:
            print("\nDivided Difference Table:")
            print_divided_diff_table(x_k, y_k)
            print()
        
        result, divided_diff = newton_interpolation(x_k, y_k, x_eval)
        print(f"N{k}({x_eval}) = {result:.8f}")
        
        if results:
            print(f"del{k} = {abs(result - results[-1]):.8f}")
        
        results.append(result)
        all_divided_diffs.append(divided_diff)
        all_x_newton.append(x_k)
    
    print("\n\nDivided Difference Table (all nodes):")
    print_divided_diff_table(x_data[sorted_indices], y_data[sorted_indices])
    
    print("\n\nNewton Polynomial Equations (Newton Form):\n")
    for k, (x_k, divided_diff) in enumerate(zip(all_x_newton, all_divided_diffs), start=2):
        print_newton_form(x_k, divided_diff, f"N{k}(x)")
        print()
    
    print("\nConvergence Table:")
    print(f"{'Degree':>8} {'Nk(3.5)':>15} {'delk':>15}")
    for i, res in enumerate(results, start=2):
        if i == 2:
            print(f"{i:>8} {res:>15.8f} {'-':>15}")
        else:
            print(f"{i:>8} {res:>15.8f} {abs(res - results[i-3]):>15.8e}")

    degrees = list(range(2, 2 + len(results)))
    delk_degrees = degrees[1:]
    delk_values = [abs(results[i] - results[i - 1]) for i in range(1, len(results))]
    return delk_degrees, delk_values


if __name__ == "__main__":
    lagrange_deg, lagrange_deltas = task1()
    newton_deg, newton_deltas = task2()
    plot_degree_vs_delta(lagrange_deg, lagrange_deltas, newton_deg, newton_deltas)
    plt.show()