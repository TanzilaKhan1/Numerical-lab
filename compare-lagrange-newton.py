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


def plot_temperature_interpolation(x_data, y_data, x_eval, lagrange_results):
    x_plot = np.linspace(0, 100, 200)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    degrees = [2, 3, 4, len(x_data)-1]
    titles = ['2nd Degree (Quadratic)', '3rd Degree (Cubic)', 
              '4th Degree', f'{len(x_data)-1}th Degree (Full)']
    
    for idx, (deg, title) in enumerate(zip(degrees, titles)):
        ax = axes[idx // 2, idx % 2]
        
        distances = np.abs(x_data - x_eval)
        sorted_indices = np.argsort(distances)[:deg+1]
        sorted_indices = np.sort(sorted_indices)
        x_k = x_data[sorted_indices]
        y_k = y_data[sorted_indices]
        
        y_lagrange = lagrange_interpolation(x_k, y_k, x_plot)
        
        ax.plot(x_data, y_data, 'ro', markersize=8, label='Sensor Data', zorder=3)
        ax.plot(x_k, y_k, 'go', markersize=6, label='Selected Nodes', zorder=4)
        ax.plot(x_plot, y_lagrange, 'b-', linewidth=1.5, label='Lagrange', alpha=0.7)
        
        result = lagrange_results[idx]
        ax.plot(x_eval, result, 'ks', markersize=10, label=f'T({x_eval})={result:.2f}°C', zorder=5)
        ax.axvline(x=x_eval, color='green', linestyle='--', linewidth=1.2, alpha=0.4)
        
        ax.set_xlabel('Distance (km)', fontsize=10)
        ax.set_ylabel('Temperature (°C)', fontsize=10)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlim(-5, 105)
        ax.set_ylim(20, 45)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('temperature_interpolation.png', dpi=150, bbox_inches='tight')


def plot_comparison(x_data, y_data, x_eval, all_degrees):
    x_plot = np.linspace(0, 100, 200)
    
    plt.figure(figsize=(12, 7))
    
    plt.plot(x_data, y_data, 'ro', markersize=10, label='Sensor Data', zorder=5)
    
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    labels = ['2nd Degree', '3rd Degree', '4th Degree', '5th Degree', '6th Degree', '7th Degree', '8th Degree']
    
    for idx, (deg, color, label) in enumerate(zip(all_degrees, colors, labels)):
        distances = np.abs(x_data - x_eval)
        sorted_indices = np.argsort(distances)[:deg+1]
        sorted_indices = np.sort(sorted_indices)
        x_k = x_data[sorted_indices]
        y_k = y_data[sorted_indices]
        
        y_lagrange = lagrange_interpolation(x_k, y_k, x_plot)
        plt.plot(x_plot, y_lagrange, color=color, linewidth=1.5, label=f'{label}', alpha=0.7)
    
    plt.axvline(x=x_eval, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label=f'x = {x_eval} km')
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.title('Temperature Interpolation: Lagrange Method (All Degrees)', fontweight='bold', fontsize=13)
    plt.xlim(-5, 105)
    plt.ylim(20, 45)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig('temperature_comparison_lagrange.png', dpi=150, bbox_inches='tight')


def plot_convergence_curve(lagrange_results, newton_results, degrees):
    plt.figure(figsize=(10, 6))
    
    plt.plot(degrees, lagrange_results, 'o-', linewidth=1.0, markersize=8, label='Lagrange', color='blue', alpha=0.8)
    plt.plot(degrees, newton_results, 's--', linewidth=1.0, markersize=7, label='Newton', color='red', alpha=0.8)
    
    plt.xlabel('Polynomial Degree', fontsize=12)
    plt.ylabel('Interpolated Temperature at x = 45 km (°C)', fontsize=12)
    plt.title('Convergence of Interpolation Methods', fontweight='bold', fontsize=13)
    plt.xticks(degrees)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('convergence_curve.png', dpi=150, bbox_inches='tight')


def plot_delta_convergence(lagrange_results, newton_results, degrees):
    lagrange_deltas = [abs(lagrange_results[i] - lagrange_results[i-1]) for i in range(1, len(lagrange_results))]
    newton_deltas = [abs(newton_results[i] - newton_results[i-1]) for i in range(1, len(newton_results))]
    
    delta_degrees = np.array(degrees[1:])
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(delta_degrees, lagrange_deltas, 'o-', linewidth=1.0, markersize=8, 
             label='Lagrange Δk', color='blue', alpha=0.7, markeredgecolor='darkblue', markeredgewidth=1.2)
    plt.plot(delta_degrees, newton_deltas, 's--', linewidth=1.0, markersize=7, 
             label='Newton Δk', color='red', alpha=0.7, markeredgecolor='darkred', markeredgewidth=1.2)
    plt.xlabel('Polynomial Degree k', fontsize=12)
    plt.ylabel('|Δk| = |Pk(45) - Pk-1(45)| (°C)', fontsize=12)
    plt.title('Convergence Rate: Change Between Successive Degrees', fontweight='bold', fontsize=13)
    plt.xticks(delta_degrees)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.savefig('delta_convergence.png', dpi=150, bbox_inches='tight')


def lab_report():
    print("Lab Report: Temperature Interpolation along Highway\n")
    
    x_data = np.array([0, 10, 20, 35, 50, 65, 80, 90, 100], dtype=float)
    y_data = np.array([25.0, 26.7, 29.4, 33.2, 35.5, 36.1, 37.8, 38.9, 40.0], dtype=float)
    x_eval = 45.0
    
    print("Given Sensor Data:")
    print(f"{'i':>3} {'Distance (km)':>15} {'Temperature (°C)':>18}")
    for i in range(len(x_data)):
        print(f"{i:>3} {x_data[i]:>15.1f} {y_data[i]:>18.1f}")
    
    print(f"\nEstimate temperature at x = {x_eval} km\n")
    
    print("Task 1: Lagrange Interpolation\n")
    
    lagrange_results = []
    all_degrees = [2, 3, 4, 5, 6, 7, 8]
    
    for deg in all_degrees:
        distances = np.abs(x_data - x_eval)
        sorted_indices = np.argsort(distances)[:deg+1]
        sorted_indices = np.sort(sorted_indices)
        x_k = x_data[sorted_indices]
        y_k = y_data[sorted_indices]
        
        result = lagrange_interpolation(x_k, y_k, x_eval)
        
        print(f"Degree {deg}: P{deg}(x)")
        print(f"Selected nodes: {x_k}")
        print(f"P{deg}({x_eval}) = {result:.8f} °C")
        
        if lagrange_results:
            print(f"Change from previous: {abs(result - lagrange_results[-1]):.8f} °C\n")
        else:
            print(f"Change from previous: --\n")
        
        lagrange_results.append(result)
    
    print("Lagrange Convergence Table:")
    print(f"{'Degree':>8} {'Temperature (°C)':>20} {'Change':>15}")
    for i, (deg, res) in enumerate(zip(all_degrees, lagrange_results)):
        if i == 0:
            print(f"{deg:>8} {res:>20.8f} {'-':>15}")
        else:
            print(f"{deg:>8} {res:>20.8f} {abs(res - lagrange_results[i-1]):>15.8e}")
    
    print("\n\nTask 2: Newton's Divided Difference Interpolation\n")
    
    newton_results = []
    
    for deg in all_degrees:
        distances = np.abs(x_data - x_eval)
        sorted_indices = np.argsort(distances)[:deg+1]
        sorted_indices = np.sort(sorted_indices)
        x_k = x_data[sorted_indices]
        y_k = y_data[sorted_indices]
        
        result, divided_diff = newton_interpolation(x_k, y_k, x_eval)
        
        print(f"Degree {deg}: N{deg}(x)")
        print(f"Selected nodes: {x_k}")
        print(f"N{deg}({x_eval}) = {result:.8f} °C")
        
        if newton_results:
            print(f"Change from previous: {abs(result - newton_results[-1]):.8f} °C\n")
        else:
            print(f"Change from previous: --\n")
        
        newton_results.append(result)
    
    print("Newton Convergence Table:")
    print(f"{'Degree':>8} {'Temperature (°C)':>20} {'Change':>15}")
    for i, (deg, res) in enumerate(zip(all_degrees, newton_results)):
        if i == 0:
            print(f"{deg:>8} {res:>20.8f} {'-':>15}")
        else:
            print(f"{deg:>8} {res:>20.8f} {abs(res - newton_results[i-1]):>15.8e}")
    
    print("\n\nDivided Difference Table (Full Degree):")
    distances = np.abs(x_data - x_eval)
    sorted_indices = np.argsort(distances)
    sorted_indices_sorted = np.sort(sorted_indices)
    print_divided_diff_table(x_data[sorted_indices_sorted], y_data[sorted_indices_sorted])
    
    print("\n\nComparison of Methods:")
    print(f"{'Degree':>8} {'Lagrange (°C)':>20} {'Newton (°C)':>20} {'Difference':>15}")
    for deg, lag, new in zip(all_degrees, lagrange_results, newton_results):
        print(f"{deg:>8} {lag:>20.8f} {new:>20.8f} {abs(lag - new):>15.8e}")
    
    # For plotting, use degrees 2, 3, 4, 8 from the results
    plot_lagrange_results = [lagrange_results[i] for i in [0, 1, 2, 6]]
    plot_temperature_interpolation(x_data, y_data, x_eval, plot_lagrange_results)
    plot_comparison(x_data, y_data, x_eval, all_degrees)
    plot_convergence_curve(lagrange_results, newton_results, all_degrees)
    plot_delta_convergence(lagrange_results, newton_results, all_degrees)


if __name__ == "__main__":
    lab_report()
    plt.show()