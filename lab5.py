import numpy as np
import matplotlib.pyplot as plt


def solve_euler(f, x0, y0, xf, h):
    x_vals, y_vals = [x0], [y0]
    details = []
    x, y, i = x0, y0, 0

    while x < xf - 1e-10:
        h_step = min(h, xf - x)
        f_xy = f(x, y)
        y_new = y + f_xy * h_step

        details.append([i, x, y, f_xy, y_new])

        x += h_step
        y = y_new
        x_vals.append(x)
        y_vals.append(y)
        i += 1

    return np.array(x_vals), np.array(y_vals), details


def marker_indices(length, max_markers=150):
    if length <= max_markers or max_markers <= 0:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, max_markers, dtype=int)


def format_step_size(h, decimals=6):
    formatted = f"{h:.{decimals}f}".rstrip('0').rstrip('.')
    return formatted if formatted else "0"


def table_display_ranges(length, first_rows=10, last_rows=5):
    if length <= first_rows + last_rows:
        return [range(length)], False
    start_last = max(first_rows, length - last_rows)
    return [range(first_rows), range(start_last, length)], True


def solve_heun(f, x0, y0, xf, h):
    x_vals, y_vals = [x0], [y0]
    details = []
    x, y, i = x0, y0, 0

    while x < xf - 1e-10:
        h_step = min(h, xf - x)
        f_xy = f(x, y)
        y_pred = y + f_xy * h_step
        f_pred = f(x + h_step, y_pred)
        y_new = y + (f_xy + f_pred) / 2 * h_step

        details.append([i, x, y, f_xy, y_pred, f_pred, y_new])

        x += h_step
        y = y_new
        x_vals.append(x)
        y_vals.append(y)
        i += 1

    return np.array(x_vals), np.array(y_vals), details


def print_euler_table(details):
    print("\nEuler Method Calculation Table")
    print("{:>4s} {:>12s} {:>12s} {:>14s} {:>14s}".format(
        "i", "x_i", "y_i", "f(x_i,y_i)", "y_i+1"))

    ranges, truncated = table_display_ranges(len(details))
    for idx_block, idx_range in enumerate(ranges):
        if truncated and idx_block == 1:
            print("... (rows hidden) ...")
        for idx in idx_range:
            d = details[idx]
            print("{:4d} {:12.6f} {:12.6f} {:14.6f} {:14.6f}".format(
                int(d[0]), d[1], d[2], d[3], d[4]))


def print_heun_table(details):
    print("\nHeun's Method Calculation Table")
    print("{:>4s} {:>12s} {:>12s} {:>14s} {:>14s} {:>18s} {:>14s}".format(
        "i", "x_i", "y_i", "f(x_i,y_i)", "y'_i+1", "f(x_i+1,y'_i+1)", "y_i+1"))

    ranges, truncated = table_display_ranges(len(details))
    for idx_block, idx_range in enumerate(ranges):
        if truncated and idx_block == 1:
            print("... (rows hidden) ...")
        for idx in idx_range:
            d = details[idx]
            print("{:4d} {:12.6f} {:12.6f} {:14.6f} {:14.6f} {:18.6f} {:14.6f}".format(
                int(d[0]), d[1], d[2], d[3], d[4], d[5], d[6]))


def print_results_table(x, y_euler, y_heun, y_true):
    print("\nResults Comparison Table")
    print("{:>10s} {:>16s} {:>16s} {:>16s} {:>14s} {:>14s}".format(
        "t", "y_Euler", "y_Heun", "y_exact", "Error_Euler", "Error_Heun"))

    ranges, truncated = table_display_ranges(len(x))
    for idx_block, idx_range in enumerate(ranges):
        if truncated and idx_block == 1:
            print("... (rows hidden) ...")
        for i in idx_range:
            err_euler = abs(y_true[i] - y_euler[i])
            err_heun = abs(y_true[i] - y_heun[i])
            print("{:10.6f} {:16.6f} {:16.6f} {:16.6f} {:14.6f} {:14.6f}".format(
                x[i], y_euler[i], y_heun[i], y_true[i], err_euler, err_heun))


def print_exact_table(x, y_true, h):
    h_str = format_step_size(h)
    print("\nExact Solution Table (h = {})".format(h_str))
    print("{:>10s} {:>16s}".format("t", "y(t)"))
    ranges, truncated = table_display_ranges(len(x))
    for idx_block, idx_range in enumerate(ranges):
        if truncated and idx_block == 1:
            print("... (rows hidden) ...")
        for i in idx_range:
            print("{:10.6f} {:16.6f}".format(x[i], y_true[i]))


def plot_all_methods(x_vals_dict, y_vals_dict, y_exact_func, step_sizes, title):
    plt.figure(figsize=(14, 8))

    x_min = min(x_vals_dict[h][0] for h in step_sizes)
    x_max = max(x_vals_dict[h][-1] for h in step_sizes)
    x_fine = np.linspace(x_min, x_max, 500)
    y_exact = y_exact_func(x_fine)
    plt.plot(x_fine, y_exact, 'k-', linewidth=2.5, label='Exact Solution')

    colors = ['red', 'orange', 'green', 'blue', 'purple']
    markers_euler = ['o', 's', '^', 'd', 'v']
    markers_heun = ['x', '+', '*', 'p', 'h']

    for i, h in enumerate(step_sizes):
        x = x_vals_dict[h]
        y_euler = y_vals_dict[h]['euler']
        y_heun = y_vals_dict[h]['heun']

        idx_euler = marker_indices(len(x))
        idx_heun = marker_indices(len(x))

        line_e, = plt.plot(
            x,
            y_euler,
            color=colors[i],
            linestyle='--',
            linewidth=1,
            label=f'Euler h={h}'
        )
        plt.plot(
            x[idx_euler],
            y_euler[idx_euler],
            linestyle='None',
            marker=markers_euler[i],
            markersize=4,
            color=line_e.get_color()
        )

        line_h, = plt.plot(
            x,
            y_heun,
            color=colors[i],
            linestyle='-.',
            linewidth=1,
            label=f'Heun h={h}'
        )
        plt.plot(
            x[idx_heun],
            y_heun[idx_heun],
            linestyle='None',
            marker=markers_heun[i],
            markersize=4,
            color=line_h.get_color()
        )

    plt.xlabel('t', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_task(name, f, analytical, x0, y0, xf, step_sizes, ode_str, sol_str):
    print(name)
    print(f"ODE: {ode_str}")
    print(f"Solution: {sol_str}")
    print(f"Initial: y({x0}) = {y0}")
    print(f"Range: t = {x0} to {xf}")

    x_vals_dict = {}
    y_vals_dict = {}

    for idx, h in enumerate(step_sizes):
        print("\n" + "-" * 30)
        print(f"Step size h = {h}")

        x, y_euler, euler_details = solve_euler(f, x0, y0, xf, h)

        if idx == 0:
            print_euler_table(euler_details)

        _, y_heun, heun_details = solve_heun(f, x0, y0, xf, h)

        if idx == 0:
            print_heun_table(heun_details)

        y_true = analytical(x)

        print_exact_table(x, y_true, h)
        print_results_table(x, y_euler, y_heun, y_true)

        x_vals_dict[h] = x
        y_vals_dict[h] = {'euler': y_euler, 'heun': y_heun}

    plot_all_methods(
        x_vals_dict,
        y_vals_dict,
        analytical,
        step_sizes,
        f"{name} - Comparison of All Methods and Step Sizes"
    )


def task():
    print("Lab Task 1")

    def task1_deriv(t, y):
        return (1 + 4 * t) - y

    def task1_sol(t):
        return 4 * t - 3 + 4 * np.exp(-t)

    run_task(
        "Task 1",
        task1_deriv,
        task1_sol,
        0.0,
        1.0,
        5.0,
        [0.50, 0.25, 0.1, 0.01, 0.001],
        "dy/dt = (1 + 4t) - y",
        "y(t) = 4t - 3 + 4e^(-t)"
    )

    print("Lab Task 2")

    def task2_deriv(t, y):
        return y * (np.sin(t) ** 3)

    def task2_sol(t):
        return np.exp(-np.cos(t) + (np.cos(t) ** 3) / 3 + 2 / 3)

    run_task(
        "Task 2",
        task2_deriv,
        task2_sol,
        0.0,
        1.0,
        5.0,
        [0.1, 0.005, 0.001, 0.0005, 0.0001],
        "dy/dt = y*sin³(t)",
        "y(t) = exp(-cos(t) + cos³(t)/3 + 2/3)"
    )


if __name__ == "__main__":
    task()
