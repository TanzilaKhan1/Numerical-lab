import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def pt_iter(rows, title):
    print(f"\n{title}")
    if not rows:
        return
    
    n = len(rows[0]["x"])
    
    header = ["Iter"] + [f"x{i+1}" for i in range(n)] + [f"ea{i+1}(%)" for i in range(n)] + ["Residual"]
    fmt = "{:>5s} " + "{:>14s} " * n + "{:>14s} " * n + "{:>14s}"
    print(fmt.format(*header))
    
    for r in rows:
        def f(v):
            if v is None:
                return "-"
            return f"{v:.8f}" if abs(v) >= 0.0001 else f"{v:.8e}"
        
        vals = [r["it"]] + [f(xi) for xi in r["x"]] + [f(ei) for ei in r["ea"]] + [f(r["res"])]
        fmt = "{:5d} " + "{:>14s} " * n + "{:>14s} " * n + "{:>14s}"
        print(fmt.format(*vals))



class InputError(Exception):
    pass

def zero_diagonal(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:

    A = A.copy()
    b = b.copy()
    n = A.shape[0]
    swapped = False
    
    changed = True
    max_iterations = n * n 
    iteration = 0
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        for i in range(n):
            if np.isclose(A[i, i], 0.0):
                swap_row = None
                
                for j in range(i + 1, n):
                    if not np.isclose(A[j, i], 0.0):
                        swap_row = j
                        break
                
                if swap_row is None:
                    for j in range(i):
                        if not np.isclose(A[j, i], 0.0):
                            swap_row = j
                            break
                
                if swap_row is not None:
                    A[[i, swap_row]] = A[[swap_row, i]]
                    b[[i, swap_row]] = b[[swap_row, i]]
                    swapped = True
                    changed = True  
                    print(f"Warning: Zero diagonal at position {i+1}. Swapped row {i+1} with row {swap_row+1}.")
                else:
                    raise InputError(f"Cannot fix zero diagonal at position {i+1}: no suitable row to swap.")
    
    for i in range(n):
        if np.isclose(A[i, i], 0.0):
            raise InputError(f"Failed to fix zero diagonal at position {i+1} after all swaps.")
    
    return A, b, swapped

def calc_res(A, x, b):
    return np.linalg.norm(A @ x - b)


def jacobi(A, b, x0, max_it, tol):
    n = len(b)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)
    
    try:
        A, b, _ = zero_diagonal(A, b)
    except InputError as e:
        print(f"ERROR: {e}")
        return None
    
    rows = []
    done = False
    
    for it in range(max_it):
        x_old = x.copy()
        
        for i in range(n):
            s = b[i] - np.dot(A[i, :i], x_old[:i]) - np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = s / A[i, i]
        
        ea = np.zeros(n)
        for i in range(n):
            if abs(x[i]) > 1e-10:
                ea[i] = abs((x[i] - x_old[i]) / x[i]) * 100
            else:
                ea[i] = abs(x[i] - x_old[i]) * 100
        
        max_ea = np.max(ea)
        res = calc_res(A, x, b)
        
        rows.append({"it": it+1, "x": x.copy(), "ea": ea.copy(), "res": res})
        
        if max_ea < tol or res < tol:
            done = True
            break
    
    pt_iter(rows, "JACOBI METHOD")
    
    if not done:
        print(f"WARNING: Did not converge in {max_it} iterations")
    
    return x, rows


def gauss_seidel(A, b, x0, max_it, tol):
    n = len(b)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)
    
    try:
        A, b, _ = zero_diagonal(A, b)
    except InputError as e:
        print(f"ERROR: {e}")
        return None
    
    rows = []
    done = False
    
    for it in range(max_it):
        x_old = x.copy()
        
        for i in range(n):
            s = b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x[i+1:])
            x[i] = s / A[i, i]
        
        ea = np.zeros(n)
        for i in range(n):
            if abs(x[i]) > 1e-10:
                ea[i] = abs((x[i] - x_old[i]) / x[i]) * 100
            else:
                ea[i] = abs(x[i] - x_old[i]) * 100
        
        max_ea = np.max(ea)
        res = calc_res(A, x, b)
        
        rows.append({"it": it+1, "x": x.copy(), "ea": ea.copy(), "res": res})
        
        if max_ea < tol or res < tol:
            done = True
            break
    
    pt_iter(rows, "GAUSS-SEIDEL METHOD")
    
    if not done:
        print(f"WARNING: Did not converge in {max_it} iterations")
    
    return x, rows


def compare(r_j, r_gs):
    print("\nCOMPARISON")
    
    if not r_j or not r_gs:
        print("Cannot compare - method(s) failed")
        return
    
    it_j = len(r_j)
    it_gs = len(r_gs)
    
    print(f"\nIterations:")
    print(f"  Jacobi:       {it_j}")
    print(f"  Gauss-Seidel: {it_gs}")
    
    diff = it_j - it_gs
    if diff > 0:
        print(f"  Gauss-Seidel is faster by {diff} iteration(s)")
    elif diff < 0:
        print(f"  Jacobi is faster by {-diff} iteration(s)")
    else:
        print(f"  Both converged in same iterations")
    
    print(f"\nFinal Residuals:")
    print(f"  Jacobi:       {r_j[-1]['res']:.8e}")
    print(f"  Gauss-Seidel: {r_gs[-1]['res']:.8e}")
    
    plt.figure(figsize=(10, 6))
    iter_j = [r["it"] for r in r_j]
    res_j = [r["res"] for r in r_j]
    iter_gs = [r["it"] for r in r_gs]
    res_gs = [r["res"] for r in r_gs]
    
    plt.plot(iter_j, res_j, 'o-', label='Jacobi', linewidth=1, markersize=6)
    plt.plot(iter_gs, res_gs, 'o-', label='Gauss-Seidel', linewidth=1, markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('Residual Error')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


def get_input():
    try:
        print("USER INPUT")
        
        n = int(input("\nEnter number of equations: "))
        if n < 2:
            print("ERROR: n must be >= 2")
            return None
        
        print(f"\nEnter coefficient matrix A (row-wise):")
        A = []
        for _ in range(n):
            row = list(map(float, input().split()))
            if len(row) != n:
                print(f"ERROR: Row must have {n} elements or there must be {n} rows")
                return None
            A.append(row)
        
        print(f"\nEnter constants vector b:")
        b = []
        for _ in range(n):
            b.append(float(input()))
        
        try:
            A_np, b_np, _ = zero_diagonal(np.array(A, dtype=float), np.array(b, dtype=float))
            A = A_np.tolist()
            b = b_np.tolist()
        except InputError as e:
            print(f"ERROR: {e}")
            return None

        print(f"\nEnter initial guess vector:")
        x0 = []
        for _ in range(n):
            x0.append(float(input()))
        
        max_it = int(input("\nEnter maximum iterations: "))
        if max_it <= 0:
            print("ERROR: max_it must be > 0")
            return None
        
        tol = float(input("Enter tolerance: "))
        if tol <= 0:
            print("ERROR: tolerance must be > 0")
            return None
        
        return {'n': n, 'A': A, 'b': b, 'x0': x0, 'max_it': max_it, 'tol': tol}
    
    except ValueError:
        print("ERROR: Invalid input")
        return None
    
    
            
def solve(A, b, x0, max_it, tol, title=""):
    if title:
        print(f"\n{title}")
    
    n = len(b)
    
    print("\nSystem:")
    for i in range(n):
        terms = []
        for j in range(n):
            if A[i][j] >= 0 and j > 0:
                terms.append(f"+ {A[i][j]}*x{j+1}")
            else:
                terms.append(f"{A[i][j]}*x{j+1}")
        print(f"  {' '.join(terms)} = {b[i]}")
    
    def fmt_sol(x):
        if x is None:
            return "[]"
        return f"[{', '.join(f'{xi:.6f}' for xi in x)}]"
    
    print(f"\nInitial: x = {x0}")
    print(f"Max iterations: {max_it}")
    print(f"Tolerance: {tol}")
    
    x_j, r_j = jacobi(A, b, x0, max_it, tol)
    if x_j is not None:
        print(f"\nSolution: {fmt_sol(x_j)}")
    
    x_gs, r_gs = gauss_seidel(A, b, x0, max_it, tol)
    if x_gs is not None:
        print(f"\nSolution: {fmt_sol(x_gs)}")
    
    compare(r_j, r_gs)


def test_case_1():
    A = [[4, -1, 0],
        [-1, 4, -1],
        [0, -1, 4]]
    b = [12, -1, 5]
    x0 = [0, 0, 0]
    solve(A, b, x0, 50, 0.0001, "TEST CASE 1")
    print("\nExpected: x = [3.000, 1.000, 1.500]")


def test_case_2():
    A = [[10, -1, 2, 0],
        [-1, 11, -1, 3],
        [2, -1, 10, -1],
        [0, 3, -1, 8]]
    b = [6, 25, -11, 15]
    x0 = [0, 0, 0, 0]
    solve(A, b, x0, 50, 0.0001, "TEST CASE 2")
    print("\nExpected: x = [1.000, 2.000, -1.000, 1.000]")


if __name__ == "__main__":
    while True:
        print("\n\nMENU:")
        print("1. Test Case 1 (3x3)")
        print("2. Test Case 2 (4x4)")
        print("3. Custom Input")
        print("4. Exit")
        
        ch = input("\nChoice: ")
        
        if ch == "1":
            test_case_1()
        elif ch == "2":
            test_case_2()
        elif ch == "3":
            data = get_input()
            if data:
                solve(data['A'], data['b'], data['x0'], 
                      data['max_it'], data['tol'], "CUSTOM INPUT")
        elif ch == "4":
            print("\nExiting...")
            break
        else:
            print("Invalid choice")