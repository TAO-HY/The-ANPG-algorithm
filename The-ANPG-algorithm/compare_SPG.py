# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 13:18:50 2025

@author: 22472
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd
import time

def plot_sci_style():
    """Set SCI journal style for plots - professional with consistent font sizes"""
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Computer Modern']
    plt.rcParams['font.size'] = 16  # Unified base font size
    plt.rcParams['axes.labelsize'] = 18  # Axis label font size
    plt.rcParams['axes.titlesize'] = 20  # Title font size
    plt.rcParams['legend.fontsize'] = 16  # Legend font size
    plt.rcParams['xtick.labelsize'] = 14  # x-axis tick label font size
    plt.rcParams['ytick.labelsize'] = 14  # y-axis tick label font size
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['lines.linewidth'] = 1.8
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['mathtext.it'] = 'serif:italic'
    plt.rcParams['mathtext.bf'] = 'serif:bold'
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.format'] = 'pdf'

def prox_l1_scaled(v, b, lambda_val, m):
    """
    Compute proximal operator for f(y) = (1/m) * ||y - b||_1
    """
    v = np.asarray(v)
    b = np.asarray(b)
    t = lambda_val / m
    
    a = v - b
    y_star = b + np.sign(a) * np.maximum(np.abs(a) - t, 0)
    
    return y_star

def moreau_envelope_l1_scaled(v, b, lambda_val, m):
    """
    Compute Moreau envelope value for f(y) = (1/m) * ||y - b||_1
    """
    v = np.asarray(v)
    b = np.asarray(b)
    t = lambda_val / m
    
    a = v - b
    abs_a = np.abs(a)
    
    contributions = np.where(
        abs_a <= t,
        a**2 / (2 * lambda_val),
        abs_a / m - t / (2 * m)
    )
    
    M_value = np.sum(contributions)
    return M_value

def moreau_from_prox(v, b, lambda_val, m, prox_result=None):
    """
    Compute Moreau envelope value via proximal operator result
    """
    v = np.asarray(v)
    b = np.asarray(b)
    
    if prox_result is None:
        t = lambda_val / m
        a = v - b
        y_star = b + np.sign(a) * np.maximum(np.abs(a) - t, 0)
    else:
        y_star = np.asarray(prox_result)
    
    f_value = np.sum(np.abs(y_star - b)) / m
    quadratic_penalty = np.sum((y_star - v)**2) / (2 * lambda_val)
    M_value = f_value + quadratic_penalty
    
    return M_value, y_star

def GANPG(A, b, lambda_, nu0, x0, mu_0):
    """GANPG algorithm implementation with oracle counting"""
    epsilon = 1e-6
    x = x0.copy()
    m, n = A.shape
    x_old = x.copy()
    y = x.copy()
    
    max_iter = 500
    norm_A = la.norm(A, ord=2)**2
    u = 0.99 * lambda_ / (np.sqrt(m) / m * norm_A)
    mu = mu_0 * norm_A
    gamma = mu / norm_A

    # Initial objective function value
    hzt = np.sum(np.maximum(x - nu0, 0) + np.maximum(0, -x - nu0))
    FO_rho = 1/m * np.linalg.norm(A @ x - b, 1) + lambda_ / nu0 * (np.linalg.norm(x, 1) - hzt)
    
    F_values = [FO_rho]
    oracle_calls = [0]  # Record cumulative oracle call count
    
    beta = 0
    
    # Initial gradient calculation (1 oracle call)
    Ay = A @ y
    prox_Ay = prox_l1_scaled(Ay, b, mu, m)
    grad = 1/mu * A.T @ (Ay - prox_Ay)
    M_y = moreau_from_prox(Ay, b, mu, m, prox_Ay)[0]
    alpha0 = 1/8
    Alpha=[]
    for iter in range(max_iter):
        nu = max(nu0 * 0.9**(iter), u)
        
        # Calculate d
        d = np.zeros(n)
        d[(x >= nu)] = 1
        d[(x <= -nu)] = -1
        
        # Line search (may call oracle multiple times per iteration)
        line_search_oracles = 0
        for i in range(20):
            alpha = alpha0 * 2**i
            w = y - gamma/alpha * grad
            s = gamma/alpha * lambda_ / nu
            w = w + s * d
            x_hat = np.sign(w) * np.maximum(np.abs(w) - s, 0)
            
            # Compute Moreau envelope (1 oracle call)
            M_hat = moreau_envelope_l1_scaled(A @ x_hat, b, mu, m)
            
            line_search_oracles += 1
            
            if M_hat <= M_y + grad.T @ (x_hat - y) + alpha/(2*gamma) * np.linalg.norm(x_hat - y)**2:
                break
        alpha0 = alpha
        Alpha.append(alpha)
        x_old = x.copy()
        x = x_hat.copy()
        
        # Update parameters
        beta = (iter) / (iter + 3.1)
        
        # Compute next iterate y
        y = x + beta * (x - x_old) * np.abs(d)
        
        # Compute gradient (1 oracle call)
        Ay = A @ y
        mu = mu_0 * norm_A / ((iter + 3.1) * np.log(iter + 3.1) ** (11/20))
        gamma = mu / norm_A
        prox_Ay = prox_l1_scaled(Ay, b, mu, m)
        grad = 1/mu * A.T @ (Ay - prox_Ay)
        
        M_y = moreau_from_prox(Ay, b, mu, m, prox_Ay)[0]
        oracle_calls.append(oracle_calls[-1] + line_search_oracles*2 + 1)  # Update cumulative oracle call count
        
        # Compute new objective function value is only used for stopping criterion
        # All algorithm iterations do not require computing this, and we do not count it towards Oracle calls
        hzt = np.sum(np.maximum(x - nu, 0) + np.maximum(0, -x - nu))
        F_rho = 1/m * np.linalg.norm(A @ x - b, 1) + lambda_ / nu * (np.linalg.norm(x, 1) - hzt)
        F_values.append(F_rho)
        
        # Check convergence condition
        if np.abs(FO_rho - F_rho) < epsilon:
            break
            
        FO_rho = F_rho
    
    final_F_value = F_rho
    return x, iter + 1, final_F_value, F_values, oracle_calls

def f_tilde(A, b, x, mu):
    """Smooth approximation function and its gradient"""
    m = A.shape[0]
    er = A @ x - b
    abs_er = np.abs(er)
    ind_smooth = np.where(abs_er <= mu)[0]
    ind_right = np.where(er > mu)[0]
    ind_left = np.where(er < -mu)[0]
    
    # Compute smooth function phi(t,mu)
    phi = np.zeros(m)
    phi[ind_smooth] = (er[ind_smooth]**2) / (2 * mu) + mu/2
    phi[ind_right] = er[ind_right]
    phi[ind_left] = -er[ind_left]
    
    # Compute derivative
    part_phi = np.zeros(m)
    part_phi[ind_smooth] = er[ind_smooth] / mu
    part_phi[ind_right] = 1
    part_phi[ind_left] = -1
    
    # Compute gradient
    grad = 1/m * A.T @ part_phi
    # Compute function value
    f_val = 1/m * np.sum(phi)
    return f_val, grad

def SPG(A, b, lambda_, nu0, x0, mu_0):
    """SPG algorithm implementation with oracle counting"""
    epsilon = 1e-6
    x = x0.copy()
    m, n = A.shape
    norm_A = la.norm(A, ord=2)**2
    gamma1 = 0.25 / m * norm_A
    max_iter = 500
    mu = mu_0
    mu_1 = mu_0
    u = 0.99 * lambda_ / (np.sqrt(m) / m * norm_A)
    
    # Initial objective function value
    hzt = np.sum(np.maximum(x - nu0, 0) + np.maximum(0, -x - nu0))
    FO_rho = 1/m * np.linalg.norm(A @ x - b, 1) + lambda_ / nu0 * (np.linalg.norm(x, 1) - hzt)
    
    F_values = [FO_rho]
    oracle_calls = [0]  # Record cumulative oracle call count
    
    kappa = 0.5
    alpha = 1
    
    for iter in range(max_iter):
        nu = max(nu0 * 0.9**(iter), u)
        d = np.zeros(n)
        d[(x >= nu)] = 1
        d[(x <= -nu)] = -1
        
        # Compute function value and gradient at current point (1 oracle call)
        f_val, grad = f_tilde(A, b, x, mu)
        
        line_search_count = 0
        for i in range(20):
            line_search_count += 1
            gamma = gamma1 * 2**i
            w = x - mu / gamma * grad
            s = mu * lambda_ / (nu * gamma)
            w = w + s * d
            x_hat = np.sign(w) * np.maximum(np.abs(w) - s, 0)
            
            # Compute function value at new point (1 oracle call)
            f_val_hat, _ = f_tilde(A, b, x_hat, mu)
            
            if f_val_hat <= f_val + grad.T @ (x_hat - x) + gamma/(2*mu) * np.linalg.norm(x_hat - x)**2:
                break
        
        # Update cumulative oracle call count (each line search attempt calls oracle once)
        oracle_calls.append(oracle_calls[-1] + line_search_count*2 + 1)
        
        x_old = x.copy()
        x = x_hat.copy()
        
        hzt = np.sum(np.maximum(x - nu, 0) + np.maximum(0, -x - nu))
        F_rho = 1/m * np.linalg.norm(A @ x - b, 1) + lambda_ / nu * (np.linalg.norm(x, 1) - hzt)
        
        phi_new = np.sum(np.minimum(np.abs(x) / nu, 1))
        F_new = f_val_hat + lambda_ * phi_new + kappa * mu
        phi_old = np.sum(np.minimum(np.abs(x_old) / nu, 1))
        F_old = f_val + lambda_ * phi_old + kappa * mu_1
        F_values.append(F_rho)
        
        if np.abs(FO_rho - F_rho) < epsilon:
            break
    
        mu_1 = mu
        if F_new - F_old > -alpha * mu_1**2:
            mu = mu_0 * (iter + 1) ** (-11/20)
        FO_rho = F_rho
        
    final_F_value = F_rho
    return x, iter + 1, final_F_value, F_values, oracle_calls

def ANPG(A, b, lambda_, nu0, x0, mu_0):
    """ANPG algorithm implementation with oracle counting"""
    epsilon = 1e-6
    x = x0.copy()
    m, n = A.shape
    x_old = x.copy()
    y = x.copy()
    
    max_iter = 500
    t1 = 1
    norm_A = la.norm(A, ord=2)**2
    u = 0.99 * lambda_ / (np.sqrt(m) / m * norm_A)
    mu = mu_0 * norm_A
    gamma = mu / norm_A

    # Initial objective function value
    hzt = np.sum(np.maximum(x - nu0, 0) + np.maximum(0, -x - nu0))
    FO_rho = 1/m * np.linalg.norm(A @ x - b, 1) + lambda_ / nu0 * (np.linalg.norm(x, 1) - hzt)
    
    F_values = [FO_rho]
    oracle_calls = [0]  # Record cumulative oracle call count
    
    beta = 0
    
    # Initial gradient calculation (1 oracle call)
    Ay = A @ y
    prox_Ay = prox_l1_scaled(Ay, b, mu, m)
    grad = 1/mu * A.T @ (Ay - prox_Ay)
    M_y = moreau_from_prox(Ay, b, mu, m, prox_Ay)[0]
    alpha0 = 1/8
    Alpha=[]
    for iter in range(max_iter):
        nu = max(nu0 * 0.9**(iter), u)
        
        # Calculate d
        d = np.zeros(n)
        d[(x >= nu)] = 1
        d[(x <= -nu)] = -1
        
        # Line search (may call oracle multiple times per iteration)
        line_search_oracles = 0
        for i in range(20):
            alpha = alpha0 * 2**i
            w = y - gamma/alpha * grad
            s = gamma/alpha * lambda_ / nu
            w = w + s * d
            x_hat = np.sign(w) * np.maximum(np.abs(w) - s, 0)
        
            # Compute Moreau envelope (1 oracle call)
            M_hat = moreau_envelope_l1_scaled(A @ x_hat, b, mu, m)
            line_search_oracles += 1
            if M_hat <= M_y + grad.T @ (x_hat - y) + alpha/(2*gamma) * np.linalg.norm(x_hat - y)**2:
                break
        alpha0 = alpha
        Alpha.append(alpha0)
        x_old = x.copy()
        x = x_hat.copy()
        
        # Update parameters
        t2 = np.sqrt(t1**2 + 2 * t1)
        beta = (t1 - 1) / t2
        
        # Compute next iterate y
        y = x + beta * (x - x_old) * np.abs(d)
        # Compute gradient (1 oracle call)
        Ay = A @ y
        mu = mu * t1**2 / (t2**2 - t2)
        gamma = mu / norm_A
        prox_Ay = prox_l1_scaled(Ay, b, mu, m)
        grad = 1/mu * A.T @ (Ay - prox_Ay)
        
        oracle_calls.append(oracle_calls[-1] + line_search_oracles*2 + 1)
        
        M_y = moreau_from_prox(Ay, b, mu, m, prox_Ay)[0]
        
        # Compute new objective function value
        hzt = np.sum(np.maximum(x - nu, 0) + np.maximum(0, -x - nu))
        F_rho = 1/m * np.linalg.norm(A @ x - b, 1) + lambda_ / nu * (np.linalg.norm(x, 1) - hzt)
        F_values.append(F_rho)
        
        # Check convergence condition
        if np.abs(FO_rho - F_rho) < epsilon:
            break
            
        FO_rho = F_rho
        t1 = t2
    
    final_F_value = F_rho
    return x, iter + 1, final_F_value, F_values, oracle_calls

def calculate_metrics(x_true, x_est, threshold=1e-3):
    """Calculate MSE and F1 score for support recovery"""
    # MSE
    mse = np.mean((x_true - x_est) ** 2)
    
    # True support (non-zero indices)
    true_support = np.where(np.abs(x_true) > threshold)[0]
    
    # Estimated support
    est_support = np.where(np.abs(x_est) > threshold)[0]
    
    # True positive, false positive, false negative
    tp = len(np.intersect1d(true_support, est_support))
    fp = len(np.setdiff1d(est_support, true_support))
    fn = len(np.setdiff1d(true_support, est_support))
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return mse, f1, precision, recall

def run_experiment_1():
    """Experiment 1: SPG fixed parameters vs ANPG/GANPG varying nu"""
    np.random.seed(42)
    
    # Parameters
    n = 1000
    m = 3000
    s = 50
    
    # Generate sparse signal x_true
    x_true = np.zeros(n)
    T = np.random.permutation(n)[:s]
    x_true[T] = np.sign(np.random.randn(s)) * (1 + 0.5 * np.random.rand(s))
    
    # Generate measurement matrix
    A_mat = np.random.randn(m, n)
    A_mat = A_mat / la.norm(A_mat, 2)
    
    # Generate measurements with outliers
    b_clean = A_mat @ x_true + 0.01 * np.random.randn(m)
    
    # Add 10% outliers
    contamination_rate = 0.1
    n_outliers = int(m * contamination_rate)
    outlier_indices = np.random.choice(m, n_outliers, replace=False)
    
    b_contaminated = b_clean.copy()
    outlier_magnitude = np.max(np.abs(b_clean)) * 5
    b_contaminated[outlier_indices] = outlier_magnitude * np.sign(np.random.randn(n_outliers))
    
    # Fixed parameters for SPG
    spg_mu_fixed = 0.1
    spg_nu_fixed = 0.1
    lambda_val = 0.00004
    
    # Fixed mu for ANPG and GANPG, varying nu
    anpg_ganpg_mu_fixed = 200
    nu_values = [0.05, 0.08, 0.1, 0.12, 0.15]
    
    # Store results
    results = {}
    metrics_table = []
    
    print("Running Experiment 1: SPG fixed vs ANPG/GANPG varying nu")
    print("SPG: μ = 0.1, ν = 0.1 (fixed)")
    print("ANPG/GANPG: μ = 200 (fixed), ν varying")
    
    # Run SPG with fixed parameters
    print(f"\nRunning SPG with μ={spg_mu_fixed}, ν={spg_nu_fixed}...")
    x_spg, iterations_spg, final_F_spg, obj_values_spg, oracle_calls_spg = SPG(
        A_mat, b_contaminated, lambda_val, spg_nu_fixed, np.zeros(n), spg_mu_fixed)
    
    mse_spg, f1_spg, precision_spg, recall_spg = calculate_metrics(x_true, x_spg)
    metrics_table.append(['SPG', spg_mu_fixed, spg_nu_fixed, mse_spg, f1_spg, precision_spg, recall_spg, iterations_spg])
    
    results['SPG_fixed'] = {
        'obj_values': obj_values_spg,
        'oracle_calls': oracle_calls_spg,
        'iterations': iterations_spg,
        'mse': mse_spg,
        'f1': f1_spg,
        'mu': spg_mu_fixed,
        'nu': spg_nu_fixed
    }
    
    print(f"SPG F1-Score: {f1_spg:.6f}")
    
    # Run ANPG and GANPG with varying nu values
    for nu_val in nu_values:
        print(f"\nTesting ν = {nu_val}")
        
        # ANPG
        print("Running ANPG...")
        x_anpg, iterations_anpg, final_F_anpg, obj_values_anpg, oracle_calls_anpg = ANPG(
            A_mat, b_contaminated, lambda_val, nu_val, np.zeros(n), anpg_ganpg_mu_fixed)
        
        mse_anpg, f1_anpg, precision_anpg, recall_anpg = calculate_metrics(x_true, x_anpg)
        metrics_table.append(['ANPG', anpg_ganpg_mu_fixed, nu_val, mse_anpg, f1_anpg, precision_anpg, recall_anpg, iterations_anpg])
        
        results[f'ANPG_ν{nu_val}'] = {
            'obj_values': obj_values_anpg,
            'oracle_calls': oracle_calls_anpg,
            'iterations': iterations_anpg,
            'mse': mse_anpg,
            'f1': f1_anpg,
            'mu': anpg_ganpg_mu_fixed,
            'nu': nu_val
        }
        
        print(f"ANPG (ν={nu_val}) F1-Score: {f1_anpg:.6f}")
        
        # GANPG
        print("Running GANPG...")
        x_ganpg, iterations_ganpg, final_F_ganpg, obj_values_ganpg, oracle_calls_ganpg = GANPG(
            A_mat, b_contaminated, lambda_val, nu_val, np.zeros(n), anpg_ganpg_mu_fixed)
        
        mse_ganpg, f1_ganpg, precision_ganpg, recall_ganpg = calculate_metrics(x_true, x_ganpg)
        metrics_table.append(['GANPG', anpg_ganpg_mu_fixed, nu_val, mse_ganpg, f1_ganpg, precision_ganpg, recall_ganpg, iterations_ganpg])
        
        results[f'GANPG_ν{nu_val}'] = {
            'obj_values': obj_values_ganpg,
            'oracle_calls': oracle_calls_ganpg,
            'iterations': iterations_ganpg,
            'mse': mse_ganpg,
            'f1': f1_ganpg,
            'mu': anpg_ganpg_mu_fixed,
            'nu': nu_val
        }
        
        print(f"GANPG (ν={nu_val}) F1-Score: {f1_ganpg:.6f}")
    
    # Create and display metrics table
    metrics_df = pd.DataFrame(metrics_table, 
                             columns=['Algorithm', 'μ', 'ν', 'MSE', 'F1-Score', 'Precision', 'Recall', 'Iterations'])
    print("\n" + "="*100)
    print("Experiment 1: Performance Metrics Comparison")
    print("SPG: μ=0.1, ν=0.1 (fixed)")
    print("ANPG/GANPG: μ=200 (fixed), ν varying")
    print("="*100)
    print(metrics_df.round(6))
    print("="*100)
    
    # Save metrics to CSV
    metrics_df.to_csv('experiment1_performance_metrics.csv', index=False)
    print("Metrics saved to 'experiment1_performance_metrics.csv'")
    
    return results, metrics_df

def run_experiment_2():
    """Experiment 2: SPG fixed parameters vs ANPG/GANPG varying mu"""
    np.random.seed(42)
    
    # Parameters
    n = 1000
    m = 3000
    s = 50
    
    # Generate sparse signal x_true
    x_true = np.zeros(n)
    T = np.random.permutation(n)[:s]
    x_true[T] = np.sign(np.random.randn(s)) * (1 + 0.5 * np.random.rand(s))
    
    # Generate measurement matrix
    A_mat = np.random.randn(m, n)
    A_mat = A_mat / la.norm(A_mat, 2)
    
    # Generate measurements with outliers
    b_clean = A_mat @ x_true + 0.01 * np.random.randn(m)
    
    # Add 10% outliers
    contamination_rate = 0.1
    n_outliers = int(m * contamination_rate)
    outlier_indices = np.random.choice(m, n_outliers, replace=False)
    
    b_contaminated = b_clean.copy()
    outlier_magnitude = np.max(np.abs(b_clean)) * 5
    b_contaminated[outlier_indices] = outlier_magnitude * np.sign(np.random.randn(n_outliers))
    
    # Fixed parameters for SPG
    lambda_spg = 0.00004
    nu0_spg = 0.1
    mu0_spg = 0.1
    
    # Varying parameters for ANPG and GANPG
    mu_values = [50, 80, 100, 120, 200, 400]
    lambda_anpg = 0.00004
    nu0_anpg = 0.1
    
    # Store results
    results = {}
    metrics_table = []
    
    print("\nRunning Experiment 2: SPG fixed vs ANPG/GANPG varying mu")
    print("SPG: μ = 0.1, ν = 0.1 (fixed)")
    print("ANPG/GANPG: ν = 0.1 (fixed), μ varying")
    
    # Run SPG with fixed parameters
    print("Running SPG with fixed parameters...")
    x_spg, iterations_spg, final_F_spg, obj_values_spg, oracle_calls_spg = SPG(
        A_mat, b_contaminated, lambda_spg, nu0_spg, np.zeros(n), mu0_spg)
    
    mse_spg, f1_spg, precision_spg, recall_spg = calculate_metrics(x_true, x_spg)
    metrics_table.append(['SPG', mu0_spg, nu0_spg, mse_spg, f1_spg, precision_spg, recall_spg, iterations_spg])
    
    results['SPG'] = {
        'obj_values': obj_values_spg,
        'oracle_calls': oracle_calls_spg,
        'iterations': iterations_spg,
        'mse': mse_spg,
        'f1': f1_spg,
        'mu': mu0_spg,
        'nu': nu0_spg
    }
    
    print(f"SPG F1-Score: {f1_spg:.6f}")
    
    # Run ANPG and GANPG with varying μ parameters
    for mu_val in mu_values:
        print(f"\nTesting μ = {mu_val}")
        
        # ANPG
        print("Running ANPG...")
        x_anpg, iterations_anpg, final_F_anpg, obj_values_anpg, oracle_calls_anpg = ANPG(
            A_mat, b_contaminated, lambda_anpg, nu0_anpg, np.zeros(n), mu_val)
        
        mse_anpg, f1_anpg, precision_anpg, recall_anpg = calculate_metrics(x_true, x_anpg)
        metrics_table.append(['ANPG', mu_val, nu0_anpg, mse_anpg, f1_anpg, precision_anpg, recall_anpg, iterations_anpg])
        
        results[f'ANPG_μ{mu_val}'] = {
            'obj_values': obj_values_anpg,
            'oracle_calls': oracle_calls_anpg,
            'iterations': iterations_anpg,
            'mse': mse_anpg,
            'f1': f1_anpg,
            'mu': mu_val,
            'nu': nu0_anpg
        }
        
        print(f"ANPG (μ={mu_val}) F1-Score: {f1_anpg:.6f}")
        
        # GANPG
        print("Running GANPG...")
        x_ganpg, iterations_ganpg, final_F_ganpg, obj_values_ganpg, oracle_calls_ganpg = GANPG(
            A_mat, b_contaminated, lambda_anpg, nu0_anpg, np.zeros(n), mu_val)
        
        mse_ganpg, f1_ganpg, precision_ganpg, recall_ganpg = calculate_metrics(x_true, x_ganpg)
        metrics_table.append(['GANPG', mu_val, nu0_anpg, mse_ganpg, f1_ganpg, precision_ganpg, recall_ganpg, iterations_ganpg])
        
        results[f'GANPG_μ{mu_val}'] = {
            'obj_values': obj_values_ganpg,
            'oracle_calls': oracle_calls_ganpg,
            'iterations': iterations_ganpg,
            'mse': mse_ganpg,
            'f1': f1_ganpg,
            'mu': mu_val,
            'nu': nu0_anpg
        }
        
        print(f"GANPG (μ={mu_val}) F1-Score: {f1_ganpg:.6f}")
    
    # Create and display metrics table
    metrics_df = pd.DataFrame(metrics_table, 
                             columns=['Algorithm', 'μ', 'ν', 'MSE', 'F1-Score', 'Precision', 'Recall', 'Iterations'])
    print("\n" + "="*100)
    print("Experiment 2: Performance Metrics Comparison")
    print("SPG: μ=0.1, ν=0.1 (fixed)")
    print("ANPG/GANPG: ν=0.1 (fixed), μ varying")
    print("="*100)
    print(metrics_df.round(6))
    print("="*100)
    
    # Save metrics to CSV
    metrics_df.to_csv('experiment2_performance_metrics.csv', index=False)
    print("Metrics saved to 'experiment2_performance_metrics.csv'")
    
    return results, metrics_df

def plot_experiment1_results(results):
    """Plot results for Experiment 1: varying nu"""
    plot_sci_style()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Colors for different nu values
    nu_values = [0.05, 0.08, 0.1, 0.12, 0.15]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Left plot: ANPG
    for i, nu_val in enumerate(nu_values):
        key = f'ANPG_ν{nu_val}'
        if key in results:
            anpg_data = results[key]
            oracle_calls_anpg = anpg_data['oracle_calls']
            obj_values_anpg = anpg_data['obj_values']
            
            # Truncate data to minimum length
            min_len = min(len(oracle_calls_anpg), len(obj_values_anpg))
            oracle_calls_anpg = oracle_calls_anpg[:min_len]
            obj_values_anpg = obj_values_anpg[:min_len]
            
            ax1.plot(oracle_calls_anpg, obj_values_anpg,
                    color=colors[i], linestyle='-', linewidth=1.8,
                    label=rf'$\nu_0$={nu_val}')
    
    # Right plot: GANPG
    for i, nu_val in enumerate(nu_values):
        key = f'GANPG_ν{nu_val}'
        if key in results:
            ganpg_data = results[key]
            oracle_calls_ganpg = ganpg_data['oracle_calls']
            obj_values_ganpg = ganpg_data['obj_values']
            
            # Truncate data to minimum length
            min_len = min(len(oracle_calls_ganpg), len(obj_values_ganpg))
            oracle_calls_ganpg = oracle_calls_ganpg[:min_len]
            obj_values_ganpg = obj_values_ganpg[:min_len]
            
            ax2.plot(oracle_calls_ganpg, obj_values_ganpg,
                    color=colors[i], linestyle='-', linewidth=1.8,
                    label=rf'$\nu_0$={nu_val}')
    
    # SPG shown in both plots
    if 'SPG_fixed' in results:
        spg_data = results['SPG_fixed']
        oracle_calls_spg = np.array(spg_data['oracle_calls'])
        obj_values_spg = spg_data['obj_values']
        
        min_len = min(len(oracle_calls_spg), len(obj_values_spg))
        oracle_calls_spg = oracle_calls_spg[:min_len]
        obj_values_spg = obj_values_spg[:min_len]
        
        ax1.plot(oracle_calls_spg, obj_values_spg,
                color='black', linestyle='--', linewidth=2.5,
                label='SPG')
        ax2.plot(oracle_calls_spg, obj_values_spg,
                color='black', linestyle='--', linewidth=2.5,
                label='SPG')
    
    # Set symmetric logarithmic scale
    ax1.set_xscale('symlog', linthresh=1)
    ax2.set_xscale('symlog', linthresh=1)
    
    # Left plot settings
    ax1.set_xlabel('Oracle calls', fontsize=18)
    ax1.set_ylabel('Objective Function Value', fontsize=18)
    ax1.legend(fontsize=16, title='ANPG ($\mu_1=200$)', title_fontsize=18, frameon=False)
    
    # Right plot settings
    ax2.set_xlabel('Oracle calls', fontsize=18)
    ax2.legend(fontsize=16, title='GANPG ($\mu_1=200$)', title_fontsize=18, frameon=False)
    
    # Right plot: keep tick marks but hide tick labels
    ax2.tick_params(axis='y', which='both', labelleft=False, left=True)
    
    # Compact layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03)
    
    plt.savefig('experiment1_objective_convergence.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('experiment1_objective_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_experiment2_results(results):
    """Plot results for Experiment 2: varying mu"""
    plot_sci_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Colors for different mu values
    mu_values = [50, 80, 100, 120, 200, 400]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Left plot: ANPG
    for i, mu_val in enumerate(mu_values):
        key = f'ANPG_μ{mu_val}'
        if key in results:
            anpg_data = results[key]
            oracle_calls_anpg = anpg_data['oracle_calls']
            obj_values_anpg = anpg_data['obj_values']
            
            min_len = min(len(oracle_calls_anpg), len(obj_values_anpg))
            oracle_calls_anpg = oracle_calls_anpg[:min_len]
            obj_values_anpg = obj_values_anpg[:min_len]
            
            ax1.plot(oracle_calls_anpg, obj_values_anpg,
                    color=colors[i], linestyle='-', linewidth=1.8,
                    label=f'$\mu_1$={mu_val}')
    
    # Right plot: GANPG
    for i, mu_val in enumerate(mu_values):
        key = f'GANPG_μ{mu_val}'
        if key in results:
            ganpg_data = results[key]
            oracle_calls_ganpg = ganpg_data['oracle_calls']
            obj_values_ganpg = ganpg_data['obj_values']
            
            min_len = min(len(oracle_calls_ganpg), len(obj_values_ganpg))
            oracle_calls_ganpg = oracle_calls_ganpg[:min_len]
            obj_values_ganpg = obj_values_ganpg[:min_len]
            
            ax2.plot(oracle_calls_ganpg, obj_values_ganpg,
                    color=colors[i], linestyle='-', linewidth=1.8,
                    label=f'$\mu_1$={mu_val}')
    
    # SPG shown in both plots
    if 'SPG' in results:
        spg_data = results['SPG']
        oracle_calls_spg = np.array(spg_data['oracle_calls'])
        obj_values_spg = spg_data['obj_values']
        
        min_len = min(len(oracle_calls_spg), len(obj_values_spg))
        oracle_calls_spg = oracle_calls_spg[:min_len]
        obj_values_spg = obj_values_spg[:min_len]
        
        ax1.plot(oracle_calls_spg, obj_values_spg,
                color='black', linestyle='--', linewidth=2.5,
                label='SPG')
        ax2.plot(oracle_calls_spg, obj_values_spg,
                color='black', linestyle='--', linewidth=2.5,
                label='SPG')
    
    # Set symmetric logarithmic scale
    ax1.set_xscale('symlog', linthresh=1)
    ax2.set_xscale('symlog', linthresh=1)
    
    # Left plot settings
    ax1.set_xlabel('Oracle calls', fontsize=18)
    ax1.set_ylabel('Objective Function Value', fontsize=18)
    ax1.legend(fontsize=16, title=rf'ANPG ($\nu_0=0.1$)', title_fontsize=18, frameon=False)
    
    # Right plot settings
    ax2.set_xlabel('Oracle calls', fontsize=18)
    ax2.legend(fontsize=16, title=rf'GANPG ($\nu_0=0.1$)', title_fontsize=18, frameon=False)
    
    # Right plot: keep tick marks but hide tick labels
    ax2.tick_params(axis='y', which='both', labelleft=False, left=True)
    
    # Compact layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03)
    
    plt.savefig('experiment2_objective_convergence.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('experiment2_objective_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run both experiments
    print("Starting Experiment 1: Varying nu")
    results1, metrics_df1 = run_experiment_1()
    
    print("\nStarting Experiment 2: Varying mu")
    results2, metrics_df2 = run_experiment_2()
    
    # Plot results for both experiments
    print("\nPlotting Experiment 1 results...")
    plot_experiment1_results(results1)
    
    print("Plotting Experiment 2 results...")
    plot_experiment2_results(results2)
    
    print("\nAll experiments completed successfully!")