# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 14:39:10 2025

@author: 22472
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy.linalg import solve
from sklearn.linear_model import Lasso
from sklearn.linear_model import QuantileRegressor
import time
import pandas as pd
import os
from CSpack import CSpack

# --- Matplotlib global style ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 16,
    'axes.labelsize': 24,
    'axes.titlesize': 18,
    'legend.fontsize': 16,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.linewidth': 0.8,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

# ========== Algorithm Functions ==========

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
        
        # Update cumulative oracle call count
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
    return x, iter + 1, F_values

def prox_l1_scaled(v, b, lambda_val, m):
    """Compute the proximal operator for f(y) = (1/m) * ||y - b||_1"""
    v = np.asarray(v)
    b = np.asarray(b)
    t = lambda_val / m
    
    a = v - b
    y_star = b + np.sign(a) * np.maximum(np.abs(a) - t, 0)
    
    return y_star

def moreau_envelope_l1_scaled(v, b, lambda_val, m):
    """Compute the Moreau envelope value for f(y) = (1/m) * ||y - b||_1"""
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
    """Compute Moreau envelope value via proximal operator result"""
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

def prox_maLq(a, alpha_lam, q):
    """Proximal operator"""
    a = np.asarray(a)
    
    if q == 0:
        threshold = np.sqrt(2 * alpha_lam)
        result = a.copy()
        result[np.abs(a) <= threshold] = 0
        return result
        
    elif q == 0.5:
        threshold = (3/2) * alpha_lam**(2/3)
        result = np.zeros_like(a)
        mask = np.abs(a) > threshold
        a_masked = a[mask]
        phi = np.arccos((alpha_lam/4) * (3/np.abs(a_masked))**(3/2))
        result[mask] = (4/3) * a_masked * (np.cos((np.pi - phi)/3))**2
        return result
        
    else:
        # Simplified handling: use soft thresholding approximation for other q values
        threshold = alpha_lam
        result = np.sign(a) * np.maximum(np.abs(a) - threshold, 0)
        return result

# Define Lp norm solver for 1/2||Ax-b||_2^2+lambda||x||_p^p
def Lp_norm_solver(A, b, lambda_, p, x0):
    epsilon = 1e-6
    x = x0.copy()
    n = len(x)
    alpha = la.norm(A, 2)**2
    for iter in range(2000):
        gf = A.T @ (A @ x - b)
        z = prox_maLq(x - gf / alpha, lambda_ / alpha, p)
        TERM = np.linalg.norm(A @ x - b)**2 / 2 + lambda_ * np.sum(np.abs(x)**p) - (np.linalg.norm(A @ z - b)**2 / 2 + lambda_ * np.sum(np.abs(z)**p))
        if np.abs(TERM) < epsilon:
                break
        x = z.copy()
    return x, iter

# Fixed HA function return statement
def HA(A, b, lambda_, t, x0):
    epsilon = 1e-4
    eta = 1.0e-1
    x = x0.copy()
    n = len(x)
    s = lambda_ / t
    x_new = x.copy()
    alpha = la.norm(A, 2)**2
    
    hzt = np.sum(np.maximum(x - t, 0) + np.maximum(0, -x - t))
    FO_rho = np.linalg.norm(A @ x - b)**2 / 2 + s * (np.linalg.norm(x, 1) - hzt)

    for iter in range(2000):
        dxt = np.maximum(x - t, 0) + np.maximum(0, -x - t)
        II = np.zeros(n)
        II[x - t == dxt] = 1
        II[0 == dxt] = 0
        II[-x - t == dxt] = -1
        gf = A.T @ (A @ x - b)
        gx = II
        z = np.maximum(s * gx - gf + alpha * x - s, np.minimum(0, s * gx - gf + alpha * x + s)) / alpha
        hzt = np.sum(np.maximum(z - t, 0) + np.maximum(0, -z - t))
        F_rho = np.linalg.norm(A @ z - b)**2 / 2 + s * (np.linalg.norm(z, 1) - hzt)
        TERM = FO_rho - F_rho

        if TERM >= epsilon:
            x_new = z
        else:
            I = np.zeros((n, 3))
            dxt = np.maximum(x - t, 0) + np.maximum(0, -x - t) - eta
            I[(x - t) >= dxt, 0] = 1
            I[0 >= dxt, 1] = 1
            I[(-x - t) >= dxt, 2] = 1
            Is = np.sum(I, axis=1)
            num = np.sum(Is == 2)
            Num = 2**num
            funv = np.inf

            for i in range(1, Num + 1):
                i22 = np.array([int(x) for x in np.binary_repr(i - 1, width=num)])
                gx_temp = gx.copy()
                mask = Is == 2
                gx_temp[mask] = np.sign(x[mask]) * i22
                gf = A.T @ (A @ x - b)
                z = np.maximum(s * gx_temp - gf + alpha * x - s, 
                              np.minimum(0, s * gx_temp - gf + alpha * x + s)) / alpha
                hzt = np.sum(np.maximum(z - t, 0) + np.maximum(0, -z - t))
                F_rho = np.linalg.norm(A @ z - b)**2 / 2 + s * (np.linalg.norm(z, 1) - hzt)
                TERM = FO_rho - F_rho

                if TERM >= epsilon:
                    x_new = z
                    break
                
                l_rho = (gf - s * gx_temp).T @ (z - x) + 1/2 * np.linalg.norm(z - x)**2 + s * np.linalg.norm(z, 1) - s * np.sum(gx_temp * z - gx_temp**2 * t)
                if l_rho < funv:
                    x_new = z
                    funv = l_rho

            hzt = np.sum(np.maximum(x_new - t, 0) + np.maximum(0, -x_new - t))
            F_rho = np.linalg.norm(A @ x_new - b)**2 / 2 + s * (np.linalg.norm(x_new, 1) - hzt)
            TERM = FO_rho - F_rho
            
            if TERM < epsilon:
                if epsilon <= 1e-6:
                    break
                epsilon *= 0.1
        
        FO_rho = F_rho
        x = x_new.copy()
    
    return x, iter

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
    oracle_calls = [0]
    
    beta = 0
    
    # Initial gradient computation
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
        
        # Line search
        line_search_oracles = 0
        for i in range(20):
            alpha = alpha0 * 2**i
            w = y - gamma/alpha * grad
            s = gamma/alpha * lambda_ / nu
            w = w + s * d
            x_hat = np.sign(w) * np.maximum(np.abs(w) - s, 0)
        
            # Compute Moreau envelope
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
        
        # Compute next iteration's y
        y = x + beta * (x - x_old) * np.abs(d)
        # Compute gradient
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
        
    return x, iter, F_values

def GANPG(A, b, lambda_, nu0, x0, mu_0):
    """GANPG algorithm implementation"""
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
    
    beta = 0
    
    # Initial gradient calculation
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
        
        # Line search
        for i in range(20):
            alpha = alpha0 * 2**i
            w = y - gamma/alpha * grad
            s = gamma/alpha * lambda_ / nu
            w = w + s * d
            x_hat = np.sign(w) * np.maximum(np.abs(w) - s, 0)
            
            # Compute Moreau envelope
            M_hat = moreau_envelope_l1_scaled(A @ x_hat, b, mu, m)
            
            if M_hat <= M_y + grad.T @ (x_hat - y) + alpha/(2*gamma) * np.linalg.norm(x_hat - y)**2:
                break
        alpha0 = alpha
        Alpha.append(alpha)
        x_old = x.copy()
        x = x_hat.copy()
        
        # Update parameters
        beta = iter / (iter + 3.1)
        
        # Compute next iterate y
        y = x + beta * (x - x_old) * np.abs(d)
        
        # Compute gradient
        Ay = A @ y
        mu = mu_0 * norm_A / ((iter + 3.1) * np.log(iter + 3.1) ** (11/20))
        gamma = mu / norm_A
        prox_Ay = prox_l1_scaled(Ay, b, mu, m)
        grad = 1/mu * A.T @ (Ay - prox_Ay)
        
        M_y = moreau_from_prox(Ay, b, mu, m, prox_Ay)[0]
        
        # Compute new objective function value
        hzt = np.sum(np.maximum(x - nu, 0) + np.maximum(0, -x - nu))
        F_rho = 1/m * np.linalg.norm(A @ x - b, 1) + lambda_ / nu * (np.linalg.norm(x, 1) - hzt)
        F_values.append(F_rho)
        
        # Check convergence condition
        if np.abs(FO_rho - F_rho) < epsilon:
            break
            
        FO_rho = F_rho
    
    final_F_value = F_rho
    return x, iter, F_values

# LAD-LASSO implementation
def lad_lasso(A, b, alpha):
    """LAD-LASSO implementation"""
    n_features = A.shape[1]
    
    # Use QuantileRegressor to implement LAD-LASSO
    model = QuantileRegressor(alpha=alpha, quantile=0.5, solver='highs', fit_intercept=False)
    model.fit(A, b)
    
    return model.coef_

# Standard LASSO implementation
def standard_lasso(A, b, alpha):
    """Standard LASSO implementation"""
    lasso = Lasso(alpha=alpha, max_iter=5000, fit_intercept=False)
    lasso.fit(A, b)
    return lasso.coef_

# Grid search for optimal parameters
def grid_search_lasso(A, b, x_true, true_support):
    """Grid search for LASSO to find optimal parameters (based on MSE minimization)"""
    alpha_values = np.logspace(-3, -1, 20)
    best_mse = np.inf
    best_x = None
    best_alpha = None
    
    for alpha in alpha_values:
        try:
            x_lasso = standard_lasso(A, b, alpha)
            mse = np.linalg.norm(x_lasso - x_true)**2 / len(x_true)
            
            if mse < best_mse:
                best_mse = mse
                best_x = x_lasso
                best_alpha = alpha
        except:
            continue
    
    if best_alpha is not None:
        start_time = time.perf_counter()
        best_x = standard_lasso(A, b, best_alpha)
        solve_time = time.perf_counter() - start_time
        return best_x, best_mse, best_alpha, solve_time
    
    return None, np.inf, None, np.inf

def grid_search_lad_lasso(A, b, x_true, true_support):
    """Grid search for LAD-LASSO to find optimal parameters (based on MSE minimization)"""
    alpha_values = np.logspace(-3, -1, 20)
    best_mse = np.inf
    best_x = None
    best_alpha = None
    
    for alpha in alpha_values:
        try:
            x_lad = lad_lasso(A, b, alpha)
            mse = np.linalg.norm(x_lad - x_true)**2 / len(x_true)
            
            if mse < best_mse:
                best_mse = mse
                best_x = x_lad
                best_alpha = alpha
        except:
            continue
    
    if best_alpha is not None:
        start_time = time.perf_counter()
        best_x = lad_lasso(A, b, best_alpha)
        solve_time = time.perf_counter() - start_time
        return best_x, best_mse, best_alpha, solve_time
    
    return None, np.inf, None, np.inf

def grid_search_HA(A, b, x_true, true_support):
    """Grid search for HA algorithm for lambda (based on MSE minimization)"""
    lambda_values = np.logspace(-2, -1, 20)
    t = 0.1
    
    best_mse = np.inf
    best_x = None
    best_lambda = None
    best_iter = None
    
    for lambda_val in lambda_values:
        try:
            x0 = np.zeros(A.shape[1])
            x_ha, iter_ha = HA(A, b, lambda_val, t, x0)
            mse = np.linalg.norm(x_ha - x_true)**2 / len(x_true)
            
            if mse < best_mse:
                best_mse = mse
                best_x = x_ha
                best_lambda = lambda_val
                best_iter = iter_ha
        except Exception as e:
            continue
    
    if best_lambda is not None:
        x0 = np.zeros(A.shape[1])
        start_time = time.perf_counter()
        best_x, best_iter = HA(A, b, best_lambda, t, x0)
        solve_time = time.perf_counter() - start_time
        return best_x, best_mse, best_lambda, best_iter, solve_time
    
    return None, np.inf, None, None, np.inf

def grid_search_Lp(A, b, x_true, true_support):
    """Grid search for Lp_norm_solver algorithm for lambda (based on MSE minimization)"""
    lambda_values = np.logspace(-2, -1, 20)
    p = 0.5
    
    best_mse = np.inf
    best_x = None
    best_lambda = None
    best_iter = None
    
    for lambda_val in lambda_values:
        try:
            x0 = np.zeros(A.shape[1])
            x_lp, iter_lp = Lp_norm_solver(A, b, lambda_val, p, x0)
            mse = np.linalg.norm(x_lp - x_true)**2 / len(x_true)
            
            if mse < best_mse:
                best_mse = mse
                best_x = x_lp
                best_lambda = lambda_val
                best_iter = iter_lp
        except Exception as e:
            continue
    
    if best_lambda is not None:
        x0 = np.zeros(A.shape[1])
        start_time = time.perf_counter()
        best_x, best_iter = Lp_norm_solver(A, b, best_lambda, p, x0)
        solve_time = time.perf_counter() - start_time
        return best_x, best_mse, best_lambda, best_iter, solve_time
    
    return None, np.inf, None, None, np.inf



# NL0R algorithm wrapper
def run_nl0r(A_mat, b, x_true, true_support, n, s):
    """Run NL0R algorithm"""
    try:
        A_func = lambda var: A_mat @ var
        At_func = lambda var: A_mat.T @ var
        
        start_time = time.perf_counter()
        out = CSpack(A=A_func, At=At_func, b=b, n=n, s=s, solver='NL0R')
        solve_time = time.perf_counter() - start_time
        x_nl0r = out['sol']
        
        iter_nl0r = out.get('iter', 'N/A')
        
        return x_nl0r, iter_nl0r, solve_time
        
    except ImportError:
        return None, 'N/A', np.inf
    except Exception as e:
        return None, 'N/A', np.inf

# Calculate F1-score and other metrics
def calculate_metrics(x_pred, x_true, true_support, n):
    """Calculate precision, recall, and F1-score"""
    threshold = 1e-4
    predicted_support = np.where(np.abs(x_pred) > threshold)[0]
    
    TP = len(np.intersect1d(predicted_support, true_support))
    FP = len(np.setdiff1d(predicted_support, true_support))
    FN = len(np.setdiff1d(true_support, predicted_support))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    mse = np.linalg.norm(x_pred - x_true)**2 / n
    
    nnz = len(predicted_support)
    
    return precision, recall, f1_score, mse, nnz, TP, FP, FN

# Format MSE value in scientific notation
def format_mse_scientific(mse_value):
    """Format MSE value as scientific notation display"""
    if mse_value == np.inf:
        return "Inf"
    elif mse_value == 0:
        return "0.000e+00"
    else:
        return f"{mse_value:.3e}"

# Run single outlier proportion experiment
# Run single outlier proportion experiment
def run_single_experiment(contamination_rate, n, m, s, A_mat, xopt, T):
    """Run experiment for single outlier proportion"""
    print(f"\nProcessing outlier proportion: {contamination_rate*100}%")
    
    # Generate clean measurement vector b
    b_clean = A_mat[:, T] @ xopt[T] + 0.01 * np.random.randn(m)
    
    # Add contamination
    n_outliers = int(m * contamination_rate)
    outlier_indices = np.random.choice(m, n_outliers, replace=False)
    
    b_contaminated = np.copy(b_clean)
    outlier_magnitude = np.max(np.abs(b_clean)) * 5
    b_contaminated[outlier_indices] = outlier_magnitude * np.sign(np.random.randn(n_outliers))
    
    # Store results
    results = {}
    x_solutions = {}
    
    # 1. ANPG algorithm test
    print("ANPG algorithm:")
    x0 = np.zeros(n)
    lambda_ = 0.0004
    nu = 0.1
    mu_0 = 60
    
    start_time = time.perf_counter()
    x_anpg, iter_anpg, F_values = ANPG(A_mat, b_contaminated, lambda_, nu, x0, mu_0)
    solve_time_anpg = time.perf_counter() - start_time
    
    if solve_time_anpg < 1e-6:
        solve_time_anpg = 1e-6
    
    precision_anpg, recall_anpg, f1_anpg, mse_anpg, nnz_anpg, TP, FP, FN = calculate_metrics(x_anpg, xopt, T, n)
    
    results['ANPG'] = {
        'Precision': precision_anpg, 'Recall': recall_anpg, 'F1-score': f1_anpg, 'MSE': mse_anpg,
        'Iterations': iter_anpg, 'Non-zero count': nnz_anpg, 'Time (s)': solve_time_anpg,
        'TP': TP, 'FP': FP, 'FN': FN
    }
    x_solutions['ANPG'] = x_anpg
    
    # 2. GANPG algorithm test (Using the same parameters as ANPG)
    print("GANPG algorithm:")
    x0 = np.zeros(n)
    lambda_ = 0.0004
    nu = 0.1
    mu_0 = 60
    
    start_time = time.perf_counter()
    x_ganpg, iter_ganpg, F_values_ganpg = GANPG(A_mat, b_contaminated, lambda_, nu, x0, mu_0)
    solve_time_ganpg = time.perf_counter() - start_time
    
    if solve_time_ganpg < 1e-6:
        solve_time_ganpg = 1e-6
    
    precision_ganpg, recall_ganpg, f1_ganpg, mse_ganpg, nnz_ganpg, TP, FP, FN = calculate_metrics(x_ganpg, xopt, T, n)
    
    results['GANPG'] = {
        'Precision': precision_ganpg, 'Recall': recall_ganpg, 'F1-score': f1_ganpg, 'MSE': mse_ganpg,
        'Iterations': iter_ganpg, 'Non-zero count': nnz_ganpg, 'Time (s)': solve_time_ganpg,
        'TP': TP, 'FP': FP, 'FN': FN
    }
    x_solutions['GANPG'] = x_ganpg
    
    # 3. SPG algorithm test
    print("SPG algorithm:")
    x0 = np.zeros(n)
    lambda_ = 0.0004
    nu = 0.1
    mu_0 = 0.1  # Using μ=0.1
    
    start_time = time.perf_counter()
    x_spg, iter_spg, F_values_spg = SPG(A_mat, b_contaminated, lambda_, nu, x0, mu_0)
    solve_time_spg = time.perf_counter() - start_time
    
    if solve_time_spg < 1e-6:
        solve_time_spg = 1e-6
    
    precision_spg, recall_spg, f1_spg, mse_spg, nnz_spg, TP, FP, FN = calculate_metrics(x_spg, xopt, T, n)
    
    results['SPG'] = {
        'Precision': precision_spg, 'Recall': recall_spg, 'F1-score': f1_spg, 'MSE': mse_spg,
        'Iterations': iter_spg, 'Non-zero count': nnz_spg, 'Time (s)': solve_time_spg,
        'TP': TP, 'FP': FP, 'FN': FN
    }
    x_solutions['SPG'] = x_spg
    
    # 4. HA algorithm test
    print("HA algorithm:")
    x_ha, ha_mse, best_lambda_ha, iter_ha, solve_time_ha = grid_search_HA(A_mat, b_contaminated, xopt, T)
    
    if x_ha is not None:
        if solve_time_ha < 1e-6:
            solve_time_ha = 1e-6
            
        precision_ha, recall_ha, f1_ha, mse_ha, nnz_ha, TP, FP, FN = calculate_metrics(x_ha, xopt, T, n)
        
        results['HA'] = {
            'Precision': precision_ha, 'Recall': recall_ha, 'F1-score': f1_ha, 'MSE': mse_ha,
            'Iterations': iter_ha, 'Non-zero count': nnz_ha, 'Time (s)': solve_time_ha,
            'TP': TP, 'FP': FP, 'FN': FN
        }
        x_solutions['HA'] = x_ha
    else:
        print("HA algorithm failed to find valid solution")
    
    # 5. Lp_norm_solver algorithm test
    print("Lp_norm_solver algorithm:")
    x_lp, lp_mse, best_lambda_lp, iter_lp, solve_time_lp = grid_search_Lp(A_mat, b_contaminated, xopt, T)
    
    if x_lp is not None:
        if solve_time_lp < 1e-6:
            solve_time_lp = 1e-6
            
        precision_lp, recall_lp, f1_lp, mse_lp, nnz_lp, TP, FP, FN = calculate_metrics(x_lp, xopt, T, n)
        
        results['Lp_norm_solver'] = {
            'Precision': precision_lp, 'Recall': recall_lp, 'F1-score': f1_lp, 'MSE': mse_lp,
            'Iterations': iter_lp, 'Non-zero count': nnz_lp, 'Time (s)': solve_time_lp,
            'TP': TP, 'FP': FP, 'FN': FN
        }
        x_solutions['Lp_norm_solver'] = x_lp
    else:
        print("Lp_norm_solver algorithm failed to find valid solution")
    
    # 6. NL0R algorithm test
    print("NL0R algorithm:")
    x_nl0r, iter_nl0r, time_nl0r = run_nl0r(A_mat, b_contaminated, xopt, T, n, s)
    
    if x_nl0r is not None:
        if time_nl0r < 1e-6:
            time_nl0r = 1e-6
            
        precision_nl0r, recall_nl0r, f1_nl0r, mse_nl0r, nnz_nl0r, TP, FP, FN = calculate_metrics(x_nl0r, xopt, T, n)
        
        results['NL0R'] = {
            'Precision': precision_nl0r, 'Recall': recall_nl0r, 'F1-score': f1_nl0r, 'MSE': mse_nl0r,
            'Iterations': iter_nl0r, 'Non-zero count': nnz_nl0r, 'Time (s)': time_nl0r,
            'TP': TP, 'FP': FP, 'FN': FN
        }
        x_solutions['NL0R'] = x_nl0r
    else:
        print("NL0R algorithm failed to find valid solution")
    
    # 7. LASSO algorithm test
    print("LASSO algorithm:")
    x_lasso, lasso_mse, best_alpha_lasso, solve_time_lasso = grid_search_lasso(A_mat, b_contaminated, xopt, T)
    
    if x_lasso is not None:
        if solve_time_lasso < 1e-6:
            solve_time_lasso = 1e-6
            
        precision_lasso, recall_lasso, f1_lasso, mse_lasso, nnz_lasso, TP, FP, FN = calculate_metrics(x_lasso, xopt, T, n)
        
        results['LASSO'] = {
            'Precision': precision_lasso, 'Recall': recall_lasso, 'F1-score': f1_lasso, 'MSE': mse_lasso,
            'Iterations': 'N/A', 'Non-zero count': nnz_lasso, 'Time (s)': solve_time_lasso,
            'TP': TP, 'FP': FP, 'FN': FN
        }
        x_solutions['LASSO'] = x_lasso
    else:
        print("LASSO algorithm failed to find valid solution")
    
    # 8. LAD-LASSO algorithm test
    print("LAD-LASSO algorithm:")
    x_lad, lad_mse, best_alpha_lad, solve_time_lad = grid_search_lad_lasso(A_mat, b_contaminated, xopt, T)
    
    if x_lad is not None:
        if solve_time_lad < 1e-6:
            solve_time_lad = 1e-6
            
        precision_lad, recall_lad, f1_lad, mse_lad, nnz_lad, TP, FP, FN = calculate_metrics(x_lad, xopt, T, n)
        
        results['LAD-LASSO'] = {
            'Precision': precision_lad, 'Recall': recall_lad, 'F1-score': f1_lad, 'MSE': mse_lad,
            'Iterations': 'N/A', 'Non-zero count': nnz_lad, 'Time (s)': solve_time_lad,
            'TP': TP, 'FP': FP, 'FN': FN
        }
        x_solutions['LAD-LASSO'] = x_lad
    else:
        print("LAD-LASSO algorithm failed to find valid solution")
    
    # Print detailed results
    print(f"\nResults for outlier proportion {contamination_rate*100}%:")
    print(f"{'Algorithm':<15} {'Precision':<10} {'Recall':<8} {'F1-score':<10} {'MSE':<12} {'Iterations':<12} {'Non-zero':<10} {'Time (s)':<10}")
    print("-" * 110)
    for algo in ['ANPG', 'GANPG', 'SPG', 'HA', 'Lp_norm_solver', 'NL0R', 'LASSO', 'LAD-LASSO']:
        if algo in results:
            data = results[algo]
            mse_display = format_mse_scientific(data['MSE'])
            iter_display = data['Iterations'] if data['Iterations'] != 'N/A' else 'N/A'
            print(f"{algo:<15} {data['Precision']:<10.4f} {data['Recall']:<8.4f} {data['F1-score']:<10.4f} {mse_display:<12} {iter_display:<12} {data['Non-zero count']:<10} {data['Time (s)']:<10.6f}")
    
    return results, x_solutions, b_contaminated

# ========== Plotting Functions ==========

def plot_individual_algorithms_stem(all_x_solutions, xopt, true_support, n, contamination_rates):
    """
    Plot 8 subplots for each outlier rate, using stem style to show recovery performance of each algorithm
    """
    algorithm_styles = {
        'ANPG': {'label': 'ANPG'},
        'GANPG': {'label': 'GANPG'},  # Added GANPG
        'SPG': {'label': 'SPG'},
        'HA': {'label': 'HA'},
        'Lp_norm_solver': {'label': 'IHT$_{1/2}$'},
        'NL0R': {'label': 'NL0R'},
        'LASSO': {'label': 'LASSO'},
        'LAD-LASSO': {'label': 'LAD-LASSO'}
    }

    algorithms = list(algorithm_styles.keys())
    
    # Modified to 8 subplots, 2 rows and 4 columns layout
    subplot_positions = [
        [0.04, 0.58, 0.20, 0.28],  # Row 1, Column 1
        [0.28, 0.58, 0.20, 0.28],  # Row 1, Column 2
        [0.52, 0.58, 0.20, 0.28],  # Row 1, Column 3
        [0.76, 0.58, 0.20, 0.28],  # Row 1, Column 4
        [0.04, 0.18, 0.20, 0.28],  # Row 2, Column 1
        [0.28, 0.18, 0.20, 0.28],  # Row 2, Column 2
        [0.52, 0.18, 0.20, 0.28],  # Row 2, Column 3
        [0.76, 0.18, 0.20, 0.28],  # Row 2, Column 4
    ]
    
    for rate in contamination_rates:
        fig = plt.figure(figsize=(20, 9))
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#f26419', linewidth=3, marker='o', markersize=8, 
                   markeredgewidth=1.5, markerfacecolor='none', label='Ground-Truth'),
            Line2D([0], [0], color='#1c8ddb', linewidth=3, linestyle=':', marker='o', markersize=8, 
                   markeredgewidth=1.2, markerfacecolor='#1c8ddb', label='Recovered')
        ]
        
        for i, algo in enumerate(algorithms):
            if i >= len(subplot_positions):  # Prevent index out of bounds
                break
                
            if algo in all_x_solutions[rate]:
                x_sol = all_x_solutions[rate][algo]
                
                pos = subplot_positions[i]
                ax = fig.add_axes(pos)
                
                xo_flat = xopt.flatten()
                x_flat = x_sol.flatten()
                
                # Plot ground truth
                idx_xo = np.where(xo_flat != 0)[0]
                vals_xo = xo_flat[idx_xo]
                
                if len(idx_xo) > 0:
                    markerline, stemlines, baseline = ax.stem(idx_xo, vals_xo, linefmt='-', markerfmt='o', basefmt=' ')
                    plt.setp(stemlines, 'color', '#f26419', 'linewidth', 1.5)
                    plt.setp(markerline, 'color', '#f26419', 'markersize', 8, 'markerfacecolor', 'none', 'markeredgewidth', 1.5)
                
                # Plot recovered signal
                idx_x = np.where(np.abs(x_flat) > 1e-4)[0]
                vals_x = x_flat[idx_x]
                
                if len(idx_x) > 0:
                    markerline, stemlines, baseline = ax.stem(idx_x, vals_x, linefmt=':', markerfmt='o', basefmt=' ')
                    plt.setp(stemlines, 'color', '#1c8ddb', 'linewidth', 1.5)
                    plt.setp(markerline, 'color', '#1c8ddb', 'markersize', 6, 'markerfacecolor', '#1c8ddb', 'markeredgewidth', 1.2)
                
                ax.grid(True, alpha=0.3)
                
                # Calculate performance metrics
                true_support_set = set(np.where(xo_flat != 0)[0])
                recovered_support_set = set(idx_x)
                correct_support = true_support_set & recovered_support_set
                
                recovery_rate = len(correct_support) / len(true_support_set) * 100 if len(true_support_set) > 0 else 0
                
                ax.set_title(f"{algorithm_styles[algo]['label']}\nRecovery: {len(correct_support)}/{len(true_support_set)} ({recovery_rate:.1f}%)", 
                           fontsize=11, weight='normal')
                
                ax.set_xlim([-2, 102])
                ax.set_xticks([0, 20, 40, 60, 80, 100])
                ax.tick_params(axis='x', labelsize=10)
                ax.tick_params(axis='y', labelsize=10)
                
                all_values = set(vals_xo) | set(vals_x)
                if all_values:
                    y_min = min(all_values) - 0.1
                    y_max = max(all_values) + 0.1
                    ax.set_ylim([y_min, y_max])
                
                # Add axis labels
                if i >= 4:  # Bottom row
                    ax.set_xlabel('Index', fontsize=12)
                if i in [0, 4]:  # Left column
                    ax.set_ylabel('$x_i$', fontsize=12)
        
        # Add legend
        fig.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.97), ncol=2, fontsize=16, frameon=True,
                  fancybox=True, shadow=True)
        
        # Add main title
        fig.suptitle(f'Sparse Signal Recovery Comparison (Outlier Rate: {rate*100:.0f}%)', 
                    fontsize=20, y=0.99)
        
        plt.savefig(f'results/algorithm_comparison_stem_rate_{rate*100:.0f}percent.png', 
                   dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.show()

# ========== Main Function ==========

def main():
    np.random.seed(42)
    
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")
    
    # Parameter settings
    n = 100
    m = 300
    s = 10
    
    # Generate sparse signal xopt
    xopt = np.zeros(n)
    T = np.random.permutation(n)[:s]
    xopt[T] = np.sign(np.random.randn(s))
    
    # Generate measurement matrix A_mat
    A_mat = np.random.randn(m, n)
    A_mat = A_mat / np.linalg.norm(A_mat, 2)
    
    # Outlier proportions
    contamination_rates = [0, 0.05, 0.1, 0.15, 0.2]
    
    # Store all results
    all_results = {}
    all_x_solutions = {}
    all_b_contaminated = {}
    
    print("=" * 60)
    print("Sparse Signal Recovery Algorithms - Multiple Outlier Proportion Tests")
    print("=" * 60)
    print(f"Signal dimension: {n}, Measurements: {m}, Sparsity: {s}")
    print(f"True non-zero element positions: {np.sort(T)}")
    print()
    
    # Run experiment for each outlier proportion
    for rate in contamination_rates:
        print(f"\n{'='*60}")
        print(f"Outlier proportion: {rate*100}%")
        print(f"{'='*60}")
        
        results, x_solutions, b_contaminated = run_single_experiment(rate, n, m, s, A_mat, xopt, T)
        
        all_results[rate] = results
        all_x_solutions[rate] = x_solutions
        all_b_contaminated[rate] = b_contaminated
    
    # Plot results
    print("\nPlotting individual algorithm recovery performance (stem style)...")
    plot_individual_algorithms_stem(all_x_solutions, xopt, T, n, contamination_rates)
    
    # Summarize all results
    print(f"\n{'='*120}")
    print("Summary of Results for All Outlier Proportions")
    print(f"{'='*120}")
    
    # Create summary table
    algorithms = ['ANPG', 'GANPG', 'SPG', 'HA', 'Lp_norm_solver', 'NL0R', 'LASSO', 'LAD-LASSO']
    metrics = ['Precision', 'Recall', 'F1-score', 'MSE', 'Iterations', 'Non-zero count', 'Time (s)']
    
    for metric in metrics:
        print(f"\n{metric} Comparison:")
        header = f"{'Outlier %':<12} " + " ".join([f"{algo:<12}" for algo in algorithms])
        print(header)
        print("-" * 120)
        for rate in contamination_rates:
            row = f"{rate*100:<12.1f}%"
            for algo in algorithms:
                if algo in all_results[rate]:
                    value = all_results[rate][algo][metric]
                    if isinstance(value, (int, float)):
                        if metric == 'MSE':
                            formatted_value = format_mse_scientific(value)
                            row += f"{formatted_value:<12}"
                        elif metric == 'Time (s)':
                            row += f"{value:<12.6f}"
                        else:
                            row += f"{value:<12.4f}"
                    else:
                        row += f"{value:<12}"
                else:
                    row += f"{'N/A':<12}"
            print(row)
    
    return all_results, all_x_solutions, xopt, T, all_b_contaminated

if __name__ == "__main__":
    all_results, all_x_solutions, xopt, T, all_b_contaminated = main()