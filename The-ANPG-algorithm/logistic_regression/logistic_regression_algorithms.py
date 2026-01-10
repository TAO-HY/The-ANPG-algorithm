# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 14:03:43 2025

@author: 22472
"""

import numpy as np
import time
from scipy.sparse import issparse
from scipy.sparse.linalg import svds
from sklearn.metrics import accuracy_score, classification_report
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, Union, Callable
from numpy.typing import ArrayLike
from scipy import sparse as sp
from scipy.sparse.linalg import LinearOperator, svds
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
from typing import Callable, Dict, Optional, Tuple, Union


class BaseLogisticRegression(ABC):
    """Base class for logistic regression algorithms"""
    
    def __init__(self, lambda_=1.0, t=1.0, max_iter=1000, tol=1e-4, alpha=None, verbose=False):
        self.lambda_ = lambda_
        self.t = t
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.verbose = verbose
        self.weights = None
        self.loss_history = []
        self.time_history = []
    
    def _sigmoid(self, z):
        """Sigmoid function with sparse matrix support"""
        if issparse(z):
            z = z.toarray()
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_lipschitz_constant(self, X):
        """Compute Lipschitz constant"""
        m = X.shape[0]
        if m == 0:
            return 1.0
        
        if issparse(X):
            try:
                spectral_norm = svds(X, k=1, which='LM', return_singular_vectors=False)[0]
            except:
                v = np.random.randn(X.shape[1])
                for _ in range(10):
                    v = X.T.dot(X.dot(v))
                    v_norm = np.linalg.norm(v)
                    if v_norm > 0:
                        v = v / v_norm
                Av = X.T.dot(X.dot(v))
                spectral_norm = np.sqrt(np.dot(v, Av))
        else:
            spectral_norm = np.linalg.svd(X, compute_uv=False, full_matrices=False)[0]
        
        L = (spectral_norm ** 2) / (4 * m)
        return L
    
    def _logistic_loss(self, X, y, w):
        """Compute logistic regression loss"""
        m = len(y)
        if issparse(X):
            z = X.dot(w)
        else:
            z = np.dot(X, w)
        
        h = self._sigmoid(z)
        epsilon = 1e-10
        h = np.clip(h, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
        return loss
    
    def _logistic_gradient(self, X, y, w):
        """Compute logistic regression gradient"""
        m = len(y)
        if issparse(X):
            z = X.dot(w)
        else:
            z = np.dot(X, w)
        
        h = self._sigmoid(z)
        error = h - y
        
        if issparse(X):
            gradient = X.T.dot(error) / m
            if issparse(gradient):
                gradient = gradient.toarray().flatten()
        else:
            gradient = np.dot(X.T, error) / m
            
        return gradient
    
    def _capped_l1_penalty(self, x, t):
        """Compute capped L1 penalty term"""
        hzt = np.sum(np.maximum(x - t, 0) + np.maximum(0, -x - t))
        return self.lambda_ / t * (np.linalg.norm(x, 1) - hzt)
    
    def _objective_function(self, X, y, w):
        """Compute objective function value"""
        loss = self._logistic_loss(X, y, w)
        penalty = self._capped_l1_penalty(w, self.t)
        return loss + penalty
    
    @abstractmethod
    def fit(self, X, y):
        """Train the model"""
        pass
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if self.weights is None:
            raise ValueError("Model not trained yet. Call fit method first.")
        
        if issparse(X):
            z = X.dot(self.weights)
        else:
            z = np.dot(X, self.weights)
        
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Predict classes"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def score(self, X, y):
        """Compute accuracy score"""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def get_nonzero_weights(self, threshold=1e-4):
        """Get number and indices of non-zero weights"""
        if self.weights is None:
            return 0, []
        
        non_zero_indices = np.where(np.abs(self.weights) > threshold)[0]
        return len(non_zero_indices), non_zero_indices
    
    def get_sparsity(self, threshold=1e-4):
        """Compute model sparsity"""
        if self.weights is None:
            return 0.0
        n_nonzero = np.sum(np.abs(self.weights) > threshold)
        return n_nonzero / len(self.weights)


class LqLogisticRegression(BaseLogisticRegression):
    """Lq norm regularized logistic regression"""
    
    def __init__(self, q=0.5, lambda_=0.1, max_iter=5000, tol=1e-6, alpha=None, verbose=False):  # 注意tol已改为1e-6
        super().__init__(lambda_, t=None, max_iter=max_iter, tol=tol, alpha=alpha, verbose=verbose)
        self.q = q
    
    def _prox_maLq(self, a, alpha_lam, q):
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
    
    def _objective_function(self, X, y, w):
        """Compute objective function value - override base method to use Lq penalty"""
        loss = self._logistic_loss(X, y, w)
        
        # Lq penalty term
        if self.q == 0:
            # L0 approximation
            penalty = self.lambda_ * np.sum(w != 0)
        elif self.q > 0:
            # Lq penalty (q != 1)
            penalty = self.lambda_ * np.sum(np.abs(w) ** self.q)
        else:
            penalty = 0
            
        return loss + penalty
    
    def fit(self, X, y):
        """Train the model"""
        n_samples, n_features = X.shape
        
        if len(y) != n_samples:
            raise ValueError(f"Input shape mismatch: X {X.shape}, y {y.shape}")
        
        # Initialize weights (assuming X already includes bias term)
        w = np.zeros(n_features)
        
        # Compute Lipschitz constant
        if self.alpha is None:
            self.alpha = self._compute_lipschitz_constant(X)
            if self.verbose:
                print(f"Computed Lipschitz constant: {self.alpha:.6f}")
        
        # Step size
        alpha_step = 1.0 / self.alpha if self.alpha > 0 else 1.0
        
        self.loss_history = []
        self.time_history = []
        start_time = time.time()
        
        # 初始化前一次损失值
        prev_loss = float('inf')
        
        for k in range(self.max_iter):
            current_time = time.time() - start_time
            self.time_history.append(current_time)
            
            # Compute current loss
            current_loss = self._objective_function(X, y, w)
            self.loss_history.append(current_loss)
            
            # 检查收敛：目标函数值变化小于1e-6
            if k > 0 and abs(current_loss - prev_loss) < 1e-6:  # 改为目标函数差小于1e-6
                if self.verbose:
                    print(f"Converged at iteration {k}: |loss_k - loss_{k-1}| = {abs(current_loss - prev_loss):.2e} < 1e-6")
                break
            
            # 更新前一次损失值
            prev_loss = current_loss
            
            # Compute gradient
            grad = self._logistic_gradient(X, y, w)
            
            # Gradient step
            gradient_step = w - alpha_step * grad
            
            # Proximal operator step
            w_new = self._prox_maLq(gradient_step, alpha_step * self.lambda_, self.q)
            
            w = w_new
            
            if self.verbose and k % 100 == 0:
                non_zero = np.sum(np.abs(w) > 1e-4)
                print(f"Iter {k:4d}, Loss: {current_loss:.6f}, ΔLoss: {abs(current_loss - (self.loss_history[-2] if k>0 else current_loss)):.2e}, Non-zero: {non_zero}")
        
        # Save weights
        self.weights = w
        
        if self.verbose:
            non_zero_final, _ = self.get_nonzero_weights()
            final_loss = self._objective_function(X, y, self.weights)
            print(f"Training completed. Final loss: {final_loss:.6f}, Non-zero weights: {non_zero_final}")
        
        return self


class HALogisticRegression(BaseLogisticRegression):
    """HA algorithm logistic regression"""
    
    def __init__(self, lambda_=1.0, t=1.0, max_iter=2000, tol=1e-4, alpha=None, verbose=False):
        super().__init__(lambda_, t, max_iter, tol, alpha, verbose)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        if len(y) != n_samples:
            raise ValueError(f"Input shape mismatch: X {X.shape}, y {y.shape}")
        
        if self.alpha is None:
            self.alpha = self._compute_lipschitz_constant(X)
            if self.verbose:
                print(f"Computed Lipschitz constant: {self.alpha:.6f}")
        
        x = np.zeros(n_features)
        epsilon = self.tol
        eta = 1.0e-1
        s = self.lambda_ / self.t
        
        hzt = np.sum(np.maximum(x - self.t, 0) + np.maximum(0, -x - self.t))
        FO_rho = self._logistic_loss(X, y, x) + s * (np.linalg.norm(x, 1) - hzt)
        
        self.loss_history = []
        self.time_history = []
        start_time = time.time()
        
        for iter in range(self.max_iter):
            current_time = time.time() - start_time
            self.time_history.append(current_time)
            
            s = self.lambda_ / self.t
            hzt = np.sum(np.maximum(x - self.t, 0) + np.maximum(0, -x - self.t))
            current_loss = self._objective_function(X, y, x)
            self.loss_history.append(current_loss)
            
            dxt = np.maximum(x - self.t, 0) + np.maximum(0, -x - self.t)
            II = np.zeros(n_features)
            II[x - self.t == dxt] = 1
            II[0 == dxt] = 0
            II[-x - self.t == dxt] = -1
            
            gf = self._logistic_gradient(X, y, x)
            gx = II
            
            z = np.maximum(s * gx - gf + self.alpha * x - s, 
                          np.minimum(0, s * gx - gf + self.alpha * x + s)) / self.alpha
            
            hzt_z = np.sum(np.maximum(z - self.t, 0) + np.maximum(0, -z - self.t))
            F_rho = self._logistic_loss(X, y, z) + s * (np.linalg.norm(z, 1) - hzt_z)
            TERM = FO_rho - F_rho
            
            if self.verbose and iter % 100 == 0:
                non_zero = np.sum(np.abs(x) > 1e-4)
                print(f"Iter {iter:4d}, Loss: {current_loss:.6f}, Non-zero: {non_zero}, TERM: {TERM:.6f}")
            
            if TERM >= epsilon:
                x_new = z
            else:
                I = np.zeros((n_features, 3))
                dxt_eta = np.maximum(x - self.t, 0) + np.maximum(0, -x - self.t) - eta
                I[(x - self.t) >= dxt_eta, 0] = 1
                I[0 >= dxt_eta, 1] = 1
                I[(-x - self.t) >= dxt_eta, 2] = 1
                Is = np.sum(I, axis=1)
                num = np.sum(Is == 2)
                
                # Add num > 20 stopping condition
                if num > 20:
                    if self.verbose:
                        print(f"Stopped at iteration {iter} because num={num}>20")
                    break
                
                Num = 2 ** num
                funv = np.inf
                x_candidate = x.copy()
                
                for i in range(1, Num + 1):
                    i22 = np.array([int(x) for x in np.binary_repr(i - 1, width=num)])
                    gx_temp = gx.copy()
                    gx_temp[Is == 2] = np.sign(x[Is == 2]) * i22
                    
                    z_temp = np.maximum(s * gx_temp - gf + self.alpha * x - s, 
                                      np.minimum(0, s * gx_temp - gf + self.alpha * x + s)) / self.alpha
                    
                    hzt_temp = np.sum(np.maximum(z_temp - self.t, 0) + np.maximum(0, -z_temp - self.t))
                    F_rho_temp = self._logistic_loss(X, y, z_temp) + s * (np.linalg.norm(z_temp, 1) - hzt_temp)
                    TERM_temp = FO_rho - F_rho_temp
                    
                    if TERM_temp >= epsilon:
                        x_new = z_temp
                        break
                    
                    l_rho = (gf - s * gx_temp).T @ (z_temp - x) + 0.5 * np.linalg.norm(z_temp - x)**2 + \
                           s * np.linalg.norm(z_temp, 1) - s * np.sum(gx_temp * z_temp - gx_temp**2 * self.t)
                    
                    if l_rho < funv:
                        x_candidate = z_temp
                        funv = l_rho
                else:
                    x_new = x_candidate
                
                hzt_new = np.sum(np.maximum(x_new - self.t, 0) + np.maximum(0, -x_new - self.t))
                F_rho_new = self._logistic_loss(X, y, x_new) + s * (np.linalg.norm(x_new, 1) - hzt_new)
                TERM_new = FO_rho - F_rho_new
                
                if TERM_new < epsilon:
                    if epsilon <= 1e-6:
                        if self.verbose:
                            print(f"Converged at iteration {iter}")
                        break
                    epsilon *= 0.1
                    if self.verbose:
                        print(f"Reduced tolerance to {epsilon}")
            
            FO_rho = F_rho if 'F_rho' in locals() else self._logistic_loss(X, y, x_new) + s * (np.linalg.norm(x_new, 1) - hzt_new)
            x = x_new
        
        self.weights = x
        if self.verbose:
            non_zero_final, _ = self.get_nonzero_weights()
            final_loss = self._objective_function(X, y, self.weights)
            print(f"Training completed. Final loss: {final_loss:.6f}, Non-zero weights: {non_zero_final}")
        
        return self


class EPDCAeLogisticRegression(BaseLogisticRegression):
    """EPDCAe algorithm logistic regression"""
    
    def __init__(self, lambda_=1.0, t=1.0, max_iter=1000, tol=1e-6, alpha=None, verbose=False):
        super().__init__(lambda_, t, max_iter, tol, alpha, verbose)
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        if len(y) != n_samples:
            raise ValueError(f"Input shape mismatch: X {X.shape}, y {y.shape}")
        
        if self.alpha is None:
            self.alpha = self._compute_lipschitz_constant(X)
            if self.verbose:
                print(f"Computed Lipschitz constant: {self.alpha:.6f}")
        
        epsilon = self.tol
        eta = 1e-4
        theta = np.zeros(200)
        beta_v = np.zeros(200)
        theta[0] = (1 + np.sqrt(5)) / 2
        beta_v[0] = 0
        
        for i in range(1, 200):
            theta[i] = (1 + np.sqrt(1 + 4 * theta[i-1]**2)) / 2
            beta_v[i] = 1 * (theta[i-1] - 1) / theta[i]

        c = 1
        x = np.zeros(n_features)
        bx = x.copy()
        s = self.lambda_ / self.t
        
        hzt = np.sum(np.maximum(x - self.t, 0) + np.maximum(0, -x - self.t))
        FO_rho = self._logistic_loss(X, y, x) + s * (np.linalg.norm(x, 1) - hzt)
        gf = self._logistic_gradient(X, y, bx)
        
        self.loss_history = [FO_rho]
        self.time_history = [0.0]
        start_time = time.time()
        
        for iter in range(self.max_iter):
            current_time = time.time() - start_time
            self.time_history.append(current_time)
            
            I = np.zeros((n_features, 3))
            dxt = np.maximum(x - self.t, 0) + np.maximum(0, -x - self.t) - eta
            I[x - self.t >= dxt, 0] = 1
            I[0 >= dxt, 1] = 1
            I[-x - self.t >= dxt, 2] = 1
            Is = np.sum(I, axis=1)
            num = np.sum(Is == 2)
            
            if num > 20:
                if self.verbose:
                    print(f"Stopped at iteration {iter} because num={num}>20")
                break
            
            Num = 2**num
            Gx = np.column_stack((np.ones(n_features), np.zeros(n_features), -np.ones(n_features)))
            nzg = np.sum(I * Gx, axis=1)
            gx = np.zeros(n_features)
            gx[Is == 1] = nzg[Is == 1]
            
            funv = np.inf
            
            for i in range(Num):
                i22 = np.array([int(b) for b in bin(i)[2:].zfill(num)], dtype=int)
                gx[Is == 2] = np.sign(x[Is == 2]) * i22
                z = np.maximum(s * gx - gf + self.alpha * x + c * bx - s, 
                              np.minimum(0, s * gx - gf + self.alpha * x + c * bx + s)) / (self.alpha + c)
                hzt = np.sum(np.maximum(z - self.t, 0) + np.maximum(0, -z - self.t))
                F_rho = self._logistic_loss(X, y, z) + s * (np.linalg.norm(z, 1) - hzt) + (z - x).T @ (z - x) / 2
                
                if F_rho < funv:
                    x_new = z
                    funv = F_rho
            
            hzt = np.sum(np.maximum(x_new - self.t, 0) + np.maximum(0, -x_new - self.t))
            F_rho = self._logistic_loss(X, y, x_new) + s * (np.linalg.norm(x_new, 1) - hzt)
            self.loss_history.append(F_rho)
            
            if abs(FO_rho - F_rho) < epsilon:
                if self.verbose:
                    print(f"Converged at iteration {iter}")
                break
            
            FO_rho = F_rho
            
            if iter % 200 == 0:
                beta = 0
            else:
                remm = iter % 200
                beta = beta_v[remm]
            
            bx = x_new + beta * (x_new - x)
            x = x_new
            gf = self._logistic_gradient(X, y, bx)
            
            if self.verbose and iter % 100 == 0:
                non_zero = np.sum(np.abs(x) > 1e-4)
                print(f"Iter {iter:4d}, Loss: {F_rho:.6f}, Non-zero: {non_zero}")
        
        self.weights = x
        if self.verbose:
            non_zero_final, _ = self.get_nonzero_weights()
            final_loss = self._objective_function(X, y, self.weights)
            print(f"Training completed. Final loss: {final_loss:.6f}, Non-zero weights: {non_zero_final}")
        
        return self


class NEPDCALogisticRegression(BaseLogisticRegression):
    """NEPDCA algorithm logistic regression"""
    
    def __init__(self, lambda_=1.0, t=1.0, max_iter=1000, tol=1e-6, alpha=None, verbose=False):
        super().__init__(lambda_, t, max_iter, tol, alpha, verbose)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        if len(y) != n_samples:
            raise ValueError(f"Input shape mismatch: X {X.shape}, y {y.shape}")
        
        if self.alpha is None:
            self.alpha = self._compute_lipschitz_constant(X)
            if self.verbose:
                print(f"Computed Lipschitz constant: {self.alpha:.6f}")
        
        epsilon = self.tol
        eta = 1e-4
        x = np.zeros(n_features)
        s = self.lambda_ / self.t
        
        hzt = np.sum(np.maximum(x - self.t, 0) + np.maximum(0, -x - self.t))
        FO_rho = self._logistic_loss(X, y, x) + s * (np.linalg.norm(x, 1) - hzt)
        
        M = 5
        Fv = np.zeros(M)
        gamma = 1e-4
        alpha_min = 1e-8
        alpha_max = 1e8
        gf = self._logistic_gradient(X, y, x)
        
        self.loss_history = [FO_rho]
        self.time_history = [0.0]
        start_time = time.time()
        
        for iter in range(self.max_iter):
            current_time = time.time() - start_time
            self.time_history.append(current_time)
            
            I = np.zeros((n_features, 3))
            dxt = np.maximum(x - self.t, 0) + np.maximum(0, -x - self.t) - eta
            I[x - self.t >= dxt, 0] = 1
            I[0 >= dxt, 1] = 1
            I[-x - self.t >= dxt, 2] = 1
            Is = np.sum(I, axis=1)
            num = np.sum(Is == 2)
            
            if num > 20:
                if self.verbose:
                    print(f"Stopped at iteration {iter} because num={num}>20")
                break
            
            Num = 2**num
            Gx = np.column_stack((np.ones(n_features), np.zeros(n_features), -np.ones(n_features)))
            nzg = np.sum(I * Gx, axis=1)
            gx = np.zeros(n_features)
            gx[Is == 1] = nzg[Is == 1]

            remm = iter % M
            Fv[remm] = FO_rho

            while True:
                funv = np.inf
                II = np.zeros((Num, 2))
                
                II[:, 0] = self._logistic_loss(X, y, x) + s * np.linalg.norm(x, 1)
                for i in range(Num):
                    i22 = np.array([int(b) for b in bin(i)[2:].zfill(num)], dtype=int)
                    gx[Is == 2] = np.sign(x[Is == 2]) * i22
                    z = np.maximum(s * gx - gf + self.alpha * x - s, 
                                  np.minimum(0, s * gx - gf + self.alpha * x + s)) / self.alpha
                    II[i, 1] = np.linalg.norm(z - x)**2
                    hzt = np.sum(np.maximum(z - self.t, 0) + np.maximum(0, -z - self.t))
                    F_rho = self._logistic_loss(X, y, z) + s * (np.linalg.norm(z, 1) - hzt) + gamma / 2 * II[i, 1]
                    II[i, 0] = II[i, 0] - s * np.sum(gx * x - gx**2 * self.t)
                    
                    if F_rho < funv:
                        x_new = z
                        funv = F_rho
                
                Mum = 0
                hzt = np.sum(np.maximum(x_new - self.t, 0) + np.maximum(0, -x_new - self.t))
                F_rho = self._logistic_loss(X, y, x_new) + s * (np.linalg.norm(x_new, 1) - hzt)
                for i in range(Num):
                    if F_rho <= max(np.max(Fv), II[i, 0]) - gamma * np.linalg.norm(x_new - x)**2 / 2 - gamma * II[i, 1] / 2:
                        Mum += 1
                if Mum == Num:
                    break
                else:
                    self.alpha *= 2
            
            self.loss_history.append(F_rho)
            TERM = FO_rho - F_rho
            
            if abs(TERM) < epsilon:
                if self.verbose:
                    print(f"Converged at iteration {iter}")
                break

            gfn = self._logistic_gradient(X, y, x_new)
            if np.linalg.norm(x - x_new) == 0:
                self.alpha = alpha_max
            else:
                self.alpha = abs((gfn - gf).T @ (x_new - x)) / np.linalg.norm(x_new - x)**2
                self.alpha = max(alpha_min, min(self.alpha, alpha_max))

            gf = gfn
            FO_rho = F_rho
            x = x_new
            
            if self.verbose and iter % 100 == 0:
                non_zero = np.sum(np.abs(x) > 1e-4)
                print(f"Iter {iter:4d}, Loss: {F_rho:.6f}, Non-zero: {non_zero}")

        self.weights = x
        if self.verbose:
            non_zero_final, _ = self.get_nonzero_weights()
            final_loss = self._objective_function(X, y, self.weights)
            print(f"Training completed. Final loss: {final_loss:.6f}, Non-zero weights: {non_zero_final}")
        
        return self


class APGLogisticRegression(BaseLogisticRegression):
    """APG (Accelerated Proximal Gradient) algorithm logistic regression"""
    
    def __init__(self, lambda_=1.0, t=1.0, max_iter=5000, tol=1e-6, alpha=None, verbose=False):
        super().__init__(lambda_, t, max_iter, tol, alpha, verbose)
    
    def _soft_threshold(self, x, threshold):
        """Soft thresholding function"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        if len(y) != n_samples:
            raise ValueError(f"Input shape mismatch: X {X.shape}, y {y.shape}")
        
        if self.alpha is None:
            self.alpha = self._compute_lipschitz_constant(X)
            if self.verbose:
                print(f"Computed Lipschitz constant: {self.alpha:.6f}")
        
        # Initialize parameters
        x = np.zeros(n_features)
        x_prev = x.copy()
        s = self.lambda_ / self.t
        
        # Record history
        self.loss_history = []
        self.time_history = []
        
        # APG parameters
        gamma = 0.99 / self.alpha  # Step size
        start_time = time.time()
        if self.verbose:
            print(f"Using step size: {gamma:.6f}")
        
        for iter in range(self.max_iter):
            current_time = time.time() - start_time
            self.time_history.append(current_time)
            
            # Compute current objective function value
            hzt = np.sum(np.maximum(x - self.t, 0) + np.maximum(0, -x - self.t))
            current_loss = self._logistic_loss(X, y, x) + s * (np.linalg.norm(x, 1) - hzt)
            self.loss_history.append(current_loss)
            
            # Compute d vector (subgradient of capped L1)
            d = np.zeros(n_features)
            d[(x >= self.t)] = 1
            d[(x <= -self.t)] = -1
            
            # APG update step
            beta = (iter) / (iter + 3)  # Nesterov acceleration parameter
            
            # Extrapolation point
            y_k = x + beta * (x - x_prev)
            
            # Compute gradient at extrapolation point
            grad_y = self._logistic_gradient(X, y, y_k)
            
            # Update parameters
            w = y_k - gamma * grad_y
            w = w + gamma * s * d  # Add capped L1 term
            
            # Soft thresholding operation
            x_new = self._soft_threshold(w, gamma * s)
            
            # Update variables
            x_prev = x.copy()
            x = x_new
            
            # Convergence check
            if iter > 0 and abs(self.loss_history[-2] - current_loss) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iter}")
                break
            
            if self.verbose and iter % 100 == 0:
                non_zero = np.sum(np.abs(x) > 1e-4)
                print(f"Iter {iter:4d}, Loss: {current_loss:.6f}, Non-zero: {non_zero}")
        
        self.weights = x
        if self.verbose:
            non_zero_final, _ = self.get_nonzero_weights()
            final_loss = self._objective_function(X, y, self.weights)
            print(f"Training completed. Final loss: {final_loss:.6f}, Non-zero weights: {non_zero_final}")
        
        return self


"""
NL0R -> NL0RLogisticRegression
Converted to sparse logistic regression with L0 iterative update (sklearn-like API).
Author: converted by ChatGPT
Date: 2025-10-30
"""

import numpy as np
import time
from typing import Optional, Dict, Tuple, Union, Callable, Any

try:
    import scipy.sparse as sp
except Exception:
    sp = None  # allow running without scipy sparse

Array = np.ndarray
DataDict = Dict[str, object]


class NL0RLogisticRegression:
    """
    NL0R style sparse logistic regression solver (L0 with iterative lambda update).
    Keeps NL0R algorithm structure but replaces least-squares objective with
    logistic loss (average negative log-likelihood).
    Supports dense and scipy.sparse feature matrices.

    Parameters
    ----------
    lambda_init : float
        Initial lambda for L0 regularization.
    tau : float
        Initial tau (prox step parameter).
    rate : float
        Rate used in lambda update heuristics.
    max_iter : int
        Maximum number of outer iterations.
    tol : float
        Tolerance used in stopping conditions.
    disp : bool
        If True, prints progress.
    fit_intercept : bool
        If True, fit intercept term (adds a column of ones to X).
    random_state : Optional[int]
        For any random fallback.
    """

    def __init__(self,
                 lambda_init: float = 0.1,
                 tau: Optional[float] = None,
                 rate: float = 0.5,
                 max_iter: int = 2000,
                 tol: float = 1e-8,
                 disp: bool = True,
                 fit_intercept: bool = False,
                 random_state: Optional[int] = None):
        self.lambda_init = float(lambda_init)
        self.tau = tau  # if None, will be set according to n in fit
        self.rate = float(rate)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.disp = bool(disp)
        self.fit_intercept = bool(fit_intercept)
        self.random_state = random_state

        # attributes set after fit
        self.coef_: Optional[Array] = None
        self.n_iter_: int = 0
        self.loss_history_: list = []
        self.time_history_: list = []
        self.sparsity_history_: list = []

    # ---------------- Fixed numerical stability methods ----------------
    @staticmethod
    def _sigmoid(z: Array) -> Array:
        """More stable sigmoid function to prevent numerical overflow"""
        z = np.asarray(z, dtype=np.float64)
        out = np.empty_like(z)
        
        # Handle extreme cases separately
        pos = z > 20   # Large positive numbers directly set to 1
        neg = z < -20  # Large negative numbers directly set to 0
        mid = ~(pos | neg)
        
        out[pos] = 1.0
        out[neg] = 0.0
        # For middle values use standard sigmoid
        out[mid] = 1.0 / (1.0 + np.exp(-z[mid]))
        
        return out

    @staticmethod
    def _log1pexp_stable(x: Array) -> Array:
        """Stable computation of log(1 + exp(x)) to prevent numerical overflow"""
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        
        # Handle different ranges separately
        mask_large_pos = x > 50   # exp(x) will overflow, approximate as x
        mask_large_neg = x < -50  # exp(x) close to 0, approximate as 0
        mask_mid = ~(mask_large_pos | mask_large_neg)
        
        out[mask_large_pos] = x[mask_large_pos]  # log(1+exp(x)) ≈ x when x is large
        out[mask_large_neg] = 0.0                # log(1+exp(x)) ≈ 0 when x is very small
        out[mask_mid] = np.logaddexp(0.0, x[mask_mid])  # Standard stable computation
        
        return out

    # ---------------- objective / gradient / Hessian blocks for logistic ----------------
    def _logistic_fgH(self, x: Array, T1: Optional[Array], T2: Optional[Array], data: DataDict) -> Tuple[Any, Optional[Any]]:
        """
        Fixed logistic regression objective function, gradient and Hessian computation
        Ensures objective function is always non-negative
        """
        A = data['A']
        b = np.asarray(data['b'], dtype=np.float64).ravel()
        x = np.asarray(x, dtype=np.float64).ravel()

        # If intercept term is included, A already contains it
        if not callable(A):
            is_sparse = (sp is not None and sp.issparse(A))
            m = A.shape[0]
            
            # Compute Ax
            if is_sparse:
                Ax = A.dot(x)
            else:
                Ax = A @ x
            
            # Use stable sigmoid computation
            s = self._sigmoid(Ax)
            
            # Use stable log(1+exp(Ax)) computation
            log1pexp_Ax = self._log1pexp_stable(Ax)
            
            # Compute objective function: f = (1/m) * sum(log(1+exp(Ax)) - b*Ax)
            f = (np.sum(log1pexp_Ax) - np.dot(b, Ax)) / float(m)
            
            # Ensure objective function is non-negative (handle numerical errors)
            f = max(float(f), 0.0)
            
            if T1 is None and T2 is None:
                # Compute gradient: g = (1/m) * A^T (sigmoid(Ax) - b)
                resid = s - b  # shape (m,)
                if is_sparse:
                    g = A.T.dot(resid) / float(m)
                    if sp.issparse(g):
                        g = np.asarray(g.toarray()).ravel()
                else:
                    g = (A.T @ resid) / float(m)
                return f, g
            
            # Compute Hessian related parts
            S = s * (1.0 - s)  # shape (m,)
            # Extract columns
            AT1 = A[:, T1] if not is_sparse else A[:, T1]
            
            if T2 is None:
                # Return H(T1,T1)
                if AT1.shape[1] <= 1000:
                    # Small matrix direct computation
                    if is_sparse:
                        Xw = AT1.multiply(S.reshape(-1, 1))
                        H = (Xw.T @ AT1) / float(m)
                        if sp.issparse(H):
                            H = H.toarray()
                    else:
                        H = (AT1.T * S) @ AT1 / float(m)
                else:
                    # Large matrix return linear operator
                    if is_sparse:
                        H = (lambda v, AT1=AT1, S=S, m=m: (AT1.T @ (S * (AT1 @ v))) / float(m))
                    else:
                        H = (lambda v, AT1=AT1, S=S, m=m: AT1.T @ (S * (AT1 @ v)) / float(m))
                return H, None
            else:
                AT2 = A[:, T2] if not is_sparse else A[:, T2]
                if (AT1.shape[1] <= 1000) and (AT2.shape[1] <= 1000):
                    if is_sparse:
                        Xw = AT1.multiply(S.reshape(-1, 1))
                        D = (Xw.T @ AT2) / float(m)
                        if sp.issparse(D):
                            D = D.toarray()
                    else:
                        D = (AT1.T * S) @ AT2 / float(m)
                else:
                    D = (lambda v, AT1=AT1, AT2=AT2, S=S, m=m: AT1.T @ (S * (AT2 @ v)) / float(m))
                
                # Also return H(T1,T1)
                if AT1.shape[1] <= 1000:
                    if is_sparse:
                        Xw = AT1.multiply(S.reshape(-1, 1))
                        H = (Xw.T @ AT1) / float(m)
                        if sp.issparse(H):
                            H = H.toarray()
                    else:
                        H = (AT1.T * S) @ AT1 / float(m)
                else:
                    H = (lambda v, AT1=AT1, S=S, m=m: AT1.T @ (S * (AT1 @ v)) / float(m))
                return H, D

        # Operator form not implemented in this version
        raise ValueError("Only matrix-form data with 'A' (ndarray or scipy.sparse) is supported.")

    # ---------------- helpers reused from original ----------------
    @staticmethod
    def _sparse_approx(xT: Array, T: Array) -> Array:
        xabs = np.abs(np.asarray(xT, dtype=np.float64)).ravel()
        nz = xabs[xabs != 0.0]
        if nz.size == 0:
            return np.array([], dtype=np.int64)
        sx = np.sort(nz)  # ascending
        if sx.size <= 2:
            th = sx[-1]
        else:
            ratio = sx[1:] / (sx[:-1] + 1e-16)
            nr = np.linalg.norm(ratio)
            ratio_n = ratio / (nr if nr > 0 else 1.0)
            itmax = int(np.argmax(ratio_n))
            mx = ratio_n[itmax]
            if (mx > 10.0) and (itmax + 1 > 1):
                th = sx[itmax + 1]
            else:
                th = 0.0
        take = np.abs(xT).ravel() > th
        return T[take]

    @staticmethod
    def _as_linear_operator(H: Union[Array, Callable[[Array], Array]]) -> Callable[[Array], Array]:
        if callable(H):
            return H
        mat = np.asarray(H, dtype=np.float64)
        return lambda v: mat @ v

    def _solve_on_support(self, H: Union[Array, Callable[[Array], Array]], rhs: Array, cgtol: float, cgit: int) -> Array:
        """Solve H d = rhs using direct solve if ndarray small, otherwise CG."""
        if callable(H):
            return self._my_cg(H, rhs, cgtol, cgit, np.zeros_like(rhs))
        else:
            H = np.asarray(H, dtype=np.float64)
            s = H.shape[0]
            if s == 0:
                return np.array([], dtype=np.float64)
            if s <= 1000:
                try:
                    return np.linalg.solve(H, rhs)
                except np.linalg.LinAlgError:
                    return np.linalg.lstsq(H, rhs, rcond=None)[0]
            else:
                Hop = lambda v: H @ v
                return self._my_cg(Hop, rhs, cgtol, cgit, np.zeros_like(rhs))

    def _apply_D(self, D: Union[Array, Callable[[Array], Array]], v: Array) -> Array:
        if callable(D):
            return D(v)
        else:
            return np.asarray(D, dtype=np.float64) @ v

    def _my_cg(self, fx: Union[Array, Callable[[Array], Array]],
               b: Array, cgtol: float, cgit: int, x0: Array) -> Array:
        op = self._as_linear_operator(fx)
        x = x0.copy()
        r = b - op(x) if np.count_nonzero(x) else b.copy()
        e = float(np.dot(r, r))
        t = e
        p = r.copy()
        for _ in range(cgit):
            if e < cgtol * t:
                break
            w = op(p)
            pw = float(np.dot(p, w))
            if pw == 0.0:
                break
            a = e / pw
            x += a * p
            r -= a * w
            e0 = e
            e = float(np.dot(r, r))
            if e0 == 0.0:
                break
            p = r + (e / e0) * p
        return x

    # ---------------- fit / predict / helper I/O ----------------
    def fit(self, X: Array, y: Array, pars: Optional[Dict[str, object]] = None):
        """
        Fit model to data (X, y).
        X: (m, n) dense ndarray or scipy.sparse matrix
        y: (m,) array with values 0 or 1
        pars: optional dictionary to override internal algorithm params:
            x0, tau, rate, disp (int), maxit, tol, obj (pobj)
        """
        if sp is not None and sp.issparse(X):
            is_sparse = True
        else:
            is_sparse = False
            X = np.asarray(X, dtype=np.float64)

        y = np.asarray(y, dtype=np.float64).ravel()
        if y.ndim != 1:
            raise ValueError("y must be a 1-D array")

        m, n_orig = X.shape
        # Handle intercept term
        if self.fit_intercept:
            # Add column of ones as first column
            if is_sparse:
                X = sp.hstack([sp.csr_matrix(np.ones((m, 1))), X], format='csr')
            else:
                X = np.hstack([np.ones((m, 1), dtype=np.float64), X])
        n = X.shape[1]

        pars = pars or {}
        rate0 = 0.5 if n <= 1000 else 1.0 / np.exp(3.0 / np.log10(n))
        tau0 = 1.0 if n <= 1000 else 0.5

        x = np.asarray(pars.get('x0', np.zeros(n, dtype=np.float64)), dtype=np.float64).ravel()
        if x.size != n:
            raise ValueError("x0 size mismatch")

        tau = float(pars.get('tau', self.tau if self.tau is not None else tau0))
        rate = float(pars.get('rate', self.rate if self.rate is not None else rate0))
        disp = int(pars.get('disp', 1 if self.disp else 0))
        itmax = int(pars.get('maxit', self.max_iter))
        pobj = float(pars.get('obj', 1e-20))
        tol = float(pars.get('tol', self.tol))

        Err = np.zeros(itmax, dtype=np.float64)
        Obj = np.zeros(itmax, dtype=np.float64)
        Nzx = np.zeros(itmax, dtype=np.int64)
        FNorm = lambda v: float(np.dot(v, v))

        # Prepare data
        data = {'A': X, 'b': y}

        if disp:
            print("\n Start to run the solver -- NL0RLogisticRegression")
            print(" -------------------------------------")
            print(" Iter     ObjVal    Sparsity     Time ")
            print(" -------------------------------------")

        # Initial objective function and gradient
        obj, g = self._logistic_fgH(x, None, None, data)
        g = np.asarray(g, dtype=np.float64).ravel()
        if FNorm(g) == 0.0:
            if disp:
                print("Starting point is a good stationary point, stop !!!")
            self.coef_ = x.copy()
            self.n_iter_ = 0
            self.loss_history_ = [float(obj)]
            self.time_history_ = [0.0]
            return self

        maxlam = (np.max(np.abs(g)) ** 2) * tau / 2.0

        # Handle NaN in gradient
        if np.isnan(g).any():
            x.fill(0.0)
            if self.random_state is not None:
                np.random.seed(self.random_state)
            x[np.random.randint(0, n)] = np.random.rand()
            obj, g = self._logistic_fgH(x, None, None, data)
            g = np.asarray(g, dtype=np.float64).ravel()

        pcgit = 5
        pcgtol = 1e-5
        beta = 0.5
        sigma = 5e-5
        delta = 1e-10
        T0 = np.array([], dtype=np.int64)
        nx = 0

        t0 = time.time()

        for it in range(itmax):
            x0 = x.copy()
            xtg = x0 - tau * g
            # Threshold support set
            T = np.flatnonzero(np.abs(xtg) > np.sqrt(2.0 * tau * self.lambda_init))
            nT = T.size

            # Heuristic support set optimization
            if nT > max(0.12, 0.2 / np.log2(1 + (it + 1))) * n:
                Tnew = self._sparse_approx(xtg[T], T)
                nTnew = Tnew.size
                if nTnew > 0 and (nT / nTnew) < 20:
                    T = Tnew
                    nT = nTnew

            # TTc = T0 \cap complement(T)
            TTc = np.setdiff1d(T0, T, assume_unique=False)
            flag_same_support = (TTc.size == 0)

            # Stopping condition indicators
            FxT = np.sqrt(FNorm(g[T]) + (FNorm(x[TTc]) if TTc.size else 0.0))
            Err[it] = FxT / np.sqrt(n)
            Nzx[it] = nx

            if disp:
                # Ensure objective function is non-negative when displaying
                display_obj = max(obj, 0.0)
                print(f"{it+1:4d}     {display_obj:5.2e}    {nT:4d}    {time.time()-t0:6.2f}sec")

            # Stopping conditions (same as original logic)
            stop0 = (it > 0 and abs(obj - Obj[it-1]) < 1e-6 * (1.0 + obj))
            stop1 = (Err[it] < tol and nx == nT and stop0 and flag_same_support)
            stop2 = (it > 3 and obj < pobj and nx <= np.ceil(n / 4))
            if it >= 9:
                e = Err[it-9:it+1]
                o = Obj[it-9:it+1]
                k = Nzx[it-9:it+1]
                stop3 = (
                    np.std(k) <= 0 and
                    (np.std(e) ** 2) <= np.min(e) and
                    (np.std(o) ** 2) <= np.min(o[:-1])
                )
            else:
                stop3 = False
            stop4 = (np.linalg.norm(g) < tol and nx <= np.ceil(n / 4))
            if stop1 or stop2 or stop3 or stop4:
                self.n_iter_ = it
                break

            # Update direction on support set
            if it == 0 or flag_same_support:
                H = self._logistic_fgH(x0, T, None, data)[0]
                d = self._solve_on_support(H, -g[T], pcgtol, pcgit)
                dg = float(np.dot(d, g[T]))
                ngT = FNorm(g[T])
                if (dg > max(-delta * FNorm(d), -ngT)) or np.isnan(dg):
                    d = -g[T].copy()
                    dg = ngT
            else:
                H, D = self._logistic_fgH(x0, T, TTc, data)
                rhs = self._apply_D(D, x0[TTc]) - g[T]
                d = self._solve_on_support(H, rhs, pcgtol, pcgit)

                Fnz = FNorm(x[TTc]) / (4.0 * tau) if TTc.size else 0.0
                dgT = float(np.dot(d, g[T]))
                dg = dgT - float(np.dot(x0[TTc], g[TTc])) if TTc.size else dgT

                delta0 = 1e-4 if (Fnz > 1e-4) else delta
                ngT = FNorm(g[T])
                if (dgT > max(-delta0 * FNorm(d) + Fnz, -ngT)) or np.isnan(dg):
                    d = -g[T].copy()
                    dg = ngT

            # Armijo backtracking line search
            alpha = 1.0
            obj0 = obj
            x_trial = np.zeros_like(x0)
            for _ls in range(6):
                x_trial[:] = 0.0
                x_trial[T] = x0[T] + alpha * d
                obj_trial, _ = self._logistic_fgH(x_trial, None, None, data)
                if obj_trial < obj0 + alpha * sigma * dg:
                    x = x_trial
                    obj = float(obj_trial)
                    break
                alpha *= beta
            else:
                # If no acceptable step size, use gradient descent fallback
                x = x0 - tau * g
                obj, _ = self._logistic_fgH(x, None, None, data)

            T0 = T.copy()
            obj, g = self._logistic_fgH(x, None, None, data)
            g = np.asarray(g, dtype=np.float64).ravel()
            Obj[it] = float(obj)

            # Periodic tau update (heuristic)
            if (it + 1) % 10 == 0:
                OBJ = Obj[max(0, it-9):it+1]
                if (Err[it] > 1.0 / (it + 1) ** 2) or (np.count_nonzero(OBJ[1:] > 1.5 * OBJ[:-1]) >= 2):
                    tau = tau / (1.25 if (it + 1) < 1500 else 1.5)
                else:
                    tau = tau * 1.25

            # Update lambda (iterative rule)
            nx = int(np.count_nonzero(x))
            if (it + 1) > 5 and (nx > 2 * int(np.max(Nzx[:it+1-1]) if it >= 1 else 0)) and (Err[it] < 1e-2):
                rate0 = 2.0 / rate
                x = x0.copy()  # Fallback
                nx = int(np.count_nonzero(x0))
                nx0 = Nzx[it-1]
                obj, g = self._logistic_fgH(x, None, None, data)
                g = np.asarray(g, dtype=np.float64).ravel()
                rate = 1.1
            else:
                rate0 = rate

            if 'nx0' in locals() and nx < nx0:
                rate0 = 1.0

            # Update lambda
            if it == 0:
                lam = self.lambda_init
            lam = min(maxlam, lam * (2.0 * (nx >= 0.1 * n) + rate0))

            # Record history
            self.loss_history_.append(float(obj))
            self.time_history_.append(time.time() - t0)
            self.sparsity_history_.append(nx)

        # Final processing
        self.coef_ = x.copy()
        self.n_iter_ = it if self.n_iter_ == 0 else self.n_iter_

        if self.disp:
            non_zero_final = int(np.count_nonzero(self.coef_))
            final_obj, _ = self._logistic_fgH(self.coef_, None, None, data)
            final_obj = max(final_obj, 0.0)  # Ensure final objective function is non-negative
            print("Finished. Iter: %d, Obj: %.6e, Nonzero: %d, Time: %.3f s" %
                  (self.n_iter_, float(final_obj), non_zero_final, time.time() - t0))

        return self

    def predict_proba(self, X: Array) -> Array:
        if self.coef_ is None:
            raise ValueError("Model not fitted yet. Call .fit(X, y) first.")
        if self.fit_intercept:
            # If training included intercept, no need for extra processing in prediction
            # because coefficients already include intercept term
            if sp is not None and sp.issparse(X):
                z = X.dot(self.coef_[1:]) + self.coef_[0]
            else:
                z = np.asarray(X) @ self.coef_[1:] + self.coef_[0]
        else:
            if sp is not None and sp.issparse(X):
                z = X.dot(self.coef_)
            else:
                z = np.asarray(X) @ self.coef_
        p = self._sigmoid(z)
        return np.column_stack([1 - p, p])

    def predict(self, X: Array, threshold: float = 0.5) -> Array:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def score(self, X: Array, y: Array) -> float:
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    # Convenience properties
    @property
    def coef(self) -> Array:
        return self.coef_

    def get_nonzero_weights(self, threshold: float = 1e-4) -> Tuple[int, Array]:
        """
        Get number and indices of non-zero coefficients

        Parameters
        ----------
        threshold : float
            Threshold for absolute value of coefficients

        Returns
        -------
        n_nonzero : int
            Number of non-zero coefficients
        indices : ndarray
            Indices of non-zero coefficients
        """
        if self.coef_ is None:
            return 0, np.array([], dtype=int)
        non_zero_indices = np.where(np.abs(self.coef_) > threshold)[0]
        n_nonzero = len(non_zero_indices)
        return n_nonzero, non_zero_indices

    def get_sparsity(self, threshold: float = 1e-4) -> float:
        """
        Get model sparsity

        Parameters
        ----------
        threshold : float
            Threshold for absolute value of coefficients

        Returns
        -------
        sparsity : float
            Proportion of non-zero coefficients
        """
        if self.coef_ is None:
            return 0.0
        n_nonzero = np.sum(np.abs(self.coef_) > threshold)
        sparsity = n_nonzero / len(self.coef_)
        return sparsity