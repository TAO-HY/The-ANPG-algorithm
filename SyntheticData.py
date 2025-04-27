import numpy as np
from scipy.linalg import orth
import time
import copy
import matplotlib.pyplot as plt
####################################定义funv_succ########################################################################
def funv_succ( x, xs):
    
    suc = np.linalg.norm(x - xs) / np.linalg.norm(xs) < 1e-2
    return  suc
####################################定义ASPG########################################################################
def ANPG(A, b, lambda_, t, x0):
    epsilon = 1e-6  # stopping criterion
    x = x0
   
    #t=np.maximum(t*0.99,1e-8)
    n = len(x)
    s = lambda_ / t
    hzt = np.sum(np.maximum(x - t, 0) + np.maximum(0, -x - t))
    FO_rho = np.linalg.norm(A @ x - b)**2 / 2 + s * (np.linalg.norm(x, 1) - hzt)
    #alpha = 1
    x_old=copy.deepcopy(x)
    t1=1
    
    for iter in range(1000):
        #每5次迭代减小t
        if iter%5==0:
            t=np.maximum(t*0.999,0.0002)
        s = lambda_ / t
        II= np.zeros(n)
        #x_i>=nu时d_i=2，x_i<=-nu时II=3，其余II=1
        II=(x>=t)*1+(x<=-t)*(-1)
        t2=(np.sqrt(t1**2+2*t1))
        beta=(t1-1)/t2
        #z=x_0+beta*(x_0-x_1)
        z=x+beta*(x-x_old)*np.abs(II)
        gf = A.T @ (A @ z - b)
        w=z-gf
        w=w+s*II
        x_old=copy.deepcopy(x)
        x=np.sign(w)*np.maximum(abs(w)-s,0)
        t1=copy.deepcopy(t2)
        #如果||F(x)-F(x_old)||<epsilon，停止迭代
        hzt = np.sum(np.maximum(x-t, 0) + np.maximum(0, -x-t))
        F_rho = np.linalg.norm(A @ x - b)**2 / 2 + s * (np.linalg.norm(x, 1) - hzt)
        TERM = FO_rho - F_rho
        if np.abs(TERM) < epsilon:
            break
        #更新FO_rho
        FO_rho = F_rho
    return x
    
####################################定义HA########################################################################
def HA(A, b, lambda_, t, x0):
    epsilon = 1e-4
    eta = 1.0e-1
    x = x0
    n = len(x)
    s = lambda_ / t
    x_new = x
    alpha = 1
    hzt = np.sum(np.maximum(x - t, 0) + np.maximum(0, -x - t))
    FO_rho = np.linalg.norm(A @ x - b)**2 / 2 + s * (np.linalg.norm(x, 1) - hzt)

    for iter in range(2000):
        #if iter % 5 == 0:
            #t = np.maximum(t * 0.999, 0.0002)
        #s = lambda_ / t
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
                gx[Is == 2] = np.sign(x[Is == 2]) * i22
                gf = A.T @ (A @ x - b)
                z = np.maximum(s * gx - gf + alpha * x - s, np.minimum(0, s * gx - gf + alpha * x + s)) / alpha
                hzt = np.sum(np.maximum(z - t, 0) + np.maximum(0, -z - t))
                F_rho = np.linalg.norm(A @ z - b)**2 / 2 + s * (np.linalg.norm(z, 1) - hzt)
                TERM = FO_rho - F_rho

                if TERM >= epsilon:
                    x_new = z
                    break
                
                l_rho = (gf - s * gx).T @ (z - x) + 1/2 * np.linalg.norm(z - x)**2 + s * np.linalg.norm(z, 1) - s * np.sum(gx * z - gx**2 * t)
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
        x = x_new
    
    return x

####################################定义EPDCAe########################################################################
def EPDCAe(A, b, lambd, t, x0):
    # Parameters initialization
    epsilon = 1e-6
    eta = 1e-4
    theta = np.zeros(200)
    beta_v = np.zeros(200)
    theta[0] = (1 + np.sqrt(5)) / 2
    beta_v[0] = 0
    for i in range(1, 200):
        theta[i] = (1 + np.sqrt(1 + 4 * theta[i-1]**2)) / 2
        beta_v[i] = 1 * (theta[i-1] - 1) / theta[i]
    
    alpha = 1
    c = 1
    x = x0
    bx = x0.copy()
    n = len(x)
    s = lambd/ t
    
    hzt = np.sum(np.maximum(x - t, 0) + np.maximum(0, -x - t))
    FO_rho = np.linalg.norm(A @ x - b)**2 / 2 + s * (np.linalg.norm(x, 1) - hzt)
    gf = A.T @ (A @ bx - b)
    
    for iter in range(1000):
        #if iter % 5 == 0:
            #t = np.maximum(t * 0.999, 0.0002)
        #s = lambd / t
        I = np.zeros((n, 3))
        dxt = np.maximum(x - t, 0) + np.maximum(0, -x - t) - eta
        I[x - t >= dxt, 0] = 1
        I[0 >= dxt, 1] = 1
        I[-x - t >= dxt, 2] = 1
        Is = np.sum(I, axis=1)
        num = np.sum(Is == 2)
        
        if num > 20:
            break
        
        Num = 2**num
        Gx = np.column_stack((np.ones(n), np.zeros(n), -np.ones(n)))
        nzg = np.sum(I * Gx, axis=1)
        gx = np.zeros(n)
        gx[Is == 1] = nzg[Is == 1]
        
        funv = np.inf
        
        for i in range(Num):
            i22 = np.array([int(b) for b in bin(i)[2:].zfill(num)], dtype=int)
            gx[Is == 2] = np.sign(x[Is == 2]) * i22
            z = np.maximum(s * gx - gf + alpha * x + c * bx - s, np.minimum(0, s * gx - gf + alpha * x + c * bx + s)) / (alpha + c)
            hzt = np.sum(np.maximum(z - t, 0) + np.maximum(0, -z - t))
            F_rho = np.linalg.norm(A @ z - b)**2 / 2 + s * (np.linalg.norm(z, 1) - hzt) + (z - x).T @ (z - x) / 2
            
            if F_rho < funv:
                x_new = z
                funv = F_rho
        
        hzt = np.sum(np.maximum(x_new - t, 0) + np.maximum(0, -x_new - t))
        F_rho = np.linalg.norm(A @ x_new - b)**2 / 2 + s * (np.linalg.norm(x_new, 1) - hzt)
        
        if abs(FO_rho - F_rho) < epsilon:
            break
        
        FO_rho = F_rho
        
        if iter % 200 == 0:
            beta = 0
        else:
            remm = iter % 200
            beta = beta_v[remm]
        
        bx = x_new + beta * (x_new - x)
        x = x_new
        gf = A.T @ (A @ bx - b)
    
    return x

####################################定义fnHPDCA########################################################################
def fnHPDCA(A, b, lambd, t, x0):
    epsilon = 1e-4
    eta = 1e-1
    x = x0
    n = len(x)
    s = lambd / t
    hzt = np.sum(np.maximum(x - t, 0) + np.maximum(0, -x - t))
    FO_rho = np.linalg.norm(A @ x - b)**2 / 2 + s * (np.linalg.norm(x, 1) - hzt)
    M = 5
    Fv = np.zeros(M)
    alpha = 1
    gamma = 1e-3
    alpha_min = 1e-8
    alpha_max = 1e8
    gf = A.T @ (A @ x - b)
    rho = 1

    for iter in range(1000):
        #if iter % 5 == 0:
            #t = np.maximum(t * 0.999, 0.0002)
        #s = lambd / t
        beta = alpha
        dxt = np.maximum(x - t, 0) + np.maximum(0, -x - t)
        II = np.zeros(n)
        II[x - t == dxt] = 1
        II[0 == dxt] = 0
        II[-x - t == dxt] = -1
        gx = II
        ii = iter % M
        Fv[ii] = FO_rho
        
        while True:
            z = np.maximum(s * gx - gf + alpha * x - s, np.minimum(0, s * gx - gf + alpha * x + s)) / alpha
            hzt = np.sum(np.maximum(z - t, 0) + np.maximum(0, -z - t))
            F_rho = np.linalg.norm(A @ z - b)**2 / 2 + s * (np.linalg.norm(z, 1) - hzt)
            if F_rho < max(Fv) - gamma * np.linalg.norm(z - x)**2:
                break
            alpha *= 2
            #如果alpha>alpha_max, print F_rho和max(Fv)然后braek
            
        
        
        TERM = max(Fv) - F_rho
        if TERM >= rho * epsilon:
            x_new = z
        else:
            I = np.zeros((n, 3))
            dxt = np.maximum(x - t, 0) + np.maximum(0, -x - t) - eta
            I[x - t >= dxt, 0] = 1
            I[0 >= dxt, 1] = 1
            I[-x - t >= dxt, 2] = 1
            Is = np.sum(I, axis=1)
            num = np.sum(Is == 2)
            Num = 2**num
            while True:
                lin_s = 1
                funv = np.inf
                for i in range(Num):
                    i22 = np.array([int(b) for b in bin(i)[2:].zfill(num)], dtype=int)
                    gx[Is == 2] = np.sign(x[Is == 2]) * i22
                    z = np.maximum(s * gx - gf + alpha * x - s, np.minimum(0, s * gx - gf + alpha * x + s)) / alpha
                    hzt = np.sum(np.maximum(z - t, 0) + np.maximum(0, -z - t))
                    F_rho = np.linalg.norm(A @ z - b)**2 / 2 + s * (np.linalg.norm(z, 1) - hzt)
                    if F_rho < max(Fv) - max(rho * epsilon, gamma * np.linalg.norm(z - x)**2) :
                        lin_s = 0
                        x_new = z
                        break
                    l_rho = (gf - s * gx).T @ (z - x) + 0.5 * np.linalg.norm(z - x)**2 + s * np.linalg.norm(z, 1) - s * np.sum(gx * z - gx**2 * t)
                    if l_rho < funv:
                        x_new = z
                        funv = l_rho
                
                if lin_s == 0:
                    break
                else:
                    hzt = np.sum(np.maximum(x_new - t, 0) + np.maximum(0, -x_new - t))
                    F_rho = np.linalg.norm(A @ x_new - b)**2 / 2 + s * (np.linalg.norm(x_new, 1) - hzt)
                    if F_rho < max(Fv) - gamma * np.linalg.norm(x_new - x)**2 or alpha > 1.1:
                        break
                    else:
                        alpha *= 2
                            
        TERM = max(Fv) - F_rho
        if TERM < rho * epsilon:
            if epsilon <= 1e-6:
                break
            epsilon *= 0.1
        
        gfn = A.T @ (A @ x_new - b)
        if np.linalg.norm(x - x_new) == 0:
            alpha = alpha_max
        else:
            alpha = abs((gfn - gf).T @ (x_new - x)) / np.linalg.norm(x_new - x)**2
            alpha = max(alpha_min, min(alpha, alpha_max))
        
        gf = gfn
        FO_rho = F_rho
        x = x_new
    
    return x
####################################定义NEPDCA########################################################################
def NEPDCA(A, b, lambd, t, x0):
    epsilon = 1e-6
    eta = 1e-4
    x = x0
    n = len(x)
    s = lambd / t
    hzt = np.sum(np.maximum(x - t, 0) + np.maximum(0, -x - t))
    FO_rho = np.linalg.norm(A @ x - b)**2 / 2 + s * (np.linalg.norm(x, 1) - hzt)
    M = 5
    Fv = np.zeros(M)
    alpha = 1
    gamma = 1e-4
    alpha_min = 1e-8
    alpha_max = 1e8
    gf = A.T @ (A @ x - b)
    for iter in range(1000):
        #if iter % 5 == 0:
            #t = np.maximum(t * 0.999, 0.0002)
        #s = lambd/ t
        I = np.zeros((n, 3))
        dxt = np.maximum(x - t, 0) + np.maximum(0, -x - t) - eta
        I[x - t >= dxt, 0] = 1
        I[0 >= dxt, 1] = 1
        I[-x - t >= dxt, 2] = 1
        Is = np.sum(I, axis=1)
        num = np.sum(Is == 2)
        if num > 20:
            break
        Num = 2**num
        Gx = np.column_stack((np.ones(n), np.zeros(n), -np.ones(n)))
        nzg = np.sum(I * Gx, axis=1)
        gx = np.zeros(n)
        gx[Is == 1] = nzg[Is == 1]

        remm = iter % M
        Fv[remm] = FO_rho

        while True:
            funv = np.inf
            II = np.zeros((Num, 2))
            
            II[:, 0] = np.linalg.norm(A @ x - b)**2 / 2 + s * np.linalg.norm(x, 1)
            for i in range(Num):
                i22 = np.array([int(b) for b in bin(i)[2:].zfill(num)], dtype=int)
                gx[Is == 2] = np.sign(x[Is == 2]) * i22
                z = np.maximum(s * gx - gf + alpha * x - s, np.minimum(0, s * gx - gf + alpha * x + s)) / alpha
                II[i, 1] = np.linalg.norm(z - x)**2
                hzt = np.sum(np.maximum(z - t, 0) + np.maximum(0, -z - t))
                F_rho = np.linalg.norm(A @ z - b)**2 / 2 + s * (np.linalg.norm(z, 1) - hzt) + gamma / 2 * II[i, 1]
                II[i, 0] = II[i, 0] - s * np.sum(gx * x - gx**2 * t)
                if F_rho < funv:
                    x_new = z
                    funv = F_rho
            
            Mum = 0
            hzt = np.sum(np.maximum(x_new - t, 0) + np.maximum(0, -x_new - t))
            F_rho = np.linalg.norm(A @ x_new - b)**2 / 2 + s * (np.linalg.norm(x_new, 1) - hzt)
            for i in range(Num):
                if F_rho <= max(max(Fv), II[i, 0]) - gamma * np.linalg.norm(x_new - x)**2 / 2 - gamma * II[i, 1] / 2:
                    Mum += 1
            if Mum == Num:
                break
            else:
                alpha *= 2
        
        TERM = FO_rho - F_rho
        if abs(TERM) < epsilon:
            break

        gfn = A.T @ (A @ x_new - b)
        if np.linalg.norm(x - x_new) == 0:
            alpha = alpha_max
        else:
            alpha = abs((gfn - gf).T @ (x_new - x)) / np.linalg.norm(x_new - x)**2
            alpha = max(alpha_min, min(alpha, alpha_max))

        gf = gfn
        FO_rho = F_rho
        x = x_new

    return x
####################################定义NEPDCAe########################################################################
def NEPDCAe(A, b, lambd, t, x0):
    epsilon = 1e-6
    eta = 1e-4
    theta = np.zeros(200)
    beta_v = np.zeros(200)
    theta[0] = (1 + np.sqrt(5)) / 2
    beta_v[0] = 0

    for i in range(1, 200):
        theta[i] = (1 + np.sqrt(1 + 4 * theta[i - 1]**2)) / 2
        beta_v[i] = 0.99 * (theta[i - 1] - 1) / theta[i]

    alpha = 1
    c = (0.99)**2
    x = x0
    bx = x
    n = len(x)
    s = lambd / t
    hzt = np.sum(np.maximum(x - t, 0) + np.maximum(0, -x - t))
    FO_rho = np.linalg.norm(A @ x - b)**2 / 2 + s * (np.linalg.norm(x, 1) - hzt)
    gf = A.T @ (A @ bx - b)

    for iter in range(2000):
        #if iter % 5 == 0:
            #t = np.maximum(t * 0.999, 0.0002)
        #s = lambd/ t
        I = np.zeros((n, 3))
        dxt = np.maximum(x - t, 0) + np.maximum(0, -x - t) - eta
        I[x - t >= dxt, 0] = 1
        I[0 >= dxt, 1] = 1
        I[-x - t >= dxt, 2] = 1
        Is = np.sum(I, axis=1)
        num = np.sum(Is == 2)
        if num > 20:
            break
        Num = 2**num
        Gx = np.column_stack((np.ones(n), np.zeros(n), -np.ones(n)))
        nzg = np.sum(I * Gx, axis=1)
        gx = np.zeros(n)
        gx[Is == 1] = nzg[Is == 1]
        funv = np.inf

        for i in range(Num):
            i22 = np.array([int(b) for b in bin(i)[2:].zfill(num)], dtype=int)
            gx[Is == 2] = np.sign(x[Is == 2]) * i22
            z = np.maximum(s * gx - gf + alpha * bx - s, np.minimum(0, s * gx - gf + alpha * bx + s)) / alpha
            hzt = np.sum(np.maximum(z - t, 0) + np.maximum(0, -z - t))
            F_rho = np.linalg.norm(A @ z - b)**2 / 2 + s * (np.linalg.norm(z, 1) - hzt) + c * (z - x).T @ (z - x) / 2
            if F_rho < funv:
                x_new = z
                funv = F_rho

        hzt = np.sum(np.maximum(x_new - t, 0) + np.maximum(0, -x_new - t))
        F_rho = np.linalg.norm(A @ x_new - b)**2 / 2 + s * (np.linalg.norm(x_new, 1) - hzt)
        TERM = FO_rho - F_rho
        if abs(TERM) < epsilon:
            break

        FO_rho = F_rho
        if iter % 200 == 0:
            beta = 0
        else:
            remm = iter % 200
            beta = beta_v[remm]

        bx = x_new + beta * (x_new - x)
        x = x_new
        gf = A.T @ (A @ bx - b)

    return x

####################################定义pnHPDCA########################################################################
def pnHPDCA(A, b, lambd, t, x0):
    epsilon = 1e-4
    eta = 1e-1
    x = x0
    n = len(x)
    s = lambd / t
    x_new = x
    hzt = np.sum(np.maximum(x - t, 0) + np.maximum(0, -x - t))
    FO_rho = np.linalg.norm(A @ x - b)**2 / 2 + s * (np.linalg.norm(x, 1) - hzt)
    M = 5
    Fv = np.zeros(M)
    alpha = 1
    gamma = 1e-3
    alpha_min = 1e-8
    alpha_max = 1e8
    gf = A.T @ (A @ x - b)
    kappa = 1

    for iter in range(1000):
        #if iter % 5 == 0:
            #t = np.maximum(t * 0.999, 0.0002)
        #s = lambd/ t
        dxt = np.maximum(x - t, 0) + np.maximum(0, -x - t)
        II = np.zeros(n)
        II[x - t == dxt] = 1
        II[0 == dxt] = 0
        II[-x - t == dxt] = -1
        gx = II
        ii = iter % M
        Fv[ii] = FO_rho

        while True:
            z = np.maximum(s * gx - gf + alpha * x - s, np.minimum(0, s * gx - gf + alpha * x + s)) / alpha
            hzt = np.sum(np.maximum(z - t, 0) + np.maximum(0, -z - t))
            F_rho = np.linalg.norm(A @ z - b)**2 / 2 + s * (np.linalg.norm(z, 1) - hzt)
            if F_rho < np.max(Fv) - gamma * np.linalg.norm(z - x)**2 :
                break
            alpha *= 2

        TERM = np.max(Fv) - F_rho
        if TERM >= kappa * epsilon:
            x_new = z
        else:
            I = np.zeros((n, 3))
            dxt = np.maximum(x - t, 0) + np.maximum(0, -x - t) - eta
            I[x - t >= dxt, 0] = 1
            I[0 >= dxt, 1] = 1
            I[-x - t >= dxt, 2] = 1
            Is = np.sum(I, axis=1)
            num = np.sum(Is == 2)
            Num = 2**num
            funv = np.inf

            for i in range(Num):
                i22 = np.array([int(b) for b in bin(i)[2:].zfill(num)], dtype=int)
                gx[Is == 2] = np.sign(x[Is == 2]) * i22
                z = np.maximum(s * gx - gf + x - s, np.minimum(0, s * gx - gf + x + s))
                hzt = np.sum(np.maximum(z - t, 0) + np.maximum(0, -z - t))
                F_rho = np.linalg.norm(A @ z - b)**2 / 2 + s * (np.linalg.norm(z, 1) - hzt)
                TERM = FO_rho - F_rho
                if TERM >= epsilon:
                    x_new = z
                    break

                l_rho = (gf - s * gx).T @ (z - x) + 0.5 * np.linalg.norm(z - x)**2 + s * np.linalg.norm(z, 1) - s * np.sum(gx * z - gx**2 * t)
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

        gfn = A.T @ (A @ x_new - b)
        if np.linalg.norm(x - x_new) == 0:
            alpha = alpha_max
        else:
            alpha = abs((gfn - gf).T @ (x_new - x)) / np.linalg.norm(x_new - x)**2
            alpha = max(alpha_min, min(alpha, alpha_max))

        gf = gfn
        FO_rho = F_rho
        x = x_new

    return x



####################################主函数########################################################################
def main():
    n = 2**13
    m = 2**11
    Kr = np.arange(425, 431)
    num_K = len(Kr)
    num_test = 20
    t = 0.11
    lambda_ = 0.01
    r = 0.001

    for kk in range(num_K):
        K = Kr[kk]
        funv = np.zeros(num_test)
        succ = np.zeros(num_test)
        times = np.zeros(num_test)
        Iter = np.zeros(num_test)

        for nnt in range(num_test):
            print(f"\n\nExperiment on m = {m}, n = {n}, K = {K}, \t No. test = {nnt + 1}.")
            print("----------------------------------------------------------")

            np.random.seed(nnt)
            xs = np.zeros(n)
            q = np.random.permutation(n)
            xs[q[:K]] = np.sign(np.random.randn(K))
            A = np.random.randn(m, n)
            A = orth(A.T).T
            b = A @ xs + r * np.random.randn(m)
            xinit = np.zeros_like(xs)
     
       

            start_time = time.time()
            #xs_f = HA(A, b, lambda_, t, xinit)
            #xs_f = ANPG(A, b, lambda_, t, xinit)
            #xs_f = EPDCAe(A, b, lambda_, t, xinit)
            xs_f = fnHPDCA(A, b, lambda_, t, xinit)
            #xs_f = NEPDCA(A, b, lambda_, t, xinit)
            #xs_f = NEPDCAe(A, b, lambda_, t, xinit)
            #xs_f = pnHPDCA(A, b, lambda_, t, xinit)
            times[nnt] = time.time() - start_time
      

            succ[nnt] = funv_succ( xs_f, xs)
            #Iter[nnt] = iter
            print(f" succ={succ[nnt]:.4e} time={times[nnt]:.4f}")
            print("----------------------------------------------------------")
            #print(f"iter={iter}")


        print(f"\n\nAveraged result for m = {m}, n = {n}, K = {K}")
        print("----------------------------------------------------------")
        print(f"SOLVER:   succ={np.mean(succ):.4e} time={np.mean(times):.4e}")
        print("----------------------------------------------------------")
        
if __name__ == "__main__":
    main()


