import numpy as np
import copy
import time
from tslearn.datasets import UCR_UEA_datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.linalg import norm

####################################定义ANPG########################################################################
def ASPG(A, b, lambda_, t, x0):
    epsilon = 1e-6 # stopping criterion
    x = x0
    alpha = np.linalg.norm(A.T @ A, 2)
    t = t * 0.99
    n = len(x)
    s = lambda_ / t
    x_old = copy.deepcopy(x)
    t1 = 1

    for iter in range(1000):
        II = np.zeros(n)
        II = (x >= t) * 1 + (x <= -t) * (-1)
        t2 = np.sqrt(t1**2 + 2 * t1)
        beta = (t1 - 1) / t2
        z = x + beta * (x - x_old) * np.abs(II)
        gf = A.T @ (A @ z - b)
        w = z - (gf - s * II) / alpha
        x_old = copy.deepcopy(x)
        x = np.sign(w) * np.maximum(np.abs(w) - s / alpha, 0)
        t1 = copy.deepcopy(t2)

        if iter > 1 and np.linalg.norm(x - x_old) / np.linalg.norm(x_old) < epsilon:
            break

    return x
####################################定义HA########################################################################
def HA(A, b, lambda_, t, x0):
    epsilon = 1e-4
    eta = 1.0e-1
    x = x0
    n = len(x)
    s = lambda_ / t
    x_new = x
    alpha = np.linalg.norm(A.T @ A, 2)
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
    eta = 1e-3
    theta = np.zeros(200)
    beta_v = np.zeros(200)
    theta[0] = (1 + np.sqrt(5)) / 2
    beta_v[0] = 0
    for i in range(1, 200):
        theta[i] = (1 + np.sqrt(1 + 4 * theta[i-1]**2)) / 2
        beta_v[i] = 1 * (theta[i-1] - 1) / theta[i]
    
    alpha = np.linalg.norm(A.T @ A, 2)
    c = 1
    x = x0
    bx = x0.copy()
    n = len(x)
    s = lambd / t
    
    hzt = np.sum(np.maximum(x - t, 0) + np.maximum(0, -x - t))
    FO_rho = np.linalg.norm(A @ x - b)**2 / 2 + s * (np.linalg.norm(x, 1) - hzt)
    gf = A.T @ (A @ bx - b)
    
    for iter in range(1000):
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



def fnHPDCA(A, b, lambd, t, x0, max_iter=1000):

    epsilon = 1e-4
    eta = 1e-1
    x = x0.copy()
    n = len(x)
    s = lambd / t
    hzt = np.sum(np.maximum(x - t, 0) + np.maximum(0, -x - t))
    FO_rho = 0.5 * norm(A @ x - b)**2 + s * (np.linalg.norm(x, 1) - hzt)
    M = 5
    Fv = np.zeros(M)
    alpha =np.linalg.norm(A.T @ A, 2)
    gamma = 1e-3
    alpha_min = 0.1*np.linalg.norm(A.T @ A, 2)
    alpha_max = np.linalg.norm(A.T @ A, 2)
    gf = A.T @ (A @ x - b)
    rho = 1

    for iter in range(max_iter):
        beta = alpha
        dxt = np.maximum(x - t, 0) + np.maximum(0, -x - t)
        II = np.zeros(n)
        II[(x - t) == dxt] = 1
        II[0 == dxt] = 0
        II[(-x - t) == dxt] = -1
        gx = II

        ii = iter % M
        Fv[ii] = FO_rho

        # Line search
        while True:
            temp = s * gx - gf + alpha * x
            z = np.maximum(temp - s, 0) + np.minimum(temp + s, 0)
            z /= alpha
            hzt = np.sum(np.maximum(z - t, 0) + np.maximum(0, -z - t))
            F_rho = 0.5 * norm(A @ z - b)**2 + s * (np.linalg.norm(z, 1) - hzt)
            if F_rho < np.max(Fv) - gamma * norm(z - x)**2:
                break
            alpha *= 2

        TERM = np.max(Fv) - F_rho

        if TERM >= rho * epsilon:
            x_new = z
        else:
            # Fine search
            dxt = np.maximum(x - t, 0) + np.maximum(0, -x - t) - eta
            I = np.zeros((n, 3))
            I[(x - t) >= dxt, 0] = 1
            I[0 >= dxt, 1] = 1
            I[(-x - t) >= dxt, 2] = 1
            Is = np.sum(I, axis=1)
            num = np.sum(Is == 2)
            Num = 2 ** num

            while True:
                lin_s = 1
                funv = np.inf
                for i in range(Num):
                    i22 = np.array(list(np.binary_repr(i, num)), dtype=int)
                    gx_new = gx.copy()
                    gx_new[Is == 2] = np.sign(x[Is == 2]) * i22

                    temp = s * gx_new - gf + alpha * x
                    z = np.maximum(temp - s, 0) + np.minimum(temp + s, 0)
                    z /= alpha

                    hzt = np.sum(np.maximum(z - t, 0) + np.maximum(0, -z - t))
                    F_rho = 0.5 * norm(A @ z - b)**2 + s * (np.linalg.norm(z, 1) - hzt)

                    if F_rho < np.max(Fv) - max(rho * epsilon, gamma * norm(z - x)**2):
                        lin_s = 0
                        x_new = z
                        break

                    l_rho = (gf - s * gx_new) @ (z - x) + 0.5 * norm(z - x)**2 + s * np.linalg.norm(z, 1) - s * np.sum(gx_new * z - gx_new**2 * t)
                    if l_rho < funv:
                        x_new = z
                        funv = l_rho

                if lin_s == 0:
                    break
                else:
                    hzt = np.sum(np.maximum(x_new - t, 0) + np.maximum(0, -x_new - t))
                    F_rho = 0.5 * norm(A @ x_new - b)**2 + s * (np.linalg.norm(x_new, 1) - hzt)
                    if F_rho < np.max(Fv) - gamma * norm(x_new - x)**2:
                        break
                    else:
                        alpha *= 2

            TERM = np.max(Fv) - F_rho
            if TERM < rho * epsilon:
                if epsilon <= 1e-6:
                    break
                epsilon *= 0.1

        gfn = A.T @ (A @ x_new - b)
        if np.allclose(x, x_new):
            alpha = alpha_max
        else:
            alpha = abs((gfn - gf) @ (x_new - x)) / norm(x_new - x)**2
            alpha = np.clip(alpha, alpha_min, alpha_max)

        gf = gfn
        FO_rho = F_rho
        x = x_new

    return x

####################################定义NEPDCA########################################################################
def NEPDCA(A, b, lambd, t, x0):
    epsilon = 1e-6
    eta = 1e-3
    x = x0
    n = len(x)
    s = lambd / t
    hzt = np.sum(np.maximum(x - t, 0) + np.maximum(0, -x - t))
    FO_rho = np.linalg.norm(A @ x - b)**2 / 2 + s * (np.linalg.norm(x, 1) - hzt)
    M = 5
    Fv = np.zeros(M)
    alpha_max = np.linalg.norm(A.T @ A, 2)
    alpha = alpha_max
    gamma = 1e-4
    alpha_min = 1e-8

    gf = A.T @ (A @ x - b)
    for iter in range(1000):
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
            if alpha >alpha_max:
                break
        
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
    eta = 1e-3
    theta = np.zeros(200)
    beta_v = np.zeros(200)
    theta[0] = (1 + np.sqrt(5)) / 2
    beta_v[0] = 0

    for i in range(1, 200):
        theta[i] = (1 + np.sqrt(1 + 4 * theta[i - 1]**2)) / 2
        beta_v[i] = 0.99 * (theta[i - 1] - 1) / theta[i]

    alpha = np.linalg.norm(A.T @ A, 2)
    c = (0.99)**2
    x = x0
    bx = x
    n = len(x)
    s = lambd / t
    hzt = np.sum(np.maximum(x - t, 0) + np.maximum(0, -x - t))
    FO_rho = np.linalg.norm(A @ x - b)**2 / 2 + s * (np.linalg.norm(x, 1) - hzt)
    gf = A.T @ (A @ bx - b)

    for iter in range(2000):
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
    # Algorithm 3
    epsilon = 1e-4
    eta = 1e-1
    x = np.copy(x0)
    n = len(x)
    s = lambd / t
    x_new = np.copy(x)
    
    def hzt_func(x, t):
        return np.sum(np.maximum(x - t, 0) + np.maximum(0, -x - t))
    
    hzt = hzt_func(x, t)
    FO_rho = 0.5 * np.linalg.norm(A @ x - b) ** 2 + s * (np.linalg.norm(x, 1) - hzt)
    M = 5
    Fv = np.zeros(M)
    alpha = np.linalg.norm(A.T @ A, 2)
    gamma = 1e-3
    alpha_min = 0.1*np.linalg.norm(A.T @ A, 2)
    alpha_max = np.linalg.norm(A.T @ A, 2)
    gf = A.T @ (A @ x - b)
    kappa = 1

    for iter in range(1, 1001):
        dxt = np.maximum(x - t, 0) + np.maximum(0, -x - t)
        II = np.zeros(n)
        II[(x - t) == dxt] = 1
        II[(-x - t) == dxt] = -1
        gx = np.copy(II)

        ii = iter % M
        Fv[ii] = FO_rho
        
        while True:
            z = np.maximum((s * gx - gf + alpha * x - s) / alpha,
                           np.minimum(0, (s * gx - gf + alpha * x + s) / alpha))
            hzt = hzt_func(z, t)
            F_rho = 0.5 * np.linalg.norm(A @ z - b) ** 2 + s * (np.linalg.norm(z, 1) - hzt)
            if F_rho < np.max(Fv) - gamma * np.linalg.norm(z - x) ** 2 :
                break
            alpha *= 2

        TERM = np.max(Fv) - F_rho

        if TERM >= kappa * epsilon:
            x_new = z
        else:
            I = np.zeros((n, 3))
            dxt = np.maximum(x - t, 0) + np.maximum(0, -x - t) - eta
            I[(x - t) >= dxt, 0] = 1
            I[(0) >= dxt, 1] = 1
            I[(-x - t) >= dxt, 2] = 1
            Is = np.sum(I, axis=1)
            num = np.sum(Is == 2)
            Num = 2 ** num
            funv = np.inf

            idx_two = np.where(Is == 2)[0]

            for i in range(Num):
                i22 = np.array(list(np.binary_repr(i, width=num)), dtype=int)
                gx_temp = np.copy(gx)
                gx_temp[idx_two] = np.sign(x[idx_two]) * i22

                z = np.maximum((s * gx_temp - gf + x - s),
                               np.minimum(0, (s * gx_temp - gf + x + s)))
                hzt = hzt_func(z, t)
                F_rho = 0.5 * np.linalg.norm(A @ z - b) ** 2 + s * (np.linalg.norm(z, 1) - hzt)
                TERM = FO_rho - F_rho

                if TERM >= epsilon:
                    x_new = z
                    break

                l_rho = (gf - s * gx_temp).T @ (z - x) + 0.5 * np.linalg.norm(z - x) ** 2 \
                        + s * np.linalg.norm(z, 1) - s * np.sum(gx_temp * z - gx_temp ** 2 * t)
                if l_rho < funv:
                    x_new = z
                    funv = l_rho

            hzt = hzt_func(x_new, t)
            F_rho = 0.5 * np.linalg.norm(A @ x_new - b) ** 2 + s * (np.linalg.norm(x_new, 1) - hzt)
            TERM = FO_rho - F_rho

            if TERM < epsilon:
                if epsilon <= 1e-6:
                    break
                epsilon *= 0.1

        gfn = A.T @ (A @ x_new - b)

        if np.linalg.norm(x - x_new) == 0:
            alpha = alpha_max
        else:
            alpha = np.abs((gfn - gf).T @ (x_new - x)) / (np.linalg.norm(x_new - x) ** 2)
            alpha = np.clip(alpha, alpha_min, alpha_max)

        gf = gfn
        FO_rho = F_rho
        x = np.copy(x_new)

    return x




# Dataset loader
ucr_datasets = UCR_UEA_datasets()

# List of datasets to be evaluated
datasets = ["MoteStrain", "GunPoint","Coffee","ECGFiveDays","ECG200","ToeSegmentation1","BirdChicken","ShapeletSim"
             ]
# Algorithms to be evaluated
algorithms = {
    'ANPG': ANPG,
    'HA': HA,
    'fnHPDCA': fnHPDCA,
    'pnHPDCA': pnHPDCA,
    'NEPDCA': NEPDCA,
    'NEPDCAe': NEPDCAe,
    'EPDCAe': EPDCAe
}

# Function to evaluate algorithms on a dataset
def evaluate_algorithms(X_train, y_train, X_test, y_test, algorithms,lambda_):
    results = []
    
    for name, algorithm in algorithms.items() :
    
        C=1
        #m,n = X_train.shape
        m,n = X_train.shape
        e=np.ones((m,1))
        M11=np.dot(X_train.T,X_train)+1/C*np.eye(n)
        M12=np.dot(X_train.T,e)
        M21=np.dot(e.T,X_train)
        M22=np.dot(e.T,e)
        a=np.vstack((np.hstack((M11,M12)),np.hstack((M21,M22))))
        b=np.vstack((X_train.T,e.T))@y_train
        start_time = time.time()
        # Call each algorithm (assume it returns predicted labels for test data)
        xs=algorithm(a, b,lambda_, 5.00e-05, np.zeros((n+1)))
        elapsed_time = time.time() - start_time
        w_ = xs[:n]
        b_ = xs[-1]
        train_pred = np.dot(X_train, w_) + b_
        test_pred = np.dot(X_test, w_) + b_
        train_pred = np.sign(train_pred)
        test_pred = np.sign(test_pred)
        
        # Calculate accuracies
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        
     

        results.append({
            'algorithm': name,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'time': elapsed_time,
            'Sparsity':np.sum(w_ != 0)
            
        })
    
    return results

# Main loop to go through each dataset
i=0
Lambda=[0.002,0.2,0.01,0.09,0.1,0.2,0.5,0.1]

for dataset in datasets:
    lambda_=Lambda[i]
    # Load the dataset
    X_train, y_train, X_test, y_test = ucr_datasets.load_dataset(dataset)
    
    # Train/test split if necessary
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # 假设X_train和X_test的形状为 (n_samples, n_timestamps, n_features)
    n_samples_train, n_timestamps_train, n_features_train = X_train.shape
    n_samples_test, n_timestamps_test, n_features_test = X_test.shape

    # 将X_train和X_test转换为二维矩阵 (n_samples, n_timestamps * n_features)
    X_train = X_train.reshape(n_samples_train, n_timestamps_train * n_features_train)
    X_test= X_test.reshape(n_samples_test, n_timestamps_test * n_features_test)
    y_train = np.where(y_train == 1, 1, -1)  # 只保留1，其他全设为-1
    y_test = np.where(y_test == 1, 1, -1)
    # Run the evaluation for all algorithms
    results = evaluate_algorithms(X_train, y_train, X_test, y_test, algorithms,lambda_)
    i=i+1
    # Print results for the current dataset
    print(f"Results for dataset {dataset}:")
    for result in results:
        print(f"Algorithm: {result['algorithm']}, Train Acc: {result['train_acc']:.4f}, Test Acc: {result['test_acc']:.4f}, Time: {result['time']:.4f}s，Sparsity: {result['Sparsity']}")
    
    

    
   
    
   
    
    

    
    
    
    
    
    
    
