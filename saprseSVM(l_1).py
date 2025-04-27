import numpy as np
import copy
import time
from tslearn.datasets import UCR_UEA_datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Custom implementations or imports of your algorithms
# Placeholder functions for each algorithm. You need to replace these with actual implementations.
def ANPG(A, b, lambda_, t_, x0):
    epsilon = 1e-6  # stopping criterion
    x = x0
    norm_A = np.linalg.norm(A, ord=2)
    n = len(x)
    mu_0=1
    m=A.shape[0]
    mu=norm_A**2*mu_0
    #hzt = np.sum(np.maximum(x - t, 0) + np.maximum(0, -x - t))
    #FO_rho = np.linalg.norm(A @ x - b)**2 / 2 + s * (np.linalg.norm(x, 1) - hzt)
    #alpha = 1
    x_old=x0.copy()
    t1=1
    #计算函数值
    t=t_
    
    s = lambda_ / t*(mu/norm_A**2)
    hzt_old = np.sum(np.maximum(x_old - t, 0) + np.maximum(0, -x_old - t))
    FO_rho = np.linalg.norm(A @ x_old - b)**2 / 2 + s * (np.linalg.norm(x_old, 1) - hzt_old)
    
    
    for iter in range(5000):
        if iter % 5 == 0:
            t = np.maximum(t * 0.999, 1e-8)
        s = (lambda_ / t)*(mu/norm_A**2)
        II= np.zeros(n)
        II=(x>=t)*1+(x<=-t)*(-1)
        t2=(np.sqrt(t1**2+2*t1))
        beta=(t1-1)/t2
        
        
        z=x+beta*(x-x_old)*np.abs(II)
        #z=x+beta*(x-x_old)
        z1=A @ z
        p=np.sign(z1-b)*np.maximum(abs(z1-b)-mu,0)+b
        gf = (1/norm_A**2)*A.T @ (z1 - p)
        w=z-gf+s*II
        x_old=x.copy()
        x=np.sign(w)*np.maximum(abs(w)-s,0)
        hzt = np.sum(np.maximum(x - t, 0) + np.maximum(0, -x - t))
        F_rho = np.linalg.norm(A @ x - b)**2 / 2 + s * (np.linalg.norm(x, 1) - hzt)
       
        
        mu=t1**2/(t2**2-t2)*mu
        t1=t2.copy()
        if iter>10 and np.linalg.norm(x-x_old)/(np.maximum(np.linalg.norm(x_old),1))<epsilon:
            break
     
    return x

# Dataset loader
ucr_datasets = UCR_UEA_datasets()

# List of datasets to be evaluated
datasets = ["MoteStrain", "GunPoint","Coffee","ECGFiveDays","ECG200","ToeSegmentation1","BirdChicken","ShapeletSim"
             ]
algorithms = {
    'ANPG': ANPG,
    #'HA': HA,
    #'fnHPDCA': fnHPDCA,
    #'pnHPDCA': pnHPDCA,
    #'NEPDCA': NEPDCA,
    #'NEPDCAe': NEPDCAe,
    #'EPDCAe': EPDCAe
}

# Function to evaluate algorithms on a dataset
def evaluate_algorithms(X_train, y_train, X_test, y_test, algorithms,lambda_):
    results = []
    #Lambda = [0.002,0.2,0.01,0.09,0.1,0.05,0.5,0.1],lambda_为正则化参数在Lambda中选取
    #Lambda=[0.002,0.01,0.01,0.09,0.1,0.05,0.5,0.1]
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
        #num为w_中非零元素个数
        #记录Auc
        #prob=np.dot(X_test, w_) + b_
        #prob = 1/(1+np.exp(-prob))
        #auc = roc_auc_score(y_test, prob)
        
     

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

Lambda=[ 9.00e-08,3.00e-10,1.00e-9,2.00e-08,7.00e-08, 2.00e-07,7.00e-08,1.00e-06]
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
    #将y_train和y_test中等于2的改成1
    y_train[y_train==2]=-1
    y_test[y_test==2]=-1
    y_train[y_train==0]=-1
    y_test[y_test==0]=-1
    # Run the evaluation for all algorithms
    results = evaluate_algorithms(X_train, y_train, X_test, y_test, algorithms,lambda_)
    i=i+1
    # Print results for the current dataset
    print(f"Results for dataset {dataset}:")
    for result in results:
        print(f"Algorithm: {result['algorithm']}, Train Acc: {result['train_acc']:.4f}, Test Acc: {result['test_acc']:.4f}, Time: {result['time']:.4f}s, Sparsity: {result['Sparsity']}")
    
    

    

