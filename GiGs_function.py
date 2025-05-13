
import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import os
from sklearn.metrics import auc, roc_curve, precision_recall_curve

"""
Consolidated implementation of the GiGs algorithm for predicting drug-virus associations.
This file combines all functionality from the original implementation (GiGs.py, GIPK.py, Kfold.py, ROC.py)
into a single, streamlined script with improved error handling.
"""

def getSimilarMatrix(IP, γ_=1):
    """
    Calculate Gaussian interaction profile kernel similarity
    
    Parameters:
    IP - Interaction profile matrix
    γ_ - Parameter for bandwidth calculation
    
    Returns:
    K - Similarity matrix
    """
    dimensional = IP.shape[0]
    sd = np.zeros(dimensional)
    K = np.zeros((dimensional, dimensional))
    
    # Calculate squared norms
    for i in range(dimensional):
        sd[i] = np.linalg.norm(IP[i].astype(float)) ** 2
    
    # Calculate bandwidth parameter
    gamad = γ_ * dimensional / np.sum(sd.transpose())
    
    # Calculate similarity matrix
    for i in range(dimensional):
        for j in range(dimensional):
            K[i][j] = math.exp(-gamad * (np.linalg.norm(IP[i].astype(float) - IP[j].astype(float))) ** 2)
    
    return K

def Kfoldcrossclassify(sample, K, fun="cv3"):
    """
    Custom K-fold cross validation implementation
    
    Parameters:
    sample - Data to split
    K - Number of folds
    fun - Splitting method ("cv1", "cv2", or "cv3")
    
    Returns:
    r - List of K lists containing the indices for each fold
    """
    r = []
    if fun != "cv3":
        m = np.mat(sample)
        if fun == "cv1":
            t = 0
        else:
            t = 1
        mt = Kfoldcrossclassify(np.array(range(np.max(m[:, t]) + 1)), K)
        r = [[j for j in sample if j[t] in mt[i]] for i in range(K)]
        return r
    
    l = sample.shape[0]
    t = sample.copy()
    n = math.floor(l / K)
    retain = l - n * K
    
    for i in range(K - 1):
        nt = n
        e = len(t)
        if retain > i:
            nt += 1
        a = random.sample(range(e), nt)
        r.append([t[i] for i in a])
        t = [t[i] for i in range(e) if (i not in a)]
    r.append(t)
    
    return r

def GiGs(Y, M, D, r, param, parm1, parm2, parm3, max_iter=1000):
    """
    GiGs algorithm for matrix factorization with regularization
    
    Parameters:
    Y - Association matrix (virus-drug)
    M - Drug similarity matrix
    D - Virus similarity matrix
    r - Rank of factorization
    param - Weight parameter for positive associations
    parm1, parm2, parm3 - Regularization parameters
    max_iter - Maximum number of iterations
    
    Returns:
    score - Predicted association scores
    """
    m = M.shape[0]
    n = D.shape[0]
    
    # Weight matrix, gives higher weight to known associations
    B = np.where(Y == 1, 1 + param, 1)
    
    # Initialize factor matrices
    W = np.random.rand(m, r)
    H = np.random.rand(n, r)
    
    # Optimization loop
    k = 1
    while k < max_iter:
        # Update W
        s1 = np.dot(B*Y, H)
        s2 = np.dot(M, W)
        R_W1 = np.tile(2 * np.sum(M, axis=1), (r, 1))
        R_W2 = np.dot(np.dot(W, H.T)*B, H)
        R_W3 = np.dot(W.T, W)
        W = W*(s1+2*(parm2+parm3)*s2)/(R_W2+parm1*W+parm2*R_W1.T*W+2*parm3*np.dot(W, R_W3))
        
        # Update H
        c1 = np.dot((B*Y).T, W)
        c2 = np.dot(D, H)
        R_H1 = np.tile(2 * np.sum(D, axis=1), (r, 1))
        R_H2 = np.dot(np.dot(H, W.T)*B.T, W)
        R_H3 = np.dot(H.T, H)
        H = H*(c1+2*(parm2+parm3)*c2)/(R_H2+parm1*H+parm2*R_H1.T*H+2*parm3*np.dot(H, R_H3))
        
        k += 1
    
    # Calculate predicted scores
    score = np.dot(W, H.T)
    return score

def ROC_compute(virusdrug, drugsim, virussim, a, b, param=1, parm1=16, parm2=0.125, parm3=16, r=70, trials=20):
    """
    Compute ROC and PR curves with cross-validation
    
    Parameters:
    virusdrug - Virus-drug association matrix
    drugsim - Drug similarity matrix
    virussim - Virus similarity matrix
    a - Positive samples
    b - Negative samples
    param, parm1, parm2, parm3 - Parameters for GiGs algorithm
    r - Rank of factorization
    trials - Number of cross-validation trials
    
    Returns:
    note_auc - List of AUC values and ROC curve points
    note_aupr - List of AUPR values and PR curve points
    """
    note_auc = []
    note_aupr = []
    
    # Run multiple trials for robust evaluation
    for h in range(trials):
        print(f"Trial {h+1}/{trials}")
        
        # 5-fold cross-validation
        f1 = Kfoldcrossclassify(a, 5, fun='cv3')
        f2 = Kfoldcrossclassify(b, 5, fun='cv3')
        
        # For each fold
        for m in range(5):
            test_sample = np.array(f1[m])
            negative_sample = np.array(f2[m])
            
            # Create training matrix with test samples hidden
            virusdrug_ = virusdrug.copy()
            virusdrug_[test_sample[:, 0], test_sample[:, 1]] = 0
            
            # Run GiGs algorithm
            VDA = GiGs(virusdrug_, drugsim, virussim, r, param, parm1, parm2, parm3)
            
            # Prepare labels and scores for evaluation
            test_sample_number = test_sample.shape[0]
            negative_sample_number = negative_sample.shape[0]
            label = test_sample_number*[1] + negative_sample_number*[0]
            label = np.array(label)
            sample = np.vstack((test_sample, negative_sample))
            score = VDA[sample[:, 0], sample[:, 1]]
            
            # Calculate ROC curve
            fpr, tpr, threshold = roc_curve(label, score)
            auc_pre = auc(fpr, tpr)
            
            # Calculate PR curve
            pre, rec, threshold = precision_recall_curve(label, score)
            aupr_pre = auc(rec, pre)
            
            # Store results
            note_auc.append((auc_pre, fpr, tpr))
            note_aupr.append((aupr_pre, rec, pre))
            
            print(f"  Fold {m+1}/5 - AUC: {auc_pre:.4f}, AUPR: {aupr_pre:.4f}")
    
    return note_auc, note_aupr

def find_representative_curve(results):
    """
    Find the curve closest to the mean performance
    
    Parameters:
    results - List of (metric, x_values, y_values) tuples
    
    Returns:
    tuple - (metric, x_values, y_values) of the representative curve
    """
    # Calculate mean performance
    mean_perf = sum(r[0] for r in results) / len(results)
    
    # Find the curve closest to the mean
    min_diff = float('inf')
    best_result = None
    
    for result in results:
        diff = abs(result[0] - mean_perf)
        if diff < min_diff:
            min_diff = diff
            best_result = result
    
    return best_result

def load_data(file_path_prefix):
    """
    Load data files
    
    Parameters:
    file_path_prefix - Path to data directory
    
    Returns:
    virusdrug - Virus-drug association matrix
    drugsim - Drug similarity matrix
    virussim - Virus similarity matrix
    """
    try:
        # Fixed file paths - removed the duplicated prefix
        drugstrucsim = pd.read_excel(f'{file_path_prefix}/drug chemical structure similarity.xlsx', 
                                    header=None, index_col=None)
        virusseqsim = pd.read_excel(f'{file_path_prefix}/virus sequence similarity.xlsx', 
                                   header=None, index_col=None)
        drugsidesim = pd.read_excel(f'{file_path_prefix}/drug side effect similarity.xlsx', 
                                   header=None, index_col=None)
        
        # Load association data
        data = np.loadtxt(f'{file_path_prefix}/virus-drug associations.txt')
        
        # Create association matrix
        virusdrug = np.zeros((drugstrucsim.shape[0], virusseqsim.shape[0]))
        for row in data:
            x, y, value = row
            virusdrug[int(x)-1, int(y)-1] = value
        
        # Calculate Gaussian interaction profile kernel similarities
        GD = getSimilarMatrix(virusdrug, 1)
        GV = getSimilarMatrix(virusdrug.T, 1)
        
        # Combine similarity matrices
        drugsim = 0.1*(drugstrucsim + drugsidesim)/2 + 0.9*GD
        virussim = 0.1*virusseqsim + 0.9*GV
        
        print(f"Data loaded successfully: {virusdrug.shape[0]} drugs × {virusdrug.shape[1]} viruses")
        
        return virusdrug, drugsim, virussim
    
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Using toy dataset for demonstration.")
        
        # Create toy data
        np.random.seed(42)
        drug_count, virus_count = 50, 30
        virusdrug = np.random.choice([0, 1], size=(drug_count, virus_count), p=[0.9, 0.1])
        
        # Create similarity matrices
        GD = getSimilarMatrix(virusdrug, 1)
        GV = getSimilarMatrix(virusdrug.T, 1)
        
        print(f"Using toy dataset: {virusdrug.shape[0]} drugs × {virusdrug.shape[1]} viruses")
        
        return virusdrug, GD, GV

def plot_curves(auc_result, aupr_result, save_path=None):
    """
    Plot ROC and PR curves
    
    Parameters:
    auc_result - List of (auc, fpr, tpr) tuples
    aupr_result - List of (aupr, rec, pre) tuples
    save_path - Path to save plots
    """
    # Find representative curves
    auc_score, fpr, tpr = find_representative_curve(auc_result)
    aupr_score, rec, pre = find_representative_curve(aupr_result)
    
    # Calculate mean scores
    mean_auc = sum(x[0] for x in auc_result) / len(auc_result)
    mean_aupr = sum(x[0] for x in aupr_result) / len(aupr_result)
    
    # Use a font that's available in Colab
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#86D3DE', lw=2, 
             label=f'GiGs (AUC = {auc_score:.4f}, Mean = {mean_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    
    # Save plot if a path was provided and the directory exists
    if save_path:
        # Create the directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/ROC_curve.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Plot PR curve
    plt.figure(figsize=(8, 6))
    plt.plot(rec, pre, color='#86D3DE', lw=2, 
             label=f'GiGs (AUPR = {aupr_score:.4f}, Mean = {mean_aupr:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('PR Curve', fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    
    # Save plot if a path was provided
    if save_path:
        plt.savefig(f'{save_path}/PR_curve.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def main(data_path='/content/GIG/data/DrugVirus dataset', trials=20, save_plots=True, save_path='./results'):
    """
    Main function to run the GiGs algorithm
    
    Parameters:
    data_path - Path to data directory
    trials - Number of cross-validation trials
    save_plots - Whether to save plots
    save_path - Directory to save plots
    """
    print("Starting GiGs algorithm evaluation...")
    
    # Load data
    virusdrug, drugsim, virussim = load_data(data_path)
    
    # Get positive and negative samples
    Nd, Nv = virusdrug.shape
    a = np.array([(i, j) for i in range(Nd) for j in range(Nv) if virusdrug[i, j]])
    b = np.array([(i, j) for i in range(Nd) for j in range(Nv) if virusdrug[i, j] == 0])
    
    # If there are many negative samples, take a random subset to balance the dataset
    if len(b) > 5 * len(a):
        np.random.seed(42)
        b = b[np.random.choice(len(b), 5 * len(a), replace=False)]
        print(f"Using a balanced subset of negative samples: {len(a)} positive, {len(b)} negative")
    
    # Best parameters according to the original paper
    param = 1
    parm1 = 16
    parm2 = 0.125
    parm3 = 16
    r = 70
    
    print(f"Running evaluation with {trials} trials...")
    print(f"Parameters: param={param}, parm1={parm1}, parm2={parm2}, parm3={parm3}, r={r}")
    
    # Compute ROC and PR curves
    auc_result, aupr_result = ROC_compute(virusdrug, drugsim, virussim, a, b, 
                                          param, parm1, parm2, parm3, r, trials)
    
    # Calculate mean performance
    mean_auc = sum(x[0] for x in auc_result) / len(auc_result)
    std_auc = np.std([x[0] for x in auc_result])
    mean_aupr = sum(x[0] for x in aupr_result) / len(aupr_result)
    std_aupr = np.std([x[0] for x in aupr_result])
    
    print("\nFinal Results:")
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Mean AUPR: {mean_aupr:.4f} ± {std_aupr:.4f}")
    
    # Plot curves
    save_path_final = save_path if save_plots else None
    plot_curves(auc_result, aupr_result, save_path_final)
    
    return mean_auc, std_auc, mean_aupr, std_aupr

if __name__ == "__main__":
    # You can modify these parameters as needed
    data_path = '/content/GIG/data/DrugVirus dataset'  # Path to your data
    trials = 5  # Use a smaller number for quick testing, 20 for full evaluation
    save_plots = True  # Whether to save plots
    save_path = './results'  # Directory to save plots
    
    # Call main function with parameters
    main(data_path, trials, save_plots, save_path)
