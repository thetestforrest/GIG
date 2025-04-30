import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import precision_recall_curve
import seaborn as sns
from GIPK import getSimilarMatrix
from Kfold import Kfoldcrossclassify
from ROC import ROC_compute


#DrugVirus
drugstrucsim = pd.read_excel('./data/DrugVirus/drug chemical structure similarity.xlsx',header=None, index_col=None)
virusseqsim = pd.read_excel('./data/DrugVirus/virus sequence similarity.xlsx',header=None, index_col=None)
drugsidesim = pd.read_excel('./data/DrugVirus/drug side effect similarity.xlsx',header=None, index_col=None)
data = np.loadtxt('./data/DrugVirus/virus-drug associations.txt')

virusdrug = np.zeros((drugstrucsim.shape[0], virusseqsim.shape[0]))
for row in data:
    x, y, value = row
    virusdrug[int(x)-1, int(y)-1] = value


#VDA2
# drugstrucsim = pd.read_excel('./data/VDA2/drug similarity matrix.xlsx',header=None, index_col=None)
# virusseqsim = pd.read_excel('./data/VDA2/virus similarity matrix.xlsx',header=None, index_col=None)
# drugsidesim = np.eye(drugstrucsim.shape[0])
# virusdrug = pd.read_excel('./data/VDA2/virus-drug association matrix.xlsx',header=None, index_col=None)
# virusdrug = virusdrug.drop(0,axis=0).drop(0,axis=1)
# virusdrug = np.array(virusdrug).astype('float64')
    

#HDVD
# drugstrucsim = pd.read_csv('./data/HDVD/drugsim.csv',header=None, index_col=None)
# virusseqsim = pd.read_csv('./data/HDVD/virussim.csv',header=None, index_col=None)
# drugsidesim = np.eye(drugstrucsim.shape[0])
# virusdrug = pd.read_csv('./data/HDVD/virusdrug.csv',header=None, index_col=None)
# virusdrug = np.array(virusdrug)
   

Nd, Nv = virusdrug.shape
a = [(i, j) for i in range(Nd) for j in range(Nv) if virusdrug[i, j]]
a = np.array(a)

b = [(i, j) for i in range(Nd) for j in range(Nv) if virusdrug[i, j] == 0]
b = np.array(b)


#Gaussian interaction profile kernel similarity of drugs and viruses
GD = getSimilarMatrix(virusdrug,1)
GV = getSimilarMatrix(virusdrug.T,1)

drugsim = 0.1*(drugstrucsim + drugsidesim)/2+0.9*GD
drugsim = pd.DataFrame(drugsim)
virussim = 0.1*virusseqsim+0.9*GV
virussim = pd.DataFrame(virussim)

def GiGs(Y, M, D, r, param, parm1, parm2, parm3):
    
    m = M.shape[0]
    n = D.shape[0]
    B = np.where(Y == 1, 1 + param, 1)
    W = np.random.rand(m, r)
    H = np.random.rand(n, r)
    I = np.eye(r)

    k = 1
    while k<1000:
        
        s1 = np.dot(B*Y,H)
        s2 = np.dot(M,W)
        R_W1 = np.tile(2 * np.sum(M, axis=1), (r, 1))
        R_W2 = np.dot(np.dot(W,H.T)*B,H)
        # R_W2 = (W.dot(H.T)*B).dot(H)
        R_W3 = np.dot(W.T,W)
        # W = W*(s1+2*(parm2+parm3)*s2)/(R_W2+parm1*W+parm2*R_W1.T*W+2*parm3*W.dot(R_W3))
        W = W*(s1+2*(parm2+parm3)*s2)/(R_W2+parm1*W+parm2*R_W1.T*W+2*parm3*np.dot(W,R_W3))

        c1 = np.dot((B*Y).T,W)
        c2 = np.dot(D,H)
        R_H1 = np.tile(2 * np.sum(D, axis=1), (r, 1))
        # R_H2 = (H.dot(W.T)*B.T).dot(W)
        R_H2 = np.dot(np.dot(H,W.T)*B.T,W)
        R_H3 = np.dot(H.T,H)
        # H = H*(c1+2*(parm2+parm3)*c2)/(R_H2+parm1*H+parm2*R_H1.T*H+2*parm3*H.dot(R_H3))
        H = H*(c1+2*(parm2+parm3)*c2)/(R_H2+parm1*H+parm2*R_H1.T*H+2*parm3*np.dot(H,R_H3))
        
        k += 1
   
    score = np.dot(W,H.T)
    return score


#parameter settings on DrugVirus dataset
# with open('GiGs_parameter_settings.txt','a') as file:    
#     for p in range(1,11,1):
#         param = p
#         for j in range(-3,5,1):
#             parm1 = pow(2,j)
#             for k in range(-3,5,1):
#                 parm2 = pow(2,k) 
#                 for q in range(-3,5,1):
#                     parm3 = pow(2,q)
                    
#                     AUCs = 0
#                     for h in range(20):
#                         f1 = Kfoldcrossclassify(a, 5, fun='cv3')
#                         f2 = Kfoldcrossclassify(b, 5, fun='cv3')
#                         sum = 0
#                         for m in range(5):
#                             test_sample = np.array(f1[m])
#                             negative_sample = np.array(f2[m])
#                             virusdrug_ = virusdrug.copy()
#                             virusdrug_[test_sample[:, 0], test_sample[:, 1]] = 0
#                             test_sample_number = test_sample.shape[0]
#                             negative_sample_number = negative_sample.shape[0]
#                             label = test_sample_number*[1]+negative_sample_number*[0]
#                             label = np.array(label)
#                             sample = np.vstack((test_sample, negative_sample))
#                             VDA = GiGs(virusdrug_, drugsim, virussim, 60, param, parm1, parm2)
#                             score = VDA[sample[:, 0], sample[:, 1]]
#                             fpr, tpr, threshold = roc_curve(label,score)
#                             auc_pre = auc(fpr, tpr)
#                             sum += auc_pre
#                         AUC_pre = sum/5
#                         AUCs += AUC_pre
#                     AUCs_mean = AUCs / 20   
                    
#                     result = (param, parm1, parm2, parm3, AUCs_mean)

#                     file.write(str(result) + '\n')
#                     file.flush()



#Compute auc and aupr
auc_result,aupr_result = ROC_compute(virusdrug, drugsim, virussim, a, b)

#Store fpr and tpr of best auc
mm = 0
kk = 0

for x in range(len(auc_result)):
    mm += auc_result[x][0]
nn = mm/len(auc_result)
t_auc = np.inf
dd = None

for y in range(len(auc_result)):
    kk = abs(auc_result[y][0]-nn)
    if kk < t_auc:
        t_auc = kk
        dd = (auc_result[y][0],auc_result[y][1],auc_result[y][2])
        
auc_score, fpr, tpr = dd
with open("auc_fpr.txt", "w") as fp:
    for i in fpr:
        fp.write(str(i)+' ')
with open('auc_tpr.txt', 'w') as fp:
    for i in tpr:
        fp.write(str(i)+' ')

#Store rec and pre of best aupr
mm = 0
kk = 0
for x in range(len(aupr_result)):
    mm += aupr_result[x][0]
nn = mm/len(aupr_result)
t_auc = np.inf
dd = None
for y in range(len(aupr_result)):
    kk = abs(aupr_result[y][0]-nn)
    if kk < t_auc:
        t_auc = kk
        dd = (aupr_result[y][0],aupr_result[y][1],aupr_result[y][2])
        
aupr_score, rec, pre = dd
with open("auc_rec.txt", "w") as fp:
    for i in rec:
        fp.write(str(i)+' ')
with open('auc_pre.txt', 'w') as fp:
    for i in pre:
        fp.write(str(i)+' ')



with open("auc_fpr.txt", "r") as fp:
    GiGs_fpr = [float(x) for x in fp.read().split()]
    
with open("auc_tpr.txt", "r") as fp:
    GiGs_tpr = [float(x) for x in fp.read().split()]

plt.figure()
plt.rcParams['font.family']='Arial'
plt.rcParams['font.size']=12
plt.plot(GiGs_fpr, GiGs_tpr, color='#86D3DE', lw=2, label='GiGs_AUC={}'.format(f'{auc_score:.4f}'))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize = 12)
plt.ylabel('True Positive Rate',fontsize=12)
plt.title('ROC Curve',fontsize=12)
plt.legend(loc="lower right",fontsize=10)
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.show()


with open("aupr_rec.txt", "r") as fp:
    GiGs_rec = [float(x) for x in fp.read().split()]
    
with open("aupr_pre.txt", "r") as fp:
    GiGs_pre = [float(x) for x in fp.read().split()]

plt.figure()
plt.rcParams['font.family']='Arial'
plt.rcParams['font.size']=12
plt.plot(GiGs_rec, GiGs_pre, color='#86D3DE', lw=2, label='GiGs_AUPR={}'.format(f'{aupr_score:.4f}'))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall',fontsize = 12)
plt.ylabel('Precision',fontsize = 12)
plt.title('PR Curve',fontsize = 12)
plt.legend(loc="upper right",fontsize = 10)
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.show()
