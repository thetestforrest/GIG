#Gaussian interaction profile kernel similarity
import numpy as np
import pandas as pd
import math

# drugstrucsim = pd.read_excel('drug chemical structure similarity.xlsx',header=None, index_col=None)
# virusseqsim = pd.read_excel('virus sequence similarity.xlsx',header=None, index_col=None)
# drugsidesim = pd.read_excel('drug side effect similarity.xlsx',header=None, index_col=None)
# data = np.loadtxt('virus-drug associations.txt')
# virusdrug = np.zeros((drugstrucsim.shape[0], virusseqsim.shape[0]))
# for row in data:
#     x, y, value = row
#     virusdrug[int(x)-1, int(y)-1] = value

def getSimilarMatrix(IP, γ_):
    dimensional = IP.shape[0]
    sd = np.zeros(dimensional)
    K = np.zeros((dimensional, dimensional))
    for i in range(dimensional):
        sd[i] = np.linalg.norm(IP[i].astype(float)) ** 2
    #计算带宽参数γv
    gamad = γ_*dimensional / np.sum(sd.transpose())  
    for i in range(dimensional):
        for j in range(dimensional):
            K[i][j] = math.exp(-gamad * (np.linalg.norm(IP[i].astype(float) - IP[j].astype(float))) ** 2)
    return K