import pandas as pd
import csv
import numpy as np
import sys
import random
import math

def Lasso(lamb,sparity,index,theta_init,theta,y,Z):
    epsilon = 0.00000001
    XXt = np.dot(X.T,X)
    XXt *= 2
    Z *= lamb
    left = XXt + Z
    left = np.linalg.inv(left)
    right = np.dot(X.T,y)
    theta_new = np.dot(left,right)
    theta_new *= 2  
    count1 = 0
    for i in range(0,199):
        if np.absolute(theta_new[i] - theta[i]) > 0.0001:
            count1+=1
    if(count1 != 0):
        for i in range(0,199):
            temp = theta_new[i]**2 + epsilon
            #temp = int(temp)
            Z[i][i] = (1 / math.sqrt(temp))
        Lasso(lamb,sparity,index+1,theta_init,theta_new,y,Z)
    else:
        error = 0
        #temp = np.dot(X,theta_new)
        for i in range(0,99):
            temp1 = np.absolute(theta_init[i]-theta_new[i])
            error += temp1*temp1
        print "error is: %f" % math.sqrt(error)
        print "no of iteration is: %d" % index
        print "done"
    

if __name__ == '__main__':
    X = np.random.randn(100,200)
    for i in range(0,200):
        norm = np.linalg.norm(X[:,i])
        X[:,i] = X[:,i] / norm
    print X.shape
    Z_init = np.identity(200)
    sparsity = int(sys.argv[2])
    sparsity *= 2
    theta_init = np.zeros((200,1))
    for i in range(0,sparsity):
        r = np.random.randint(0,199)
        while theta_init[r][0]!=0:
            r = np.random.randint(0,199)
        theta_init[r][0] = np.random.uniform(0.1,1.0)
    y = np.dot(X,theta_init)
    Lasso(float(sys.argv[1]),int(sys.argv[2]),1,theta_init,theta_init,y,Z_init)