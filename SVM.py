import pandas as pd
import csv
import numpy as np
import sys
import random

def LLS_theta(a,b):
	raw_data = pd.read_csv('IrisDataset.csv')
	filtered_data = raw_data[[0,1,2,3]]

	#Training
	
	k=(b/100.0)*150
	k=int(k)
	random_row_generator1 = random.sample(range(50), k/3)
	random_row_generator2 = random.sample(range(50), k/3) 
	random_row_generator3 = random.sample(range(50), k/3)	
	
	x_class1 = filtered_data[0:50]
	x_class2 = filtered_data[50:100]
	x_class3 = filtered_data[100:150]
	
	x_class1_training=pd.DataFrame(x_class1.values[random_row_generator1])
	x_class2_training=pd.DataFrame(x_class2.values[random_row_generator2])
	x_class3_training=pd.DataFrame(x_class3.values[random_row_generator3])

	
	x_without_bias = pd.concat([x_class1_training, x_class2_training, x_class3_training])
	x_without_bias = np.asarray(x_without_bias)

	z = np.ones((k,1))

	# Dimension of x : 15  * 5
	x = np.append(x_without_bias,z,axis=1)

	# Dimension of xt_x : 5 * 5
	xt_x=np.dot(x.T,x)

	lamda = a

	# Dimension of temp :5*5
	temp= np.identity(5)*lamda

	z = xt_x+ temp

	z_inv = np.linalg.inv(z)

	label1 = np.matrix('1 0 0')
	label2 = np.matrix('0 1 0')
	label3 = np.matrix('0 0 1')

	class_label1_training = np.repeat(label1,k/3, axis=0)
	class_label2_training = np.repeat(label2,k/3, axis=0)
	class_label3_training = np.repeat(label3,k/3, axis=0)

	y = np.concatenate([class_label1_training, class_label2_training, class_label3_training])

	xt_y=np.dot(x.T,y)

	theta = np.dot(z_inv,xt_y)
	
	return theta

def svm(a,b):
	theta = LLS_theta(a,b)
	raw_data = pd.read_csv('IrisDataset.csv')
	filtered_data = raw_data[[0,1,2,3]]

	#Training
	
	k=(b/100.0)*150
	k=int(k)
	random_row_generator1 = random.sample(range(50), k/3)
	random_row_generator2 = random.sample(range(50), k/3) 
	random_row_generator3 = random.sample(range(50), k/3)	
	
	x_class1 = filtered_data[0:50]
	x_class2 = filtered_data[50:100]
	x_class3 = filtered_data[100:150]
	
	x_class1_training=pd.DataFrame(x_class1.values[random_row_generator1])
	x_class2_training=pd.DataFrame(x_class2.values[random_row_generator2])
	x_class3_training=pd.DataFrame(x_class3.values[random_row_generator3])

	
	x_without_bias = pd.concat([x_class1_training, x_class2_training, x_class3_training])
	x_without_bias = np.asarray(x_without_bias)

	z = np.ones((k,1))

	# Dimension of x : 15  * 5
	x = np.append(x_without_bias,z,axis=1)

	label1 = np.matrix('1 0 0')
	label2 = np.matrix('0 1 0')
	label3 = np.matrix('0 0 1')

	class_label1_training = np.repeat(label1,k/3, axis=0)
	class_label2_training = np.repeat(label2,k/3, axis=0)
	class_label3_training = np.repeat(label3,k/3, axis=0)
        #print x.shape
        #print theta.shape
	y = np.concatenate([class_label1_training, class_label2_training, class_label3_training])
        #z = np.dot(y,np.dot(theta.T,x.T))
        #print z.shape
        
	#z1 = np.identity(15) - np.dot(y,np.dot(theta.T,x.T))
	#print z1.shape
	#print np.diag(z1)
	
	zn = []
	
	for i in range(0,k):
	       yn = np.asmatrix(y[i])
	       xn = np.asmatrix(x[i])
	       z2 = 1 - np.dot(yn,np.dot(theta.T,xn.T))
	       zn.append(z2.item(0,0))
	 
	z = np.absolute(zn).T
	z = np.asmatrix(z).T
	#print z
	#print z.shape
	
	epsilon = 0.00000001
	
	xx = np.zeros((5,5))
        yx = np.zeros((5,3))
	  	
	for i in range(0,k):
	   xn = np.asmatrix(x[i]).T
	   #print xn.shape
	   zn = np.maximum(epsilon,np.asmatrix(z.item(i,0)))
	   #print zn
	   xxn = np.dot(xn,xn.T)
	   #print xxn.shape
	   xxn /= (2*zn)
	   #print xxn.shape
	   xx = np.add(xx,xxn)
	   
	   yn = np.asmatrix(y[i])
	   #print yn.shape
	   yxn = np.dot(xn,yn)
	   
	   coeff = (1+zn)/(2*zn)
	   yxn *= coeff.item(0,0)
	   
	   #print yxn.shape
	   yx = np.add(yx,yxn)   
	
	C = 100000
	xx *= C
	
	#print xx
	#print xx.shape
	#print yx
	#print yx.shape
	
	D = np.zeros((5,5))
	D[0,0] = D[1,1] = D[2,2] = D[3,3] = 1
	#print D
	
	left = np.add(D,xx)
	#print left.shape
	
	theta_new = np.dot(np.linalg.inv(left),yx)
	theta_new *= C
	
	print theta_new
	   
	test_data = raw_data[[0,1,2,3]]

	bias = np.ones((150,1))

	final_test_data = np.append(test_data,bias,axis=1)
	
	class_labels_real = np.dot(final_test_data,theta_new)
	
	class_labels_real = np.asarray(class_labels_real)
	
	class_labels_predicted = (class_labels_real == class_labels_real.max(axis=1)[:, None]).astype(int)
	#print class_labels_predicted
	
	class_label1_test = np.repeat(label1,50, axis=0)
	class_label2_test = np.repeat(label2,50, axis=0)
	class_label3_test = np.repeat(label3,50, axis=0)
	class_labels_given = np.concatenate([class_label1_test, class_label2_test, class_label3_test])

	misclassification = (class_labels_given!=class_labels_predicted).sum()/2.0

	misclassification_error = (misclassification/150)*100
	
	print misclassification_error

if __name__ == '__main__':
    svm(float(sys.argv[1]),int(sys.argv[2]))
