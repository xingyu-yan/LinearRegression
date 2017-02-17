'''
Created on December 3, 2016
@author: xingyu, at Ecole Centrale de Lille
# This programme is: Linear Regression
# reference: Coursera Machine Learning open course (Andrew Ng)
# reference: https://github.com/royshoo/mlsn
'''
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  #pandas: Python Data Analysis Library
#from mpl_toolkits.mplot3d import axes3d
from LR_package.LRfuncs import computeCost, gradientDescent

# Part 1: Feature Normalization

print("Loading Data ...\n")
npdata = pd.read_csv('ex1data1.csv')

X = npdata.ix[:,0]
y = npdata.ix[:,1]

# Part 2: Gradient descent
#Some gradient descent settings
alpha = 0.01
num_iters = 50

a_ones = np.ones((len(X),1))
X = X.values.reshape(97,1)
print(X.shape)
#print(X)

X_new = np.hstack((a_ones,X))
print('print new X: ')
#print(X_new)
print(X_new.shape)
theta = np.zeros((2,1))
theta.shape = (len(theta),1)
print('theta looks like this ')
print(theta)
X_theta = np.dot(X_new[96,:],theta)
print('X multiply theta looks like this ')
print(X_theta)

#calculate the cost by using the cost function 
J = computeCost(X_new,y,theta)
print('the final cost is: ')
print(J)

(theta,J_history) = gradientDescent(X_new,y,theta,alpha,num_iters)
#(theta) = gradientDescent(X,y,theta,alpha,num_iters)
print ("Theta computed from gradient descent:")
print (theta)
'''print ("J_history are:")
print (J_history)'''

# Part 3: Visualizing 

#1 plot the Training data and the obtained line
plt.figure(1, figsize=(5, 3.75))
plt.plot(X,y,'ro')
plt.ylabel('y')
plt.xlabel('X')
plt.plot(X, np.dot(X_new,theta))
#plt.legend('Training data', 'Linear regression')

#Predict values for population sizes of 13,000 and 17,000
predict1 = np.dot([1, 13], theta)
print('For population = 13 000, we predict a profit of %f' % predict1)
predict2 = np.dot([1, 17], theta)
print('For population = 17 000, we predict a profit of %f' % predict2)
#plot the predictions
plt.plot(13,predict1,'bx', markersize=10)
plt.plot(17,predict2,'bx', markersize=10)
#plt.show()

#2 plot J_history 
plt.figure(2, figsize=(5, 3.75))
plt.plot(J_history)
plt.ylabel('Cost over iteration')
plt.xlabel('Number of iteration')
#plt.show() 

# Part 4: Visualizing J(theta_0, theta_1)
theta0_vals = np.linspace(-10, 10, 200)
theta1_vals = np.linspace(-1, 4, 200)

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.hstack((theta0_vals[i], theta1_vals[j]))
        J_vals[i][j] = computeCost(X_new, y, t)
        
J_vals = J_vals.T
#print(J_vals)
print(J_vals.shape)
'''fig = plt.figure(3, figsize=(5, 3.75))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals)'''
#plt.show()

plt.figure(3, figsize=(5, 3.75))
CS = plt.contour(theta0_vals, theta1_vals, J_vals)
plt.clabel(CS, inline=1, fontsize=10)
plt.plot(theta[0],theta[1],'bo', markersize=10)
plt.title('Simplest default with labels')
plt.show()