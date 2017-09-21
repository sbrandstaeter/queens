# Bayesian Approach for Sensitivity Analysis
import numpy as np
import math
import os
import pickle
import matplotlib.pyplot as plt
from pyDOE import *
from matplotlib import pyplot as plt
from sklearn import *
from random import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
import time
from  pqueens.emulators.gp_emulator import GPEmulator


## Definition of our problem
# Dimension of our problem
d = 3
# Size of the Experimental Design Set
n = [40, 50, 60, 70, 90, 120, 150, 200]
# Number of samples for our Gaussian process
Nz = 5000
# Number of bootstrap samples
B = 300
# Number of Monte-Carlo
m = 1000
#
T1 =[];
T2 =[];
T3 =[];

# Definition of our Ishigami function
def Ishigami(x1,x2,x3):
	return np.sin(np.array(x1)) + 7*np.sin(np.array(x2))**2 + 0.1*x3**4*np.sin(np.array(x1))

# Definition of the Sobol_indice function : to calculate Sobol indices
def Sobol_indice(x1,x2,m):
	return (np.sum(x1*x2)/m-(np.sum(x1+x2)/(2*m))**2)/(sum(x1**2)/m-(np.sum(x1+x2)/(2*m))**2)

for s in n:
	# Number of Latin Hypercube Samples
	h = 1
	print(s)
	# Defintion of storage variables for our different LHS
	D_exp = np.zeros((h,s,d)) # tensor of size h x n x d
	z_exp = np.zeros((s,h)) # tensor of size n x h
	S_M_N_K_L_1 = np.zeros((h,Nz,B)) # tensor of size h x Nz x B
	S_M_N_K_L_2 = np.zeros((h,Nz,B)) # tensor of size h x Nz x B
	S_M_N_K_L_3 = np.zeros((h,Nz,B)) # tensor of size h x Nz x B

	for i in range(0,h):
		D_exp[i,:,:] = 2*math.pi*lhs(d,samples = s, criterion = 'maximin')-math.pi*np.ones((s,d))

		# Build the known values of the Ishigami function at the experimental design set
		z_exp[:,i] = Ishigami(D_exp[i,:,0], D_exp[i,:,1],D_exp[i,:,2])

		# Instanciate a Gaussian Process model
		kernel = ConstantKernel() + Matern(length_scale = [1.0,1.0,1.0], nu = 5/2) + WhiteKernel(noise_level=1e-6)
		gp = GaussianProcessRegressor(alpha=1e-10, copy_X_train=True, kernel = 3.5 + Matern(length_scale = [1.0,1.0,1.0], nu = 5/2), n_restarts_optimizer= 0, normalize_y=False, optimizer='fmin_l_bfgs_b', random_state=None)

		# Fit to data using Maximum Likelihood Estimation of the parameters
		gp.fit(D_exp[i,:,:], z_exp[:,i])

		# begin of Jonas code
		params = {}
		my_emulator = GPEmulator(D_exp[i,:,:],z_exp[:,i].reshape(-1,1),params)
		# end of Jonas Code

		# Generate two samples (xi) and (xtildei) of the random vectors (Xi) and (Xtildei) with respect to the probability measure mu
		x_1 = np.random.uniform(-math.pi,math.pi,m) # donne un vecteur ligne
		x_2 = np.random.uniform(-math.pi,math.pi,m)
		x_3 = np.random.uniform(-math.pi,math.pi,m)
		x_1_tilde = np.random.uniform(-math.pi,math.pi,m)
		x_2_tilde = np.random.uniform(-math.pi,math.pi,m)
		x_3_tilde = np.random.uniform(-math.pi,math.pi,m)

		# Generate our Bootstrap indices for bootstraping (The same bootstrap samples are used for the N_z realizations of Z_n(x)
		H = np.random.randint(m, size = (B,m))

		x = [x_1, x_2, x_3]
		x = np.asarray(x)
		x = x.reshape(3,m)
		x_tilde1 = [x_1, x_2_tilde, x_3_tilde]
		x_tilde1= np.asarray(x_tilde1)
		x_tilde1 = x_tilde1.reshape(3,m)
		x_tilde2 = [x_1_tilde, x_2, x_3_tilde]
		x_tilde2 = np.asarray(x_tilde2)
		x_tilde2 = x_tilde2.reshape(3,m)
		x_tilde3 = [x_1_tilde, x_2_tilde, x_3]
		x_tilde3 = np.asarray(x_tilde3)
		x_tilde3 = x_tilde3.reshape(3,m)

		# Sample a realization z_n(x) of Z_n(x) with x = {x, xtilde}
		time_a_scikit = time.time()
		MU = gp.sample_y(x.transpose(), n_samples=Nz, random_state= np.random.RandomState(3))
		MU1_tilde = gp.sample_y(x_tilde1.transpose(), n_samples=Nz, random_state= np.random.RandomState(3))
		MU2_tilde = gp.sample_y(x_tilde2.transpose(), n_samples=Nz, random_state= np.random.RandomState(3))
		MU3_tilde = gp.sample_y(x_tilde3.transpose(), n_samples=Nz, random_state= np.random.RandomState(3))
		time_b_scikit = time.time()
		print("Generating samples took {} secs".format(time_b_scikit-time_a_scikit))
		# try emulator
		time_a_gpy = time.time()
		MU_gpy = my_emulator.compute_posterior_samples(x.transpose())
		MU1_tilde_gpy = my_emulator.compute_posterior_samples(x_tilde1.transpose())
		MU2_tilde_gpy = my_emulator.compute_posterior_samples(x_tilde2.transpose())
		MU3_tilde_gpy = my_emulator.compute_posterior_samples(x_tilde3.transpose())
		time_b_gpy = time.time()
		print("Generating samples took {} secs with GPy".format(time_b_gpy-time_a_gpy))

		# eof try emulator
		for k in range(0,Nz):
			mu = list(zip(*MU))[k]
			mu1_tilde = list(zip(*MU1_tilde))[k]
			mu2_tilde = list(zip(*MU2_tilde))[k]
			mu3_tilde = list(zip(*MU3_tilde))[k]
			mu = np.asarray(mu)
			mu1_tilde = np.asarray(mu1_tilde)
			mu2_tilde = np.asarray(mu2_tilde)
			mu3_tilde = np.asarray(mu3_tilde)
			S_M_N_K_L_1[i,k,0] = Sobol_indice(mu, mu1_tilde,m)
			S_M_N_K_L_2[i,k,0] = Sobol_indice(mu, mu2_tilde,m)
			S_M_N_K_L_3[i,k,0] = Sobol_indice(mu, mu3_tilde,m)

			# The same bootstrap samples are used for the N_z realizations of Z_n(x)
			for l in range(1,B):
				mub = mu[H[l,:]]
				mu1_tildeb = mu1_tilde[H[l,:]]
				mu2_tildeb = mu2_tilde[H[l,:]]
				mu3_tildeb = mu3_tilde[H[l,:]]
				S_M_N_K_L_1[i,k,l] =  Sobol_indice(mub, mu1_tildeb,m)
				S_M_N_K_L_2[i,k,l] =  Sobol_indice(mub, mu2_tildeb,m)
				S_M_N_K_L_3[i,k,l] =  Sobol_indice(mub, mu3_tildeb,m)

	# Calculation of the sensitivity indices
	S1 = np.mean(np.mean(S_M_N_K_L_1,axis=0))
	S2 = np.mean(np.mean(S_M_N_K_L_2,axis=0))
	S3 = np.mean(np.mean(S_M_N_K_L_3,axis=0))

	# Evaluation of the variance
	Sigma1 = np.var(np.reshape(np.mean(S_M_N_K_L_1,axis=0),(1,Nz*B)))
	Sigma2 = np.var(np.reshape(np.mean(S_M_N_K_L_2,axis=0),(1,Nz*B)))
	Sigma3 = np.var(np.reshape(np.mean(S_M_N_K_L_3,axis=0),(1,Nz*B)))

	# Evaluation of the variance due to the metamodel
	SigmaMM1 = np.mean(np.var(np.mean(S_M_N_K_L_1,axis=0),axis=0))
	SigmaMM2 = np.mean(np.var(np.mean(S_M_N_K_L_2,axis=0),axis=0))
	SigmaMM3 = np.mean(np.var(np.mean(S_M_N_K_L_3,axis=0),axis=0))
	# Evaluation of the variance due to the Monte-Carlo integration
	SigmaMC1 = np.mean(np.var(np.mean(S_M_N_K_L_1,axis=0),axis=1))
	SigmaMC2 = np.mean(np.var(np.mean(S_M_N_K_L_2,axis=0),axis=1))
	SigmaMC3 = np.mean(np.var(np.mean(S_M_N_K_L_3,axis=0),axis=1))

	# Evaluation of the 0.05 and 0.95 quantiles with a bootstrap procedure
	# Number of bootstrap procedure N
	Z1 = np.reshape(np.mean(S_M_N_K_L_1,axis=0),(1,Nz*B))
	Z1 = Z1.transpose()
	Z2 = np.reshape(np.mean(S_M_N_K_L_2,axis=0),(1,Nz*B))
	Z2 = Z2.transpose()
	Z3 = np.reshape(np.mean(S_M_N_K_L_3,axis=0),(1,Nz*B))
	Z3 = Z3.transpose()
	Q1 = np.zeros((N,2), dtype = float)
	Q2 = np.zeros((N,2), dtype = float)
	Q3 = np.zeros((N,2), dtype = float)
	h = np.random.randint(Nz*B, size = (B,Nz*B))
	Zq1 = Z1[h]
	Zq2 = Z2[h]
	Zq3 = Z3[h]
	q11 =  np.percentile(Zq1, 2.5)
	q12 =  np.percentile(Zq1, 97.5)
	q21 =  np.percentile(Zq2, 2.5)
	q22 =  np.percentile(Zq2, 97.5)
	q31 =  np.percentile(Zq3, 2.5)
	q32 =  np.percentile(Zq3, 97.5)
	Q1[:,0] = q11
	Q1[:,1] = q12
	Q2[:,0] = q21
	Q2[:,1] = q22
	Q3[:,0] = q31
	Q3[:,1] = q32

	Q_11 = np.mean(Q1[:,0])
	Q_12 = np.mean(Q1[:,1])
	Q_21 = np.mean(Q2[:,0])
	Q_22 = np.mean(Q2[:,1])
	Q_31 = np.mean(Q3[:,0])
	Q_32 = np.mean(Q3[:,1])

	resultsS1 = [Q_11, S1, Q_12, SigmaMM1, SigmaMC1, Sigma1, abs((Sigma1 - (SigmaMM1+SigmaMC1))/Sigma1*100)]
	resultsS2 = [Q_21, S2, Q_22, SigmaMM2, SigmaMC2, Sigma2, abs((Sigma2 - (SigmaMM2+SigmaMC2))/Sigma2*100)]
	resultsS3 = [Q_31, S3, Q_32, SigmaMM3, SigmaMC3, Sigma3, abs((Sigma3 - (SigmaMM3+SigmaMC3))/Sigma3*100)]

	T1 = np.concatenate((T1, resultsS1))
	T2 = np.concatenate((T2, resultsS2))
	T3 = np.concatenate((T3, resultsS3))

T1 = np.reshape(np.asarray(T1), (len(n),7))
T2 = np.reshape(np.asarray(T2), (len(n),7))
T3 = np.reshape(np.asarray(T3), (len(n),7))

# Plot of the results
plt.figure(1)
plt.plot(np.asarray(n), T1[:,0].transpose(),'b--',label="0.05 quantile")
plt.plot(np.asarray(n), T1[:,1].transpose(), 'r--', label="mean")
plt.plot(np.asarray(n), T1[:,2].transpose(), 'g--', label="0.95 quantile")
plt.plot(np.asarray(n), 0.314*np.ones((1,len(n)), dtype = np.float64).transpose(), 'y--', label="Exact Value")
plt.title('Sensitivity index S1')
plt.xlabel('Size of the Experimental Design Set')
plt.ylabel('Values of the Sensitivity Indices')
plt.legend()
plt.show()
plt.figure(2)
plt.plot(np.asarray(n),T2[:,0].transpose(),'b--',label="0.05 quantile")
plt.plot(np.asarray(n), T2[:,1].transpose(), 'r--', label="mean")
plt.plot(np.asarray(n), T2[:,2].transpose(), 'g--', label="0.95 quantile")
plt.plot(np.asarray(n), 0.442*np.ones((1,len(n)), dtype = np.float64).transpose(), 'y--', label="Exact Value")
plt.title('Sensitivity index S2')
plt.xlabel('Size of the Experimental Design Set')
plt.ylabel('Values of the Sensitivity Indices')
plt.legend()
plt.show()
plt.figure(3)
plt.plot(np.asarray(n),T3[:,0].transpose(),'b--',label="0.05 quantile")
plt.plot(np.asarray(n), T3[:,1].transpose(), 'r--', label="mean")
plt.plot(np.asarray(n), T3[:,2].transpose(), 'g--', label="0.95 quantile")
plt.plot(np.asarray(n), np.zeros((1,len(n)), dtype = np.float64).transpose(), 'y--', label="Exact Value")
plt.title('Sensitivity index S3')
plt.xlabel('Size of the Experimental Design Set')
plt.ylabel('Values of the Sensitivity Indices')
plt.legend()
plt.show()

plt.figure(4)
plt.plot(np.asarray(n),T1[:,3].transpose(),'b--',label="Meta Model")
plt.plot(np.asarray(n), T1[:,4].transpose(),'r--',label="Monte Carlo")
plt.plot(np.asarray(n), T1[:,5].transpose(),'g--',label="Total Variance" )
plt.plot(np.asarray(n), (T1[:,3]+T1[:,4]).transpose(),'y--',label="Sum of MM and MC" )
plt.title('Variance of S1')
plt.xlabel('Size of the Experimental Design Set')
plt.ylabel('Values of the Sensitivity Indices')
plt.legend()
plt.show()
plt.figure(5)
plt.plot(np.asarray(n),T2[:,3].transpose(),'b--',label="Meta Model")
plt.plot(np.asarray(n), T2[:,4].transpose(),'r--',label="Monte Carlo")
plt.plot(np.asarray(n), T2[:,5].transpose(),'g--',label="Total Variance" )
plt.plot(np.asarray(n), (T2[:,3]+T2[:,4]).transpose(),'y--',label="Sum of MM and MC" )
plt.title('Variance of S2')
plt.xlabel('Size of the Experimental Design Set')
plt.ylabel('Values of the Sensitivity Indices')
plt.legend()
plt.show()
plt.figure(6)
plt.plot(np.asarray(n),T3[:,3].transpose(),'b--',label="Meta Model")
plt.plot(np.asarray(n), T3[:,4].transpose(),'r--',label="Monte Carlo")
plt.plot(np.asarray(n), T3[:,5].transpose(),'g--',label="Total Variance" )
plt.plot(np.asarray(n), (T3[:,3]+T3[:,4]).transpose(),'y--',label="Sum of MM and MC" )
plt.title('Variance of S3')
plt.xlabel('Size of the Experimental Design Set')
plt.ylabel('Values of the Sensitivity Indices')
plt.legend()
plt.show()
