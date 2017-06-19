'''
Created on May 26th 2017
@author: jbi

'''

import matplotlib.pyplot as plt
import numpy as np

from  pqueens.designers.monte_carlo_designer import MonteCarloDesigner
from  pqueens.designers.lhs_designer import LatinHyperCubeDesigner
from  example_simulator_functions.branin_hifi import branin_hifi
from  pqueens.emulators.gp_emulator import GPEmulator

import seaborn as sns
import matplotlib.pylab as plt


params =   {   "x" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : -5,
                    "max"  : 10,
                    "distribution" : 'normal',
                    "distribution_parameter" : [0,1]
                    },
                    "y" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 0,
                    "max"  : 15,
                    "distribution" : 'normal',
                    "distribution_parameter" : [0,1]
                    }
                }

num_design_points = 50
seed = 49

# create lhs design
my_lhs = LatinHyperCubeDesigner(params, seed, num_design_points)
design_points = my_lhs.get_all_samples()

# evaluate function at all design points
y = branin_hifi(design_points[:,0],design_points[:,1])

# build GP emulator based on available samples
my_emulator = GPEmulator(design_points,y.reshape(-1,1),params)

# compute mean and variance of mean using GP emulator
mean_mean, var_mean = my_emulator.compute_mean()

print("mean_mean{}".format(mean_mean))
print("var_mean{}".format(var_mean))

# compute pdf estimate using GP emulator
my_pdf, y_plot = my_emulator.compute_pdf()

fig = plt.figure(figsize=(12, 6))
line, = plt.plot(y_plot, my_pdf['mean'], lw=2, color='red')
line2, = plt.plot(y_plot, my_pdf['median'], lw=2, color='blue')
plt.fill_between(y_plot,my_pdf['quant_low'], my_pdf['quant_high'],
                 color='blue',alpha = 0.1)
plt.tight_layout()
plt.ylabel('pdf(y)')
plt.xlabel('y')
plt.legend(('mean','median','confidence region'))
fig.savefig('pdf.pdf', format='pdf',bbox_inches='tight')


#compute cdf estimate using GP emulator
my_cdf, y_plot = my_emulator.compute_cdf()
fig = plt.figure(figsize=(12, 6))
line, = plt.plot(y_plot, my_cdf, lw=2, color='red')
plt.ylabel('cdf(y)')
plt.xlabel('y')
#line2, = plt.plot(y_plot, my_cdf['median'], lw=2, color='blue')
#plt.fill_between(y_plot,my_cdf['quant_low'], my_cdf['quant_high'],color='blue',alpha = 0.1)
plt.tight_layout()
fig.savefig('cdf.pdf', format='pdf',bbox_inches='tight')


# compute failure probabilities using GP emulator
y_quantile, my_fail_prob = my_emulator.compute_failure_probability_function()
fig = plt.figure(figsize=(12, 6))
plt.ylabel('failure probability')
plt.xlabel('y_0')
line, = plt.semilogy(y_quantile['mean'],my_fail_prob, lw=2, color='red')
line, = plt.semilogy(y_quantile['quant_high'],my_fail_prob, lw=2, color='green')
line, = plt.semilogy(y_quantile['quant_low'],my_fail_prob, lw=2, color='green')
#my_min = np.min(y_quantile['quant_high'])
#my_max = np.max(y_quantile['quant_low'])
#x = np.linspace(my_min, my_max, num=len(y_quantile['quant_low']))
#plt.fill_betweenx(x,y_quantile['quant_low'],y_quantile['quant_high'],color='blue',alpha = 0.1)
#plt.fill_between(x, y_quantile['quant_low'], y_quantile['quant_high'], facecolor='green')
#plt.fill_between(x, 1, y_quantile['quant_low'])
plt.legend(('mean','quantile 0.95','quantile 0.05'))
plt.tight_layout()
fig.savefig('failure_prob.pdf', format='pdf',bbox_inches='tight')
