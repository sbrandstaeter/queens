'''
Created on May 26th 2017
@author: jbi

'''

import matplotlib.pyplot as plt
import numpy as np

from  pqueens.designers.monte_carlo_designer import MonteCarloDesigner
from  pqueens.designers.lhs_designer import LatinHyperCubeDesigner
from  pqueens.example_simulator_functions.branin_hifi import branin_hifi
from  pqueens.emulators.gp_emulator import GPEmulator
from  pqueens.utils import pdf_estimation
import matplotlib.pylab as plt


params =   {   "x" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : -5,
                    "max"  : 10,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [-5,15]
                    },
                    "y" : {
                    "type" : "FLOAT",
                    "size" : 1,
                    "min"  : 0,
                    "max"  : 15,
                    "distribution" : 'uniform',
                    "distribution_parameter" : [0,15]
                    }
                }

num_design_points = 150
seed = 49

# create lhs design
my_lhs = LatinHyperCubeDesigner(params, seed, num_design_points)
design_points = my_lhs.get_all_samples()

# evaluate function at all design points
y = branin_hifi(design_points[:,0],design_points[:,1])

# build GP emulator based on available samples
my_emulator = GPEmulator(design_points,y.reshape(-1,1),params)

################################################################################
# compute reference solution
################################################################################
my_mc_sampler = MonteCarloDesigner(params,43,1000)
test_samples = my_mc_sampler.get_all_samples()
ref_solution_sample_values = branin_hifi(test_samples[:,0],test_samples[:,1])
# compute mean
mean_ref_solution= np.mean(ref_solution_sample_values)
# compute pdf estimate
kde_bandwidth = pdf_estimation.estimate_bandwidth_for_kde(ref_solution_sample_values,
                                                          np.amin(ref_solution_sample_values),
                                                          np.amax(ref_solution_sample_values))

my_pdf_ref, y_plot_ref  = pdf_estimation.estimate_pdf(ref_solution_sample_values,
                                                      kde_bandwidth)


# sort samples in ascending order first
sample_values_ref_sorted=np.sort(ref_solution_sample_values,axis=0)

# compute cdf
cdf_ref = np.arange(1, len(sample_values_ref_sorted)+1) / \
    float(len(sample_values_ref_sorted))


# compute failure probability
my_quantiles = np.linspace(0,100,1000)
y_quantile_ref = np.percentile(ref_solution_sample_values,my_quantiles,axis=0)
my_fail_prob_ref = 1-my_quantiles/100.0


################################################################################
# compute predictions and compute rmse
################################################################################
my_predictions_mean = my_emulator.predict(test_samples)
reference_vec = np.reshape(ref_solution_sample_values,(-1,1))
error = np.linalg.norm(reference_vec - my_predictions_mean)/ \
    np.linalg.norm(reference_vec)

print("GP emulator rmse error: {}".format(error))

################################################################################
# compute mean estimate using GP emulator
################################################################################
my_mean_mean, my_mean_var = my_emulator.compute_mean()

print("Reference mean: {}".format(mean_ref_solution))
print("GP emulator mean: {}".format(my_mean_mean))

################################################################################
# compute pdf estimate using GP emulator
################################################################################

my_pdf_hifi, y_plot_hifi  = my_emulator.compute_pdf()
fig = plt.figure()
fig.suptitle('Probability Density Function', fontsize=14)
line1, = plt.plot(y_plot_hifi, my_pdf_hifi['mean'], lw=2, color='blue',label='GP mean')
line2, = plt.plot(y_plot_hifi, my_pdf_hifi['median'], lw=2, color='green',label='GP median')
plt.fill_between(y_plot_hifi,my_pdf_hifi['quant_low'], my_pdf_hifi['quant_high'],
                  color='blue',alpha = 0.3,label='GP conf. reg.')

line4, = plt.plot(y_plot_ref, my_pdf_ref, lw=2, color='red',label='MC reference')
plt.ylabel('pdf(y)')
plt.xlabel('y')
plt.legend()
plt.show()

################################################################################
# compute cdf estimate using GP emulator
################################################################################
cdf_val, sample_values = my_emulator.compute_cdf()

fig = plt.figure()
fig.suptitle('Cumulative Distribution Function', fontsize=14)
x = np.append(sample_values['q_lower_bound'],
              sample_values['q_upper_bound'][::-1])

y = np.append(cdf_val,cdf_val[::-1])

line5, = plt.plot(sample_values_ref_sorted, cdf_ref, lw=2, color='red',
                  label='MC reference')

p = plt.Polygon(np.c_[x,y], color="blue",alpha=0.3,label='GP conf. reg.')
ax = plt.gca()
ax.add_patch(p)
line1, = plt.plot(sample_values['mean'], cdf_val, lw=2, color='blue',
                  label='GP mean')
line4, = plt.plot(sample_values['median'], cdf_val, lw=2, color='green',
                  label='GP median')

plt.legend()

plt.ylabel('cdf(y)')
plt.xlabel('y')
plt.show()



################################################################################
# compute failure probability estimate using GP emulator
################################################################################
y_quantile, my_fail_prob = my_emulator.compute_failure_probability_function()
fig = plt.figure()
plt.ylabel('failure probability')
plt.xlabel('y_0')

line1, = plt.plot(y_quantile['mean'],my_fail_prob, lw=2, color='blue',label='GP mean')
line2, = plt.plot(y_quantile['median'],my_fail_prob, lw=2, color='green',label='GP median')
line3, = plt.plot(y_quantile_ref,my_fail_prob_ref, lw=2, color='red',label='MC reference')


x = np.append(y_quantile['q_upper_bound'],
              y_quantile['q_lower_bound'][::-1])

y = np.append(my_fail_prob,my_fail_prob[::-1])

p = plt.Polygon(np.c_[x,y], color="blue",alpha=0.3,label='GP conf. reg.')
ax = plt.gca()
ax.add_patch(p)

plt.legend()
ax.set_yscale('log')
plt.show()
# fig.savefig('failure_prob.pdf', format='pdf',bbox_inches='tight')
