from scipy.stats import norm
from scipy.stats import lognorm
import scipy
import numpy as np
import numpy.random
import pqueens

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from  pqueens.randomfields.univariate_random_field_generator import UnivariateRandomFieldSimulator
from  pqueens.randomfields.random_field_gen_fourier import RandomFieldGenFourier
from  pqueens.randomfields.random_field_gen_fourier_1d import RandomFieldGenFourier1D
from  pqueens.randomfields.univariate_field_generator_factory import UniVarRandomFieldGeneratorFactory

my_distribution = norm(0,1)

#if type(my_distribution) != scipy.stats.norm:
#    print("test")
#    print(my_distribution.dist.name)
my_field_generator = UnivariateRandomFieldSimulator(my_distribution)
print(my_field_generator.get_stoch_dim())


my_field_generator = RandomFieldGenFourier(my_distribution,0.5,0.95,np.array([1,20,1,40,1,50]),3,120,12)

my_field_generator = RandomFieldGenFourier1D(my_distribution,0.5,0.95,np.array([1,20]),120,12)

my_stoch_dim = my_field_generator.get_stoch_dim
print(my_stoch_dim)
loc = np.array([1,2.3,23,])
xi = np.random.randn(240,1)
#print(xi)
my_vals = my_field_generator.evaluate_field_at_location(loc,xi)

print(my_vals)



# test factory
my_field_generator = UniVarRandomFieldGeneratorFactory.create_new_random_field_generator(my_distribution,1,'squared_exp',0.5,0.95,np.array([1,20]),40,120)
my_field_generator = UniVarRandomFieldGeneratorFactory.create_new_random_field_generator(my_distribution,2,'squared_exp',5,0.95,np.array([1,20,1,20]),60,320)

# Make data
u = np.linspace(1, 20, 100)
v = np.linspace(1, 20, 100)
x ,y = np.meshgrid(u, v)



my_stoch_dim = my_field_generator.get_stoch_dim()

x_test = np.reshape(x, (-1,1))
y_test = np.reshape(y, (-1,1))
print(x.shape)
loc = np.transpose(np.vstack((np.transpose(x_test),np.transpose(y_test))))
print('loc '+ str(loc.shape))

xi = np.random.randn(1280,1)
#print(xi)
my_vals = my_field_generator.evaluate_field_at_location(loc,xi)
print(my_vals.shape)
my_vals_test = np.reshape(my_vals, (100,100))
matplotlib.use('agg')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, my_vals_test,cmap=cm.jet)
print(x)
plt.savefig('radnom_field.svg', format='svg')
#plt.show()
