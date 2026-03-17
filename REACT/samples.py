# simulate sampling from a distribution, suppose you have samples 12, 15, 12, 12, 11, 12

import numpy as np
import matplotlib.pyplot as plt

# the samples
samples = [12, 15, 12, 12, 11, 12, 17, 20, 12, 11, 12, 11, 15, 12, 23, 11, 12, 13, 23, 12, 13, 13, 12, 12, 12, 12, 12, 13, 12, 16]

#plot the histogram of the samples
plt.hist(samples, bins=30, color='c', alpha=0.75)
plt.show()

# compute mean of the samples, and SD
mean = np.mean(samples)
std = np.std(samples)

print(mean, std)

# if this was a poisson distribution, we would have mean and sd of 12, and 3.46 
distribution = np.random.poisson(12 , 1000)

# plot the distribution
plt.hist(distribution, bins=30, color='c', alpha=0.75)
plt.show()

# compute mean of this distributiom, and SD
mean = np.mean(distribution)
std = np.std(distribution)

print("Poisson:", mean, std)