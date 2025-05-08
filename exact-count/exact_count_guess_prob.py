import scipy.stats as stats

'''
Determines the probability of a normal distribution sample being
between -0.5 and 0.5.

If I have one column with two values, then the root has a noise value, and each child has a noise value. This constitutes 6 layers of noise, each at sd=1.
'''

# Define the mean and standard deviation
mu = 0
sigma = 1.41 # equiv sd of 2 layers at sd=1
sigma = 2.45 # equiv sd of 6 layers at sd=1

# Calculate the CDF at the points of interest
cdf_at_neg_0_5 = stats.norm.cdf(-0.5, mu, sigma)
cdf_at_0_5 = stats.norm.cdf(0.5, mu, sigma)

print("cdf_at_neg_0_5:", cdf_at_neg_0_5)
print("cdf_at_0_5:", cdf_at_0_5)

# The probability of a sample being between -0.5 and 0.5 is the difference between these CDFs
probability = cdf_at_0_5 - cdf_at_neg_0_5

# Convert to percentage
percentage = probability * 100

print(percentage)