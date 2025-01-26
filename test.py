import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# Example data
data1 = np.random.randint(0, 256, 1000)  # Random data, replace with your actual data
data2 = np.random.randint(0, 256, 1000)  # Random data, replace with your actual data

# Compute histograms
hist1, bins1 = np.histogram(data1, bins=256, range=(0, 255), density=False)
hist2, bins2 = np.histogram(data2, bins=256, range=(0, 255), density=False)

# Ensure histograms have the same bin edges (align bins)
# Since `np.histogram` generates bin edges, the bins must match for comparison.
# If they don't match, we can re-align them by trimming or zero-padding.

# Make sure histograms have the same length
assert len(hist1) == len(hist2), "Histograms do not have the same number of bins."

# Handle zero values in histograms by adding a small value (pseudo-count) to avoid division by zero
epsilon = 1  # A small constant to prevent zero frequencies
hist1 = hist1 + epsilon
hist2 = hist2 + epsilon

# Perform chi-square test
chi2_stat, p_val = chisquare(hist1, hist2)

print(f"Chi-square statistic: {chi2_stat}")
print(f"P-value: {p_val}")

# If you want to visualize the histograms
plt.figure(figsize=(10, 5))
plt.hist(data1, bins=256, alpha=0.5, label="Data1", density=True)
plt.hist(data2, bins=256, alpha=0.5, label="Data2", density=True)
plt.legend()
plt.title("Comparison of Histograms")
plt.show()
