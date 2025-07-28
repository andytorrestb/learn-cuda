import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Load the data from the CSV file
data = pd.read_csv('runtime_data.csv')

print(data.values)
print(data['i'])
print(data['j'])
print(data['run_time_ms'])

# Plot the data
# Alternative: Use a 3D surface plot for better visualization

# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')

# X, Y = np.meshgrid(data['i'].values, data['j'].values)

X, Y = np.meshgrid(
    sorted(data['i'].unique()),
    sorted(data['j'].unique())
)

Z = data.pivot(index='j', columns='i', values='run_time_ms').values
Z[Z <= 0] = 1e-6  # Replace zero or negative values

plt.figure(figsize=(10, 6))
im = plt.imshow(
    Z, aspect='auto', cmap='hot',
    interpolation='nearest', origin='lower',
    norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max())
)
plt.xlabel('Thread Block Size (T)')
plt.ylabel('Matrix Size (M)')
# plt.zscale('log')
# plt.zlabel('Execution Time (ms)')
plt.title('CUDA Kernel Execution Time')
# plt.imshow(data.values, aspect='auto', cmap='hot', interpolation='nearest')
plt.colorbar(label='Execution Time (ms)')
# plt.xlabel('Thread Block Size (T)')
# plt.ylabel('Matrix Size (M)')
# plt.title('CUDA Kernel Execution Time')
plt.savefig('execution_time_heatmap.png')