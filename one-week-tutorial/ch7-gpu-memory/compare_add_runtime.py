import pandas as pd
import matplotlib.pyplot as plt

runtime = pd.read_csv('vector_add_runtime.csv')


# Plotting GPU and CPU runtimes
plt.figure(figsize=(10, 5))
plt.plot(runtime['Vector Size'], runtime['GPU Time (ms)'], label='GPU Time', marker='o')
plt.plot(runtime['Vector Size'], runtime['CPU Time (ms)'], label='CPU Time', marker='x')
plt.xlabel('Vector Size')
plt.ylabel('Time (ms)')
plt.xscale('log')  # Use logarithmic scale for better visibility
plt.xticks(runtime['Vector Size'], rotation=45)
plt.yscale('log')  # Use logarithmic scale for better visibility
plt.title('GPU vs CPU Runtime')
plt.legend()
plt.grid()
plt.savefig('gpu_vs_cpu_runtime.png')
plt.clf()

speedup = runtime['GPU Time (ms)'] / runtime['CPU Time (ms)']

plt.figure(figsize=(10, 5))
plt.plot(runtime['Vector Size'], speedup, label='Speedup', marker='o')
plt.xlabel('Vector Size')
plt.ylabel('Speedup (GPU/CPU)')
plt.title('Speedup of GPU over CPU')
plt.grid()
plt.savefig('gpu_speedup.png')
plt.clf()