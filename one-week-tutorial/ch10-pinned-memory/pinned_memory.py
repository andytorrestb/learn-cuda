import pandas as pd
import matplotlib.pyplot as plt

runtime = pd.read_csv('pinned_memory_runtime.csv')


# Plotting GPU and CPU runtimes
plt.figure(figsize=(10, 5))
plt.plot(runtime['Vector Size'], runtime['Pinned Memory Time (ms)'], label='Pinned Memory Time', marker='o')
plt.plot(runtime['Vector Size'], runtime['Paged Memory Time (ms)'], label='Paged Memory Time', marker='x')
plt.xlabel('Vector Size')
plt.ylabel('Time (ms)')
# plt.xscale('log')  # Use logarithmic scale for better visibility
# plt.xticks(runtime['Vector Size'], rotation=45)
# plt.yscale('log')  # Use logarithmic scale for better visibility
plt.title('Pinned vs Paged Memory Runtime')
plt.legend()
plt.grid()
plt.savefig('pinned_vs_paged_memory_runtime.png')
plt.clf()

speedup = runtime['Paged Memory Time (ms)'] / runtime['Pinned Memory Time (ms)']

plt.figure(figsize=(10, 5))
plt.plot(runtime['Vector Size'], speedup, label='Speedup', marker='o')
plt.xlabel('Vector Size')
plt.ylabel('Speedup (Paged/ Pinned)')
plt.title('Speedup of Paged over Pinned Memory')
plt.grid()
plt.savefig('pinned_memory_speedup.png')
plt.clf()

# Repeat plotting after dropping the last 17 rows.
# This is meant to highlight the vector size at
# which the pinned memory starts to outperform paged memory.
runtime = runtime.iloc[:10]

print(runtime)

plt.figure(figsize=(10, 5))
plt.plot(runtime['Vector Size'], runtime['Pinned Memory Time (ms)'], label='Pinned Memory Time', marker='o')
plt.plot(runtime['Vector Size'], runtime['Paged Memory Time (ms)'], label='Paged Memory Time', marker='x')
plt.xlabel('Vector Size')
plt.ylabel('Time (ms)')
plt.title('Pinned vs Paged Memory Runtime')
plt.legend()
plt.grid()
plt.savefig('pinned_vs_paged_memory_runtime_dropped.png')
plt.clf()

speedup = runtime['Paged Memory Time (ms)'] / runtime['Pinned Memory Time (ms)']

plt.figure(figsize=(10, 5))
plt.plot(runtime['Vector Size'], speedup, label='Speedup', marker='o')
plt.xlabel('Vector Size')
plt.ylabel('Speedup (Paged/ Pinned)')
plt.title('Speedup of Paged over Pinned Memory')
plt.grid()
plt.savefig('pinned_memory_speedup_dropped.png')
plt.clf()