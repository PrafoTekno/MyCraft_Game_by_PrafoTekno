
import matplotlib.pyplot as plt
import csv
import statistics as st

latencies = []
gestures = []

with open ('latency_log.csv', newline='') as f:
    reader = csv.DictReader (f)
    for r in reader:
        latency = float (r['latency_ms'])
        latencies.append (latency)
        gestures.append (r['Jumlah jari terangkat'])

mean_latency = st.mean (latencies)
max_latency = max(latencies)
min_latency = min(latencies)
stddev_latency = st.stdev (latencies) if len(latencies) > 1 else 0

print("Latency Statistics (ms)")
print(f"Mean : {mean_latency:.2f}")
print(f"Min  : {min_latency:.2f}")
print(f"Max  : {max_latency:.2f}")
print(f"Std  : {stddev_latency:.2f}")

# === PLOT LATENCY ===
plt.figure()
plt.plot(latencies, marker='o')

plt.axhline(mean_latency,
            color='red',
            linestyle=(0, (5, 5)),
            linewidth=2,
            label=f"Mean = {mean_latency:.2f} ms")

plt.text(
    0.67, 0.65,
    f"Std Dev = {stddev_latency:.2f} ms",
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.7)
)

plt.axhline(max_latency,
            color='green',
            linestyle=':',
            linewidth=2,
            label=f"Max = {max_latency:.2f} ms")

plt.axhline(min_latency,
            color='black',
            linestyle=':',
            linewidth=2,
            label=f"Min = {min_latency:.2f} ms")

plt.xlabel("Gesture ke")
plt.ylabel("Latensi (ms)")
plt.title("Latency Measurement of Gesture-Based Interaction")
plt.legend()
plt.grid(True)
plt.show()
