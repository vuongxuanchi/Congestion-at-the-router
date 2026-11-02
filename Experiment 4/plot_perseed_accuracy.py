%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

seeds = np.arange(10)

ns3_cnn  = [0.8868, 0.8854, 0.8817, 0.8853, 0.8872, 0.8845, 0.8850, 0.8825, 0.8868, 0.8849]
ns3_wcnn = [0.8920, 0.8919, 0.8915, 0.8920, 0.8898, 0.8930, 0.8904, 0.8917, 0.8915, 0.8929]

ukm_cnn  = [0.9592, 0.9750, 0.9950, 0.8834, 0.9808, 0.9567, 0.9800, 0.9883, 0.9717, 0.9750]
ukm_wcnn = [0.9684, 0.9775, 0.9875, 0.9742, 0.9442, 0.9742, 0.9917, 0.9833, 0.9226, 0.9875]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
width = 0.35

for ax, (cnn, wcnn, title, ylim) in zip(
    axes,
    [(ns3_cnn, ns3_wcnn, "NS-3", (0.85, 0.90)),
     (ukm_cnn, ukm_wcnn, "UKMDDoSDN", (0.85, 1.0))]
):
    ax.bar(seeds - width/2, cnn, width, label="CNN (ReLU)", color="tab:blue")
    ax.bar(seeds + width/2, wcnn, width, label="WCNN (Wav-KAN)", color="tab:red")
    cnn_mean, wcnn_mean = np.mean(cnn), np.mean(wcnn)
    ax.axhline(cnn_mean, color="tab:blue", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(wcnn_mean, color="tab:red", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(f"Độ chính xác kiểm thử theo seed - {title}", fontweight="bold")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Test Accuracy")
    ax.set_xticks(seeds)
    ax.set_ylim(*ylim)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

print("NS3  CNN  mean=%.4f std=%.4f" % (np.mean(ns3_cnn), np.std(ns3_cnn)))
print("NS3  WCNN mean=%.4f std=%.4f" % (np.mean(ns3_wcnn), np.std(ns3_wcnn)))
print("UKM  CNN  mean=%.4f std=%.4f" % (np.mean(ukm_cnn), np.std(ukm_cnn)))
print("UKM  WCNN mean=%.4f std=%.4f" % (np.mean(ukm_wcnn), np.std(ukm_wcnn)))
