import pandas as pd
import matplotlib.pyplot as plt

color_map = ['limegreen', 'blue', 'red', 'gold', 'pink', 'skyblue']

# Data preparation
data = {
    "Model": ["A", "B", "C", "D", "E", "X(Ours)"],
    "F1": [0.32, 0.45, 0.36, 0.21, 0.39, 0.54],
    "MOTA": [0.87, 0.88, 0.86, 0.84, 0.89, 0.89],
    "PIC": [0.291, 0.288, 0.332, 0.314, 0.306, 0.277]
}

df = pd.DataFrame(data)

# Calculating circle sizes based on (1-PIC) relative to the maximum (1-PIC) squared
_1_minus_pic = (1 - df['PIC'])
new_metric = ((_1_minus_pic-_1_minus_pic.min()) / (_1_minus_pic.max()-_1_minus_pic.min()))**2
df['Size']  = new_metric*16+4   # Adjusted size for visibility

plt.figure()

# Iterating through each row to plot
for i, row in df.iterrows():
    plt.plot(row['MOTA'], row['F1'], 'o', markersize=(row['Size']), label=row['Model'])
    plt.text(row['MOTA']+0.002, row['F1']-0.02, row['Model'], fontsize=12)

plt.xlabel('MOTA')
plt.ylabel('F1 Score')
plt.title('Model Performance Visualization')
# plt.xticks(epochs, rotation=rotation)
plt.xlim(0.83, 0.9)
plt.ylim(0.15, 0.6)
plt.grid(True, alpha=0.3)

# plt.legend()
plt.savefig("plot.png")
