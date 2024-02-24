import pandas as pd
import matplotlib.pyplot as plt

color_map = ['limegreen', 'blue', 'red', 'gold', 'pink',
             'skyblue', 'purple', 'gray', 'brown']

data = {
    "Model": ["Random", "Range", "DSA", "RRL", "BP", "BCP", "PF (FD)", "PF (RS)", "PF (FDE)", "PF (π)"],
    "F1": [16.01, 53.47, 28.97, 23.53, 22.50, 36.49, 58.43, 54.58, 64.05, 63.36],
    "MOTA": [2.12, 87.97, 85.52, 85.58, 87.66, 88.27, 89.90, 87.92, 91.37, 91.52],
    "PIC": [0.500, 0.287, 0.298, 0.289, 0.340, 0.292, 0.219, 0.292, 0.255, 0.240],
    
    "Precision": [9.53, 43.62, 54.72, 49.44, 36.35, 48.69, 53.83, 47.50, 63.54, 66.58],
    "Recall": [49.97, 69.06, 19.70, 15.44, 16.29, 29.18, 63.90, 64.15, 64.57, 60.44],
    "PIC-1s": [137.6, 87.8, 114.0, 115.6, 132.3, 112.2, 80.0, 92.6, 97.7, 91.0],
    "PIC-2s": [258.3, 158.2, 177.8, 175.4, 202.7, 173.0, 126.9, 161.0, 149.1, 141.7],
    "PIC-3s": [352.2, 202.0, 210.0, 203.4, 239.7, 206.0, 154.0, 205.7, 179.3, 169.4],
    "F1-1s": [27.50, 63.33, 36.77, 34.98, 24.69, 41.76, 67.32, 62.65, 59.13, 59.99],
    "F1-2s": [16.02, 45.16, 18.01, 24.82, 23.48, 36.80, 55.86, 42.69, 55.99, 56.39],
    "F1-3s": [6.71, 22.01, 6.18, 27.55, 15.97, 16.32, 37.73, 14.97, 42.62, 46.96],
    "IDsw rate": [47.79, 0.54, 2.67, 2.14, 1.21, 1.66, 0.85, 1.26, 1.28, 1.39],
}

df = pd.DataFrame(data)
df = df.drop([0])

# normalize PIC
_1_minus_pic = (1 - df['PIC'])
new_metric = ((_1_minus_pic-_1_minus_pic.min()) / (_1_minus_pic.max()-_1_minus_pic.min()))**2
df['Size']  = new_metric*20+4

print(df)

# plt.figure()
ax = plt.subplot(111)

# Iterating through each row to plot
for i, row in df.iterrows():
    MOTA = row['MOTA']/100.
    F1 = row['F1']/100.
    # print(row['Model'], MOTA, F1)
    pt = ax.plot(MOTA, F1, 'o', markersize=(row['Size']), label=row['Model'])
    ax.text(MOTA+0.002, F1-0.02, row['Model'], fontsize=10)

# Hide the right and top spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


ax.set_xlabel('MOTA')
ax.set_ylabel('F1 Score')
ax.set_title('Vision-based ROI Performance Visualization')
# plt.xticks(epochs, rotation=rotation)
ax.set_xlim(0.85, 0.92)
ax.set_ylim(0.21, 0.65)
ax.grid(True, alpha=0.2)
# plt.legend()

ax.figure.savefig("plot.png")
