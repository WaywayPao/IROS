import pandas as pd
import matplotlib.pyplot as plt

color_map = ['springgreen', 'red', 'greenyellow', 'dimgray', 'brown',
             'gold', 'purple', 'deepskyblue', 'tomato', 'pink']

data = {
    "Model": ["Random", "Range (10m)", "DSA", "RRL", "BP", "BCP", "PF (BCP)", "PF (FDE)", "PF (ADE)"],
    "F1": [16.01, 53.47, 28.97, 23.53, 22.50, 36.49, 50.8, 52.97, 54.4],
    "wMOTA": [1.2, 78.6, 53.3, 52.3, 54.5, 59.0, 69.3, 64.91, 65.5],
    "PIC": [0.500, 0.287, 0.298, 0.289, 0.340, 0.292, 0.293, 0.258, 0.253],
    
    # TODO
    "Precision": [9.53, 43.62, 54.72, 49.44, 36.35, 48.69, 53.7, 51.49, 55.4],
    "Recall": [49.97, 69.06, 19.70, 15.44, 16.29, 29.18, 48.2, 54.54, 53.5],
    # "PIC-1s": [137.6, 87.8, 114.0, 115.6, 132.3, 112.2, 80.0, 92.6, 97.7],
    # "PIC-2s": [258.3, 158.2, 177.8, 175.4, 202.7, 173.0, 126.9, 161.0, 149.1],
    # "PIC-3s": [352.2, 202.0, 210.0, 203.4, 239.7, 206.0, 154.0, 205.7, 179.3],
    # "F1-1s": [27.50, 63.33, 36.77, 34.98, 24.69, 41.76, 47.2, 50.8, 56.3],
    # "F1-2s": [16.02, 45.16, 18.01, 24.82, 23.48, 36.80, 42.8, 48.5, 54.2],
    # "F1-3s": [6.71, 22.01, 6.18, 27.55, 15.97, 16.32, 40.0, 46.2, 52.2],
    "IDsw rate": [47.79, 0.54, 2.67, 2.14, 1.21, 1.66, 1.3, 4.73, 4.2],
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

    if row['Model'] == "Range (10m)":
        continue
    
    F1 = row['F1']/100.
    MOTA = row['wMOTA']/100.

    # print(row['Model'], MOTA, F1)
    pt = ax.plot(F1, MOTA, 'o', markersize=(row['Size']), label=row['Model'], color=color_map[i])
    if row['Model'] == "PF (ADE)":
        ax.text(F1+0.015, MOTA+0.005, row['Model'], fontsize=10)
    elif row['Model'] == "PF (FDE)":
        ax.text(F1-0.05, MOTA-0.015, row['Model'], fontsize=10)
    elif row['Model'] == "Range (10m)":
        ax.text(F1+0.003, MOTA-0.02, row['Model'], fontsize=10)
    # elif row['Model'] == "RRL":
    #     ax.text(F1+0.003, MOTA-0.015, row['Model'], fontsize=10)
    else:
        ax.text(F1+0.005, MOTA-0.01, row['Model'], fontsize=10)


# Hide the right and top spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


ax.set_xlabel('F1 Score')
ax.set_ylabel('wMOTA')
ax.set_title('Vision-based ROI Performance Visualization')
# plt.xticks(epochs, rotation=rotation)
ax.set_xlim(0.2, 0.6)
ax.set_ylim(0.50, 0.7)
ax.grid(True, alpha=0.2)
# plt.legend()

ax.figure.savefig("plot.png")
