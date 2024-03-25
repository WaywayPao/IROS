import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

color_map = ['springgreen', 'red', 'greenyellow', 'dimgray', 'brown',
             'gold', 'purple', 'deepskyblue', 'tomato', 'pink']


"""
    org : PIC weight = np.exp(c*-(int(end_frame)-int(frame_id)))

    new : PIC weight = np.exp(c*-(int(end_frame)-int(frame_id))/60)			
"""
org_PIC_w_list = []
new_PIC_w_list = []
T = 60
c = 1.0

for t in range(T, 0, -1):

    org_PIC_w = np.exp(c*-(int(T)-int(t)))
    new_PIC_w = np.exp(c*-(int(T)-int(t))/T)

    # print(f"T-t={int(T)-int(t):2d}\tnew_PIC_w: {new_PIC_w:3.8f},\torg_PIC_w: {org_PIC_w:3.8f}")

    org_PIC_w_list.append(org_PIC_w)
    new_PIC_w_list.append(new_PIC_w)

T_list = list(range(1, T+1))

ax = plt.subplot(111)

pt = ax.plot(T_list, org_PIC_w_list, label="org PIC", color="red")
pt = ax.plot(T_list, new_PIC_w_list, label="new PIC", color="blue")
# ax.plot([], [], ' ', label="T:Critical Frame No.")

# Hide the right and top spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('T-t')
ax.set_ylabel('PIC Weight')
ax.set_title('PIC Weight Trend')
plt.xticks(list(range(0, T+1, 10)))
# ax.set_xlim(0.2, 0.6)
ax.set_ylim(-0.1, 1.0)
ax.grid(True, alpha=0.3)
ax.legend(title="T : Critical Frame No.")



ax.figure.savefig("./PIC_weight.png")
