import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import graphing
from numpy import linalg as la
import sys
from scipy.signal import savgol_filter


animate = False
save_anim = False

data = pd.read_csv(sys.argv[1])
data = np.stack([data["Time"], data["t1"], data["t2"],data["t3"]], axis=0)
# Interpolating
if True:
    # Filtering
    data[1] = savgol_filter(data[1], 11, 3)
    data[2] = savgol_filter(data[2], 11, 3)
    data[3] = savgol_filter(data[3], 11, 3)

data[1:] = np.radians(data[1:])
graphs = [graphing.Theta ,graphing.FFT]


timestep = np.average(np.gradient(data[0]))
print("FPS", 1 / timestep)


fig, axis = plt.subplots(len(graphs))

axis[0].set_title("Cradle [DATA]")

for i, func in enumerate(graphs):
    func(
        axis[i],
        theta=data[1:],
        timestep=timestep,
        time=data[0],
    )
    axis[i].legend()

plt.tight_layout()
plt.show()
