import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt

# import smplotlib
import numpy as np
from numpy import linalg as la
from scipy.signal import savgol_filter

def Theta(ax, **kwargs):
    angles = kwargs["theta"]
    time_array = kwargs["time"]

    ax.set_xlabel("Time $[s]$")
    ax.set_ylabel("Angle $[deg.]$")

    for i in range(len(angles)):
            ax.plot(time_array, np.degrees(angles[i]), label=f"Magnet {i + 1}")
    
def Energy(ax,**kwargs):
    angles = kwargs["theta"]
    pot = 1-(np.cos(angles))*kwargs["length"]*kwargs["mass"]*kwargs["gravity"]
    kin = 0.5 * kwargs["mass"]*(np.gradient(angles,axis=1)*kwargs["length"]/kwargs["timestep"])**2
    # realized need to get U from dipole too
    # Could do later but too lazy hehehe
    time_array = kwargs["time"]

    ax.set_xlabel("Time $[s]$")
    ax.set_ylabel("Energy $[J]$")

    for i in range(len(angles)):
            ax.plot(time_array, pot[i]+kin[i], label=f"Magnet {i + 1}")

def FFT(ax, **kwargs):
    print("Drawing frequency domain graph...")
    data = kwargs["theta"]
    timestep = kwargs["timestep"]
    for data in data:
        n = len(data)

        fft_result = np.fft.fft(data)
        freq = np.fft.fftfreq(n, d=timestep)
        ax.plot(freq, np.abs(fft_result))

    ax.set_xlabel("Frequency $[Hz]$")
    ax.set_ylabel("Amplitude $[arb.]$")

    ax.set_xlim(0, 4)  # Set x-axis bounds, e.g., from 0 to 10 Hz
    ax.grid()
    # ax.legend()


def Animate(magnets,length,angles,timestep):
    print("\n\nANIMATING...")
    fig_anim, ax_anim = plt.subplots(figsize=(8, 6))
    ax_anim.set_xlim(min(magnets) - length / 2, max(magnets) + length / 2)
    ax_anim.set_ylim(-length * 1.1, 0.01)
    ax_anim.set_aspect("equal")
    ax_anim.grid(True)

    ax_anim.plot([magnets[0], magnets[-1]], [0, 0], "k--", lw=1)

    lines = []  # Strings
    rects = []  # Magnets
    rect_width = 0.003
    rect_height = 0.015

    for i in range(len(magnets)):
        (line,) = ax_anim.plot([], [], "k-", lw=1, zorder=1)
        lines.append(line)

        # Magnet (rectangle)
        rect = plt.Rectangle(
            (0, 0),
            rect_width,
            rect_height,
            facecolor="tab:blue",
            edgecolor="black",
            zorder=2,
        )
        rects.append(rect)
        ax_anim.add_patch(rect)


    def init():
        for line in lines:
            line.set_data([], [])
        for rect in rects:
            rect.set_xy((0, 0))
        return lines + rects


    def update(frame):
        angles_deg = np.degrees(angles[:, frame])
        for i in range(len(magnets)):
            theta = np.radians(angles_deg[i])
            x_pivot = magnets[i]
            x_tip = x_pivot + length * np.sin(theta)
            y_tip = -length * np.cos(theta)

            lines[i].set_data([x_pivot, x_tip], [0, y_tip])

            rect_x = (
                x_tip - np.cos(theta) * rect_width / 2 + np.sin(theta) * rect_height / 2
            )
            rect_y = (
                y_tip - np.cos(theta) * rect_height / 2 - np.sin(theta) * rect_width / 2
            )

            rects[i].remove()
            rect = plt.Rectangle(
                (rect_x, rect_y),
                rect_width,
                rect_height,
                angle=np.degrees(theta),
                facecolor="grey",
                edgecolor="black",
                zorder=2,
            )
            rects[i] = rect
            ax_anim.add_patch(rect)

        return lines + rects


    ani = animation.FuncAnimation(
        fig_anim,
        update,
        frames=len(angles[0]),
        init_func=init,
        interval=1000 * timestep,
    )

    # Uncomment below to save as video (requires ffmpeg)
    # ani.save("magnetic_cradle.mp4", writer="ffmpeg", fps=30)

    plt.show()
