# Fundamental
import argparse  # noqa
import os  # noqa
import time  # noqa

# Submodules
import graphing  # noqa
from extern.bfield.bfield import solution as bioSol  # noqa
from extern.bfield.lorentz import solution as lorSol  # noqa

# Imports
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from numba import njit  # noqa
import matplotlib.pyplot as plt

# Constants
gravity = 9.81  # m/s^2

# Magnet Variables
num_mags = 4
magnets = np.linspace(-0.05, 0.05, num_mags)
length = 0.1
m = 0.1
m_rad = 0.006
m_seg = 12
mass = 0.02

drag = 0.00005  # Drag coefficient

# Initial state
initial_angles = np.radians([-8, -3, 3, 35])
initial_angular_velocities = np.zeros(num_mags)
state = np.concatenate([initial_angles, initial_angular_velocities])

t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], t_span[1] * 120)


# @njit
def step(t, state):
    angles = state[:num_mags]
    angular_velocities = state[num_mags:]

    ang_accel = np.zeros(num_mags)

    # iterate through interaction pairs
    for ind, theta in enumerate(angles[1:]):
        ind += 1
        force = np.zeros(3)
        force_a = np.zeros(3)
        pos = np.array(
            [magnets[ind] + np.sin(theta) * length, 0, length * np.cos(theta)]
        )
        theta_a = angles[ind - 1]
        pos_a = np.array(
            [
                magnets[ind - 1] + np.sin(theta_a) * length,
                0,
                length * np.cos(theta_a),
            ]
        )

        local_disp = np.array(
            [
                [np.sin(theta - theta_a), 0, np.cos(theta - theta_a)],
                [0, 1, 0],
                [np.cos(theta - theta_a), 0, -np.sin(theta - theta_a)],
            ]
        ).T @ (pos - pos_a)
        force -= lorSol(
            position=local_disp,
            orientation=np.array([np.sin(theta - theta_a), 0, np.cos(theta - theta_a)]),
            mradius=m_rad,
            mheight=0,
            moment=m,
            accuracy=np.array([1, m_seg]),
        )
        force = np.array(
            [
                np.cos(theta) * force[2] + np.sin(theta) * force[0],
                0,
                np.sin(theta) * force[2] + np.cos(theta) * force[0],
            ]
        )
        force_a = -force

        rel_pos = np.array([np.sin(theta) * length, 0, np.cos(theta) * length])
        rel_pos_a = np.array([np.sin(theta_a) * length, 0, np.cos(theta_a) * length])

        torque = np.cross(rel_pos, force)
        torque_a = np.cross(rel_pos_a, force_a)

        ang_accel[ind - 1] += torque_a[1] / (mass * length**2)
        ang_accel[ind] += torque[1] / (mass * length**2)

    for i in range(num_mags):
        force = np.array([0, 0, gravity * mass])
        rel_pos = np.array([np.sin(angles[i]) * length, 0, np.cos(angles[i]) * length])
        torque = np.cross(rel_pos, force)
        torque[1] -= drag * angular_velocities[i]
        ang_accel[i] += torque[1] / (mass * length**2)

    output = np.empty(2 * num_mags)
    output[:num_mags] = angular_velocities
    output[num_mags:] = ang_accel
    return output


# PRECOMPILING
def compile():
    print("\n\033[1m[COMPILING FUNCTIONS]\033[0m")
    t0 = time.perf_counter()
    print("Compiling B-Field..")
    bioSol(
        position=np.array([0, 0, 0]),
        mradius=1,
        mheight=1,
        moment=1,
        accuracy=[1, 1],
    )
    print("Precompiled B-Field:", int(1e3 * (time.perf_counter() - t0)), "ms\n")
    t0 = time.perf_counter()
    print("Compiling Lorentz..")
    lorSol(
        position=np.array([0, 0, 0]),
        mradius=1,
        mheight=1,
        moment=1,
        accuracy=[1, 1],
    )
    print("Precompiled Lorentz:", int(1e3 * (time.perf_counter() - t0)), "ms\n")
    t0 = time.perf_counter()
    print("Compiling Step Function..")
    step(0.00, np.zeros(num_mags * 2))
    print("Precompiled Step Function:", int(1e3 * (time.perf_counter() - t0)), "ms\n")


print("\n\033[1mMagnetic Cradle Numerical Solution\033[0m")
compile()


# Solve
sol = solve_ivp(step, t_span, state, t_eval=t_eval, method="RK45")
angles = sol.y[:num_mags]
angular_velocities = sol.y[num_mags:]

plt.figure(figsize=(12, 6))

# Plot angles
for i in range(num_mags):
    plt.plot(sol.t, np.degrees(angles[i]), label=f"Magnet {i + 1}")
plt.ylabel("Angle (deg)")
plt.xlabel("Time (s)")
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()

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

for i in range(num_mags):
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
    for i in range(num_mags):
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
    frames=len(sol.t),
    init_func=init,
    blit=False,
    interval=1000 / 60,
)

# Uncomment below to save as video (requires ffmpeg)
# ani.save("magnetic_cradle.mp4", writer="ffmpeg", fps=30)

plt.show()
