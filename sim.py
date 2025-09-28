print("\n\033[1mMagnetic Cradle Numerical Solution\033[0m")
import argparse  # noqa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[Magnetic Cradle Simulation IYPT 2026]\nWritten by Luyu, see the MIT license before using. hehe ^-^"
    )

    parser.add_argument(
        "-a", "--animate", help="Show the animation", action="store_true"
    )
    parser.add_argument("-s", "--save", help="Save the animation", action="store_true")
    parser.add_argument(
        "-v", "--version", action="version", version="[Magnetic Cradle] v0.1a"
    )
    parser.add_argument("-d", "--debug", help="Enable errors", action="store_true")

    args = parser.parse_args()
    save_anim = args.save
    animate = args.animate
    debug = args.debug

# Fundamental
import os  # noqa
import time  # noqa

# Submodules
import graphing  # noqa
from extern.bfield.bfield import solution as bioSol  # noqa
from extern.bfield.lorentz import solution as lorSol  # noqa

# Imports
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit  # noqa
from matplotlib import animation  # noqa
import matplotlib.pyplot as plt

# Constants
gravity = 9.81  # m/s^2

# Magnet Variables
num_mags = 2
magnets = np.linspace(-0.03, 0.03, num_mags)
length = 0.2
m = 0.8
m_rad = 0.04
m_seg = 12
mass = 0.04

drag = 0.0002  # Drag coefficient (simple viscous)

# Initial state
initial_angular_velocities = np.zeros(num_mags)
timestep = 1 / 60

graphs = [graphing.Theta, graphing.Energy]
animate = True


@njit(cache=True)
def step(t: float, state: np.ndarray) -> np.ndarray:
    angles = state[:num_mags]
    angular_velocities = state[num_mags:]

    ang_accel = np.zeros(num_mags)

    # iterate through interaction pairs

    for ind in range(num_mags - 1):
        ind += 1
        theta = angles[ind]
        theta_a = angles[ind - 1]

        pos = np.array(
            [magnets[ind] + np.sin(theta) * length, 0, length * np.cos(theta)]
        )
        pos_a = np.array(
            [
                magnets[ind - 1] + np.sin(theta_a) * length,
                0,
                length * np.cos(theta_a),
            ]
        )

        local_disp = np.array(
            [
                [np.sin(theta_a), 0, np.cos(theta_a)],
                [0.0, 1.0, 0.0],
                [np.cos(theta_a), 0, -np.sin(theta_a)],
            ],
        ).T @ (pos - pos_a)

        force = -lorSol(
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

        rel_pos = np.array([np.sin(theta) * length, 0, np.cos(theta) * length])

        acc = np.cross(rel_pos, force)[1] / (mass * length**2)

        ang_accel[ind - 1] -= acc  # Conservation of angular momentum
        ang_accel[ind] += acc

    for i in range(num_mags):
        force = np.array([0, 0, gravity * mass])
        rel_pos = np.array([np.sin(angles[i]) * length, 0, np.cos(angles[i]) * length])
        torque = np.cross(rel_pos, force)
        torque[1] -= drag * angular_velocities[i]
        ang_accel[i] += torque[1] / (mass * length**2)

    output = np.zeros(2 * num_mags, dtype=np.float64)
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




# Solve
def solve(conditions,t_eval):
    t0 = time.perf_counter()
    print("Solving...")
    sol = solve_ivp(step, [0,t_eval[-1]], conditions, t_eval=t_eval, method="RK45")
    print("Finished Solving:", int(1e3 * (time.perf_counter() - t0)), "ms\n")

    angles = sol.y[:num_mags]
    angular_velocities = sol.y[num_mags:]
    if __name__ == "__main__":
        fig, axis = plt.subplots(len(graphs))
        if len(graphs) == 1:
            axis = [axis]
        axis[0].set_title("Cradle [MODEL]")

        for i, func in enumerate(graphs):
            func(
                axis[i],
                theta=angles,
                theta_dot=angular_velocities,
                timestep=timestep,
                time=t_eval,
                mass=mass,
                gravity=gravity,
                length=length,
            )

        plt.tight_layout()
        plt.show()

        if animate:
            graphing.Animate(magnets,length,angles,timestep,save_anim)

    return angles, angular_velocities

if __name__ == "__main__":
    compile()

    while True:
        print("\n\033[1mInput initial conditions (θ1,θ2, [...]) or [x] to exit:\033[0m")

        initial_angles = np.radians(np.array(input().split(), dtype="float"))

        print("\n\033[1mInput simulation length (default 8)\033[0m")
        inp = input()
        if inp:
            t_eval = np.linspace(0, int(inp), int(int(inp) / timestep))
        else:
            t_eval = np.linspace(0, 8, int(8 / timestep))

        if debug:
            solve(np.concatenate([initial_angles, initial_angular_velocities]),t_eval)
        else:
            try:
                solve(np.concatenate([initial_angles, initial_angular_velocities]),t_eval)
            except (ValueError, IndexError):
                print("Exiting.")
                break
