### Updated runsim.py ###
"""
author: Peter Huang, Antonio Cuni (modified)
Run a single quadcopter with user‑specified world‑coordinate waypoints
Trajectory now starts from the quadcopter's initial position to avoid large jumps.
"""
import numpy as np
import random
from quadPlot import plot_quad_3d
import controller
import trajGen3D
from model.quadcopter import Quadcopter

# Simulation parameters
animation_frequency = 5
control_frequency   = 200  # Hz
dt = 1.0 / control_frequency
control_iterations  = control_frequency // animation_frequency

# World bounds
x_min, x_max = 0.0, 10.0
y_min, y_max = 0.0, 10.0
z_min, z_max = 0.0, 10.0

# Voxel grid for sensing (unchanged)
nx, ny, nz = 20, 20, 20
xs = np.linspace(x_min, x_max, nx)
ys = np.linspace(y_min, y_max, ny)
zs = np.linspace(z_min, z_max, nz)
Xc, Yc, Zc = np.meshgrid(xs, ys, zs, indexing='ij')
voxel_centers = np.stack([Xc.ravel(), Yc.ravel(), Zc.ravel()], axis=1)
# Occupancy maps: 0=free,1=obs,2=goal,255=unknown
truth_map = np.zeros((nx,ny,nz), dtype=np.uint8)
known_map = 255 * np.ones_like(truth_map)
# Obstacles defined as a box region
# Define voxel-index bounds for the obstacle box (inclusive low, exclusive high)
box_min = np.array([3, 3, 1], dtype=int)   # x from 3 to 6, y from 3 to 6, z from 1 to 3
box_max = np.array([7, 7, 7], dtype=int)
# Fill that cuboid in the truth map
truth_map[box_min[0]:box_max[0], box_min[1]:box_max[1], box_min[2]:box_max[2]] = 1
# Goal remains at center top
gx,gy,gz = nx//2, ny//2, nz-1
truth_map[2,2,2] = 2
# Helper: world pos -> grid idx
def world_to_grid(pos):
    ix = int((pos[0]-x_min)/(x_max-x_min)*(nx-1))
    iy = int((pos[1]-y_min)/(y_max-y_min)*(ny-1))
    iz = int((pos[2]-z_min)/(z_max-z_min)*(nz-1))
    return np.clip(ix,0,nx-1), np.clip(iy,0,ny-1), np.clip(iz,0,nz-1)
sense_r2 = 2.0**2

# Initialize quad at some height above ground
start = np.array([1.0,1.0,1.0])  # z=2 to start off the ground
quad = Quadcopter(start, (0,0,0))

# === USER‑DEFINED WAYPOINTS IN WORLD COORDINATES ===
user_waypoints = np.array([
    [5.0, 5.0, 5.0], [7,8,5]
    ])
# Prepend the current start position so trajectory begins there
waypoints = np.vstack((start, user_waypoints))

# Generate MST coefficients through the full waypoint list
coeff_x, coeff_y, coeff_z = trajGen3D.get_MST_coefficients(waypoints)

sim_time = [0.0]
simulation_done = False

def control_loop(frame_idx):
    global simulation_done
    if simulation_done:
        return None

    for _ in range(control_iterations):
        pos = quad.position()
        # Sensing update
        d2 = np.sum((voxel_centers - pos)**2, axis=1)
        known_map.ravel()[d2<=sense_r2] = truth_map.ravel()[d2<=sense_r2]

        # Generate desired state and compute control
        desired = trajGen3D.generate_trajectory(
            sim_time[0], 2.0, waypoints, coeff_x, coeff_y, coeff_z
        )
        F, M = controller.run(quad, desired)
        quad.update(dt, F, M)
        sim_time[0] += dt

        # Enforce world bounds on position and velocity
        x,y,z = quad.position()
        vx,vy,vz = quad.velocity()
        x = np.clip(x, x_min, x_max)
        y = np.clip(y, y_min, y_max)
        z = np.clip(z, z_min, z_max)
        if x in (x_min, x_max): vx = 0
        if y in (y_min, y_max): vy = 0
        if z in (z_min, z_max): vz = 0
        quad.state[0:3] = [x,y,z]
        quad.state[3:6] = [vx,vy,vz]

        # Collision/goal check
        ix,iy,iz = world_to_grid([x,y,z])
        if truth_map[ix,iy,iz] == 1:
            print(f"Hit obstacle at {(x,y,z)}, stopping.")
            simulation_done = True; break
        if truth_map[ix,iy,iz] == 2:
            print("Reached goal!")

    return quad.world_frame()

if __name__=='__main__':
    plot_quad_3d(waypoints, control_loop, lambda idx: known_map, voxel_centers)