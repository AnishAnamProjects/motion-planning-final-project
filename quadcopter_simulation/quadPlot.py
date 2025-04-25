### Updated quadPlot.py ###
"""
author: Peter Huang (modified)
Visualize multiple quadcopters, waypoints, and discovered voxels in 3D
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pointcloud

# Define world limits and grid tick spacing
x_min, x_max, xtick = 0.0, 10.0, 1
y_min, y_max, ytick = 0.0, 10.0, 1
z_min, z_max, ztick = 0.0, 10.0, 1

# Keep animation reference alive
global_anim = None

def plot_quad_3d(waypoints, get_world_frame, get_known_map, voxel_centers, limits):
    """
    waypoints: (M,3) array of waypoint coords (in world units)
    get_world_frame: function(i)-> world_frame (N,3,6) for N drones
    get_known_map: function(i)-> known_map ndarray (nx,ny,nz)
    voxel_centers: (nx*ny*nz,3) coords of voxel centers
    """

    global global_anim

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1], projection='3d')
    set_limit(ax, limits)
    plot_waypoints(ax, waypoints)

    drone_colors = ['b', 'g', 'r', 'y']  # Add more colors if needed but note they end up in the legend
    quad_scatters = [
        ax.scatter([], [], [], c=c, s=60, label=f'Drone {i}')
        for i, c in enumerate(drone_colors)
    ]
    
    free_scatter = ax.scatter([], [], [], c='gray', s=4, alpha=0.3, label='Free')
    obs_scatter  = ax.scatter([], [], [], c='red', s=20, alpha=1, label='Obstacle')
    goal_scatter = ax.scatter([], [], [], c='green', s=6, alpha=0.6, label='Discovered Goal')
    ax.legend(loc='upper right')

    def anim_callback(frame_idx):
        wfs = get_world_frame(frame_idx)
        if wfs is None:
            if global_anim:
                global_anim.event_source.stop()
            return []
        
        wfs = np.asarray(wfs)  # Shape: (N_drones, 3, 6)
        
        # Update each drone's position
        for i, scatter in enumerate(quad_scatters):
            if i < len(wfs):
                pos = wfs[i,:,4]
                scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        # Update voxel visualization
        known = get_known_map(frame_idx).ravel()
        centers = voxel_centers
        idx_free = np.where(known == 0)[0]
        free_scatter._offsets3d = (centers[idx_free,0], centers[idx_free,1], centers[idx_free,2])
        idx_obs  = np.where(known == 1)[0]
        obs_scatter._offsets3d  = (centers[idx_obs,0], centers[idx_obs,1], centers[idx_obs,2])
        idx_goal = np.where(known == 2)[0]
        goal_scatter._offsets3d = (centers[idx_goal,0], centers[idx_goal,1], centers[idx_goal,2])
        
        return quad_scatters + [free_scatter, obs_scatter, goal_scatter]

    global_anim = animation.FuncAnimation(
        fig, anim_callback, frames=500, interval=20, blit=False, repeat=False
    )
    plt.show()

# set tick according to max limit
def set_tick(limit):

    if limit <= 5:
        return 1
    elif limit <= 10:
        return 2
    elif limit <= 20:
        return 5
    elif limit <= 200:
        return 10
    else:
        return 100

def set_limit(ax, limits):

    x_min, x_max, y_min, y_max, z_min, z_max = limits

    xtick = set_tick(x_max)
    ytick = set_tick(y_max)
    ztick = set_tick(z_max)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xticks(np.arange(x_min, x_max+1, xtick))
    ax.set_yticks(np.arange(y_min, y_max+1, ytick))
    ax.set_zticks(np.arange(z_min, z_max+1, ztick))

def plot_waypoints(ax, waypoints):
    ax.plot(waypoints[:,0], waypoints[:,1], waypoints[:,2], 'k--o', label='Waypoints', alpha=0.3)