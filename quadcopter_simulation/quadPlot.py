### Updated quadPlot.py ###
"""
author: Peter Huang (modified)
Visualize a single quadcopter, waypoints, and discovered voxels in 3D, with bounds enforcement
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Keep animation reference alive
global_anim = None

def plot_quad_3d(waypoints, get_world_frame, get_known_map, voxel_centers):
    """
    waypoints: (M,3) array of waypoint coords (in world units)
    get_world_frame: function(i)-> world_frame (3x6)
    get_known_map: function(i)-> known_map ndarray (nx,ny,nz)
    voxel_centers: (nx*ny*nz,3) coords of voxel centers
    """
    global global_anim

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1], projection='3d')
    set_limit(ax)
    plot_waypoints(ax, waypoints)

    quad_scatter = ax.scatter([], [], [], c='b', s=60, label='Quad')
    free_scatter = ax.scatter([], [], [], c='gray', s=4, alpha=0.3, label='Free')
    obs_scatter  = ax.scatter([], [], [], c='red', s=20, alpha=1, label='Obstacle')
    goal_scatter = ax.scatter([], [], [], c='green',   s=6, alpha=0.6, label='Discovered Goal')
    ax.legend(loc='upper right')

    def anim_callback(frame_idx):
        wf = get_world_frame(frame_idx)
        if wf is None:
            if global_anim:
                global_anim.event_source.stop()
            return []
        wf = np.asarray(wf)
        pos = wf[:,4]
        quad_scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        known = get_known_map(frame_idx).ravel()
        centers = voxel_centers
        idx_free = np.where(known == 0)[0]
        free_scatter._offsets3d = (centers[idx_free,0], centers[idx_free,1], centers[idx_free,2])
        idx_obs  = np.where(known == 1)[0]
        obs_scatter._offsets3d  = (centers[idx_obs,0], centers[idx_obs,1], centers[idx_obs,2])
        idx_goal = np.where(known == 2)[0]
        goal_scatter._offsets3d = (centers[idx_goal,0], centers[idx_goal,1], centers[idx_goal,2])
        return [quad_scatter, free_scatter, obs_scatter, goal_scatter]

    global_anim = animation.FuncAnimation(
        fig, anim_callback, frames=500, interval=20, blit=False, repeat=False
    )
    plt.show()


def set_limit(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    ax.set_box_aspect([1,1,1])
    ax.grid(True)


def plot_waypoints(ax, waypoints):
    ax.plot(waypoints[:,0], waypoints[:,1], waypoints[:,2], 'g--o', label='Waypoints')