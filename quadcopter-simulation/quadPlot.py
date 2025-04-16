"""
author: Peter Huang
email: hbd730@gmail.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import numpy as np
import sys

history = np.zeros((500,3))
count = 0
target_reached = False
current_animation = None

def plot_quad_3d(waypoints, get_world_frame, target_position=None):
    """
    get_world_frame is a function which return the "next" world frame to be drawn
    """
    global target_reached, current_animation
    target_reached = False
    
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.plot([], [], [], '-', c='cyan')[0]
    ax.plot([], [], [], '-', c='red')[0]
    ax.plot([], [], [], '-', c='blue', marker='o', markevery=2)[0]
    ax.plot([], [], [], '.', c='red', markersize=4)[0]
    ax.plot([], [], [], '.', c='blue', markersize=2)[0]
    
    # Plot target
    if target_position is not None:
        ax.scatter([target_position[0]], [target_position[1]], [target_position[2]], c='r', s=100)
    
    set_limit()
    plot_waypoints(waypoints)
    
    def anim_callback(i):
        global target_reached, current_animation
        if target_reached:
            if current_animation:
                current_animation.event_source.stop()
            plt.close()
            return []
            
        frame = get_world_frame(i)
        if frame is None:
            target_reached = True
            if current_animation:
                current_animation.event_source.stop()
            plt.close()
            return []
            
        set_frame(frame)
        return []

    # Create animation with a maximum number of frames
    current_animation = animation.FuncAnimation(fig,
                               anim_callback,
                               frames=1000,  # Maximum number of frames
                               interval=10,
                               blit=False,
                               repeat=False)

    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        print("saving")
        current_animation.save('sim.gif', dpi=80, writer='imagemagick', fps=60)
    else:
        plt.show()

def plot_waypoints(waypoints):
    ax = plt.gca()
    lines = ax.get_lines()
    lines[-2].set_data(waypoints[:,0], waypoints[:,1])
    lines[-2].set_3d_properties(waypoints[:,2])

def set_limit():
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    
    # Make the grid look like squares
    ax.set_box_aspect([1, 1, 1])  # This ensures equal aspect ratio
    
    # Add grid lines
    ax.grid(True)
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks(np.arange(0, 11, 1))
    ax.set_zticks(np.arange(0, 11, 1))

def set_frame(frame):
    # convert 3x6 world_frame matrix into three line_data objects which is 3x2 (row:point index, column:x,y,z)
    lines_data = [frame[:,[0,2]], frame[:,[1,3]], frame[:,[4,5]]]
    ax = plt.gca()
    lines = ax.get_lines()
    for line, line_data in zip(lines[:3], lines_data):
        x, y, z = line_data
        line.set_data(x, y)
        line.set_3d_properties(z)

    # Plot search radius sphere around the drone
    drone_pos = frame[:,4]
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 3 * np.outer(np.cos(u), np.sin(v)) + drone_pos[0]
    y = 3 * np.outer(np.sin(u), np.sin(v)) + drone_pos[1]
    z = 3 * np.outer(np.ones(np.size(u)), np.cos(v)) + drone_pos[2]
    
    # Remove previous sphere if it exists
    for artist in ax.collections:
        if isinstance(artist, Poly3DCollection):
            artist.remove()
    
    # Add new sphere
    ax.plot_surface(x, y, z, color='g', alpha=0.2)

    global history, count
    # plot history trajectory
    history[count] = frame[:,4]
    if count < np.size(history, 0) - 1:
        count += 1
    zline = history[:count,-1]
    xline = history[:count,0]
    yline = history[:count,1]
    if(lines != []):
        lines[-1].set_data(xline, yline)
        lines[-1].set_3d_properties(zline)
