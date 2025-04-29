from drone import Drone
import trajGen3D
import controller
from runsim import *

import matplotlib.pyplot as plt
from environment import Environment
import open3d as o3d
import time

n_drones = 3
drones = [None] * n_drones  # Initialize a list to hold the drones
waypoints = [None] * n_drones  # Initialize a list to hold the waypoints
drone_states = [None] * n_drones  # Initialize a list to hold the drone states

#initialize plotter or the environment
file = "resources/Industrial_full_scene_0.5hires_lessfloor.ply"
goal = (10, 10, 10)
industrial = Environment(goal_position=goal, filename=file, resolution=1)

# intialize the unknown map
knowledge_map = np.zeros(industrial.obstacle_map.shape)  # Initialize the knowledge map with zeros

start_positions = np.meshgrid(np.arange(0, n_drones, 1), np.arange(0, n_drones, 1))
start_positions = np.array(start_positions).T.reshape(-1, 2)  # Reshape to get pairs of coordinates
start_positions = np.hstack((start_positions, np.zeros((start_positions.shape[0], 1))))  # Add z=0
# Initialize Drones
for i in range(n_drones):
    #start the drones in a grid at z = 0
    
    pos = start_positions[i]  # Start each drone at a different position
    attitude = (0, 0, 0)  # Initial attitude (roll, pitch, yaw)
    start_state = [pos, attitude]
    drones[i] = Drone(start_state, industrial, knowledge_map)  # Initialize each drone with its start state and environment

# Generate Waypoints. Fill this in later with motion planner
for i in range(n_drones):
    # Generate a set of waypoints along a grid sweep
    waypoints[i] = np.zeros((10, 3))  # Initialize waypoints for each drone
    waypoints[i] = np.random.uniform(0, 100, (10, 3))  # Random waypoints in a 10x10x10 space

    # generate a set of random waypoints within the bounds of the environment
    waypoints[i] = np.random.uniform([industrial.x_min, industrial.y_min, industrial.z_min],
                                  [industrial.x_max, industrial.y_max, industrial.z_max], (10, 3))

#define waypoint array of two points
# waypoints = [np.array([[0, 0, 5,]])]


# Execute the trajectory for each drone
for i in range(n_drones):
    drones[i].execute_trajectory(waypoints[i])
    drone_states[i] = np.array(drones[i].states)  # Store the states of each drone

# # do another set of waypoints
# waypoints = [np.array([[-10, -10, 10]])]

# # Execute the trajectory for each drone
# for i in range(n_drones):
#     drones[i].execute_trajectory(waypoints[i])
#     drone_states[i] = np.array(drones[i].states)  # Store the states of each drone

# Plot the trajectories of each drone
drone_trajectory_pcd = [None] * n_drones  # Initialize a list to hold the point clouds for each drone
plotting_list = [industrial.point_cloud]
for i in range(n_drones):
    drone_trajectory_pcd[i] = o3d.geometry.PointCloud()  # Create a new point cloud for each drone
    drone_trajectory_pcd[i].points = o3d.utility.Vector3dVector(drones[i].states)
    #drone_trajectory_pcd[i].paint_uniform_color([0, 0, 1])  # Set color to blue
    plotting_list.append(drone_trajectory_pcd[i])

plotting_list.append(industrial.detected_obstacles)  # Add the knowledge map to the plotting list
plotting_list.append(industrial.detected_free_space)
o3d.visualization.draw_geometries(plotting_list)

vis = o3d.visualization.Visualizer()
vis.create_window()

for i in range(n_drones):
    vis.add_geometry(drone_trajectory_pcd[i])  # Add the point cloud for each drone to the visualizer


vis.add_geometry(industrial.point_cloud)

print(len(drones[0].states))
# create an animation loop
for steps in range(1, len(drones[0].states)):
    #new_pcd.points = o3d.utility.Vector3dVector(drones[0].states[0:steps])
    for i in range(n_drones):
        # Add the point cloud for each drone to the visualizer
        drone_trajectory_pcd[i].points = o3d.utility.Vector3dVector([drones[i].states[steps]])
        vis.update_geometry(drone_trajectory_pcd[i])

    vis.poll_events()
    vis.update_renderer()
    print(steps)
    time.sleep(0.001)
vis.destroy_window()