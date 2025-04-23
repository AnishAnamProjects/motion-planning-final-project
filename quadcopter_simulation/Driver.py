from drone import Drone
import trajGen3D
import controller
from runsim import *

import matplotlib.pyplot as plt

n_drones = 10
drones = [None] * n_drones  # Initialize a list to hold the drones
waypoints = [None] * n_drones  # Initialize a list to hold the waypoints
drone_states = [None] * n_drones  # Initialize a list to hold the drone states

#initialize plotter or the environment
plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
ax.set_zlim([0, 100])

start_positions = np.meshgrid(np.arange(0, n_drones, 1), np.arange(0, n_drones, 1))
start_positions = np.array(start_positions).T.reshape(-1, 2)  # Reshape to get pairs of coordinates
start_positions = np.hstack((start_positions, np.zeros((start_positions.shape[0], 1))))  # Add z=0
# Initialize Drones
for i in range(n_drones):
    #start the drones in a grid at z = 0
    
    pos = start_positions[i]  # Start each drone at a different position
    attitude = (0, 0, 0)  # Initial attitude (roll, pitch, yaw)
    start_state = [pos, attitude]
    drones[i] = Drone(start_state)    

# Generate Waypoints. Fill this in later with motion planner
for i in range(n_drones):
    # Generate a random set of waypoints
    waypoints[i] = np.random.uniform(0, 100, (10, 3))  # Random waypoints in a 10x10x10 space

# Execute the trajectory for each drone
for i in range(n_drones):
    drones[i].execute_trajectory(waypoints[i])
    drone_states[i] = np.array(drones[i].states)  # Store the states of each drone

# Plot the trajectories of each drone
for i in range(n_drones):
    ax.scatter(waypoints[i][:, 0], waypoints[i][:, 1], waypoints[i][:, 2], marker='x')
    ax.scatter(drone_states[i][:, 0], drone_states[i][:, 1], drone_states[i][:, 2], marker='o')



plt.show()
