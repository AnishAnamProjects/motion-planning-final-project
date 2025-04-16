"""
author: Peter Huang, Antonio Cuni
email: hbd730@gmail.com, anto.cuni@gmail.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

from quadPlot import plot_quad_3d
import controller
import trajGen
import trajGen3D
from model.quadcopter import Quadcopter
import numpy as np
import time as time_module
import random

animation_frequency = 50
control_frequency = 200  # Hz for attitude control loop
control_iterations = control_frequency / animation_frequency
dt = 1.0 / control_frequency
sim_time = [0.0]
simulation_complete = False

def generate_random_target():
    # Generate target closer to the start position to avoid extreme movements
    return np.array([random.uniform(1, 10), random.uniform(1, 10), random.uniform(0, 2)])

def check_collision(quad_position, target_position, radius=3.0):
    distance = np.linalg.norm(quad_position - target_position)
    return distance <= radius

def generate_waypoints_to_target(start_pos, target_pos, num_points=2):
    """Generate waypoints in a systematic sweep pattern at target's Z height - go right, then up, then left, until all points are covered"""
    # Use target's Z height for the sweep
    fixed_z = target_pos[2]
    
    # Calculate grid dimensions
    x_min, x_max = 1.0, 9.0
    y_min, y_max = 1.0, 9.0
    step_size = 2.0
    
    # Create waypoints in a grid pattern
    waypoints = []
    y = y_min
    direction = 1  # 1 for right, -1 for left
    
    while y <= y_max:
        if direction == 1:
            # Move right
            x = x_min
            while x <= x_max:
                waypoints.append([x, y, fixed_z])
                x += step_size
        else:
            # Move left
            x = x_max
            while x >= x_min:
                waypoints.append([x, y, fixed_z])
                x -= step_size
        
        y += step_size
        direction *= -1  # Reverse direction
    
    return np.array(waypoints)

def attitudeControl(quad, sim_time, waypoints, coeff_x, coeff_y, coeff_z, target_position):
    # Use a faster velocity for quicker movement
    desired_state = trajGen3D.generate_trajectory(sim_time[0], 2.0, waypoints, coeff_x, coeff_y, coeff_z)
    F, M = controller.run(quad, desired_state)
    quad.update(dt, F, M)
    sim_time[0] += dt
    
    # Check for collision with target
    current_position = quad.world_frame()[:,4]  # Get current position
    if check_collision(current_position, target_position):
        return True
    return False

def main():
    global simulation_complete
    simulation_complete = False
    
    # Start from a more stable initial position
    pos = (1.0, 1.0, 1.0)
    attitude = (0, 0, 0)
    quadcopter = Quadcopter(pos, attitude)
    
    # Generate random target position
    target_position = generate_random_target()
    print(f"Target position: {target_position}")
    
    # Generate waypoints towards target
    start_pos = np.array(pos)  # Initial quadcopter position
    waypoints = generate_waypoints_to_target(start_pos, target_position)
    (coeff_x, coeff_y, coeff_z) = trajGen3D.get_MST_coefficients(waypoints)

    start_time = time_module.time()

    def control_loop(i):
        global simulation_complete
        if simulation_complete:
            return quadcopter.world_frame()
            
        for _ in range(int(control_iterations)):
            if attitudeControl(quadcopter, sim_time, waypoints, coeff_x, coeff_y, coeff_z, target_position):
                end_time = time_module.time()
                print(f"\nTarget reached!")
                print(f"Time taken: {end_time - start_time:.2f} seconds")
                print(f"Target position: {target_position}")
                print(f"Final quadcopter position: {quadcopter.world_frame()[:,4]}")
                simulation_complete = True
                break
        return quadcopter.world_frame()

    plot_quad_3d(waypoints, control_loop, target_position)

if __name__ == "__main__":
    main()
