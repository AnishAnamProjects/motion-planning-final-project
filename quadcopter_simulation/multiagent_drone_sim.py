"""
Multi-agent Drone Simulation with Role-Based Exploration and Task Execution
===========================================================================
This simulation implements:
1. Surveyor drones that explore and build a shared configuration space
2. Worker drones that use the map to go to targets when found
3. Performance evaluation metrics
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pointcloud

from drone import Drone
import trajGen3D
import controller
from multiagent_planner import MultiAgentPlanner, SharedMap
from quadPlot import plot_quad_3d

lowres_filename = "resources/Industrial_full_scene_2.0lowres_lessfloor.ply"
medres_filename = "resources/Industrial_main_part_1.0medres.ply"
# medres_filename = "resources/Industrial_full_scene_1.0medres_lessfloor.ply"
hires_filename = "resources/Industrial_full_scene_0.5hires_lessfloor.ply"

class MultiAgentDroneSimulation:
    # def __init__(self, n_surveyors=3, n_workers=1, space_limit=10.0): # original code
    def __init__(self, filename = "", cloud_resolution = 1.0, n_surveyors=3, n_workers=1, space_limit=10.0):

        # Simulation parameters
        self.cloud_res = 1/cloud_resolution
        self.animation_frequency = 30
        self.dt = 1.0 / self.animation_frequency
        self.space_limit = space_limit
        self.n_surveyors = n_surveyors
        self.n_workers = n_workers
        self.n_drones = n_surveyors + n_workers
        self.SAFE_DISTANCE = 1.5  # Minimum distance between drones
        self.SPEED = 0.3  # Units per frame
        
        # World bounds
        self.x_min, self.x_max = 0.0, space_limit
        self.y_min, self.y_max = 0.0, space_limit
        self.z_min, self.z_max = 0.0, space_limit

        # # World bounds from point cloud
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = pointcloud.get_min_max(filename)
        
        # # World limits to be sent to quadPlot.py
        self.limits = (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

        # # points_decimal = pointcloud.get_points_o3d(filename)
        # # rounded_points = np.around(points_decimal, decimals=0) # round up to nearest int
        # # self.points = rounded_points.astype(int) # remove the decimal

        # Get the point cloud, round the coordinates and set the points to integers
        # so they can be added to the environment as an obstacles
        self.cloud_x, self.cloud_y, self.cloud_z = pointcloud.get_points(filename)
        self.point_cloud = (self.cloud_x, self.cloud_y, self.cloud_z)
        self.cloud_x = np.round(self.cloud_x, 1)
        self.cloud_y = np.round(self.cloud_y, 1)
        self.cloud_z = np.round(self.cloud_z, 1)
        self.cloud_x = self.cloud_x.astype(int) # = int(self.cloud_x)
        self.cloud_y = self.cloud_y.astype(int) # = int(self.cloud_y)
        self.cloud_z = self.cloud_z.astype(int) # = int(self.cloud_z)
        self.point_cloud = (self.cloud_x, self.cloud_y, self.cloud_z)

        # Voxel grid for sensing
        values = (self.x_max, self.y_max, self.z_max)
        self.nx, self.ny, self.nz = tuple(values * self.cloud_res for values in values)
        self.nx = round(self.nx, 1)
        self.ny = round(self.ny, 1)
        self.nz = round(self.nz, 1)
        self.nx = int(self.nx + 1)
        self.ny = int(self.ny + 1)
        self.nz = int(self.nz + 1)
        self.xs = np.linspace(self.x_min, self.x_max, self.nx)
        self.ys = np.linspace(self.y_min, self.y_max, self.ny)
        self.zs = np.linspace(self.z_min, self.z_max, self.nz)
        Xc, Yc, Zc = np.meshgrid(self.xs, self.ys, self.zs, indexing='ij')
        self.voxel_centers = np.stack([Xc.ravel(), Yc.ravel(), Zc.ravel()], axis=1)
        
        # # Voxel grid for sensing
        # self.nx, self.ny, self.nz = 20, 20, 20
        # self.xs = np.linspace(self.x_min, self.x_max, self.nx)
        # self.ys = np.linspace(self.y_min, self.y_max, self.ny)
        # self.zs = np.linspace(self.z_min, self.z_max, self.nz)
        # Xc, Yc, Zc = np.meshgrid(self.xs, self.ys, self.zs, indexing='ij')
        # self.voxel_centers = np.stack([Xc.ravel(), Yc.ravel(), Zc.ravel()], axis=1)
        
        # Occupancy maps: 0=free, 1=obs, 2=goal, 255=unknown
        self.truth_map = np.zeros((self.nx, self.ny, self.nz), dtype=np.uint8)
        
        # Initialize drones and their states
        self.drones = [None] * self.n_drones
        self.waypoints = [[] for _ in range(self.n_drones)]
        self.current_waypoint_idx = [0] * self.n_drones
        self.done = [False] * self.n_drones
        
        # Sensing radius squared (for efficiency)
        self.sense_r2 = 2.0**2
        
        # Multi-agent planner
        space_limits = (
            self.x_min, self.x_max, 
            self.y_min, self.y_max,
            self.z_min, self.z_max
        )
        self.planner = MultiAgentPlanner(
            n_surveyors, 
            n_workers, 
            space_limits,
            grid_dims=(self.nx, self.ny, self.nz)
        )
        
        # Track drone roles
        self.drone_roles = ['surveyor'] * n_surveyors + ['worker'] * n_workers
        
        # Evaluation metrics
        self.simulation_start_time = None
        self.metrics = {
            "time_to_target": None,
            "coverage_rate": [],
            "map_accuracy": [],
            "collision_rate": 0,
            "path_optimality": None,
            "redundancy": 0
        }
        
        # Initialize the environment
        self.setup_environment()
        
    def setup_environment(self):
        """Add random obstacles and a target to the environment"""

        # Reset environment
        self.truth_map = np.zeros((self.nx, self.ny, self.nz), dtype=np.uint8)

        # Add points from point cloud to environment as obstacles
        # points = np.column_stack((self.cloud_x, self.cloud_y, self.cloud_z))
        self.truth_map[self.point_cloud] = 1
        
        # # Add random obstacles
        # num_obstacles = 50
        # for _ in range(num_obstacles):
        #     box_min = np.array([
        #         random.randint(2, self.nx-4),
        #         random.randint(2, self.ny-4),
        #         random.randint(1, self.nz-4)
        #     ])
        #     box_size = np.array([3, 3, 3])
        #     box_max = box_min + box_size
        #     self.truth_map[box_min[0]:box_max[0], 
        #                   box_min[1]:box_max[1], 
        #                   box_min[2]:box_max[2]] = 1
        
        # Add a target/goal
        while True:
            goal_x = random.randint(self.nx//4, self.nx*3//4)
            goal_y = random.randint(self.ny//4, self.ny*3//4)
            goal_z = random.randint(self.nz//4, self.nz*3//4)
            
            # Make sure target isn't in an obstacle
            if self.truth_map[goal_x, goal_y, goal_z] == 0:
                self.truth_map[goal_x, goal_y, goal_z] = 2  # 2 represents target
                self.goal_position = (goal_x, goal_y, goal_z)
                break
    
    def world_to_grid(self, pos):
        """Convert world coordinates to grid indices"""
        ix = int((pos[0]-self.x_min)/(self.x_max-self.x_min)*(self.nx-1))
        iy = int((pos[1]-self.y_min)/(self.y_max-self.y_min)*(self.ny-1))
        iz = int((pos[2]-self.z_min)/(self.z_max-self.z_min)*(self.nz-1))
        return np.clip(ix, 0, self.nx-1), np.clip(iy, 0, self.ny-1), np.clip(iz, 0, self.nz-1)
    
    def grid_to_world(self, grid_pos):
        """Convert grid indices to world coordinates"""
        ix, iy, iz = grid_pos
        x = self.x_min + (ix / (self.nx - 1)) * (self.x_max - self.x_min)
        y = self.y_min + (iy / (self.ny - 1)) * (self.y_max - self.y_min)
        z = self.z_min + (iz / (self.nz - 1)) * (self.z_max - self.z_min)
        return np.array([x, y, z])
    
    def initialize_drones(self, start_positions=None):
        """Initialize drones with given or default start positions"""
        if start_positions is None:
            # Ensure positions are well-distributed and not too close together
            start_positions = []
            for i in range(self.n_drones):
                # Distribute drones in a grid across the environment
                row = i // 2
                col = i % 2
                pos = np.array([
                    self.x_min + 1.0 + col * (self.space_limit - 2.0) / 2,
                    self.y_min + 1.0 + row * (self.space_limit - 2.0),
                    self.z_min + 3.0  # Start higher off the ground
                ])
                start_positions.append(pos)
                print(f"Initializing drone {i} at position {pos}")
        
        for i in range(self.n_drones):
            pos = start_positions[i]
            attitude = (0, 0, 0)  # Initial attitude (roll, pitch, yaw)
            start_state = [pos, attitude]
            self.drones[i] = Drone(start_state)
    
    def update_known_map(self, drone_id, drone_pos):
        """Update the known map based on drone's current position
        
        Returns: Dictionary of observed cells and their states
        """
        observed_cells = {}
        
        # Get all voxels within sensing radius
        d2 = np.sum((self.voxel_centers - drone_pos)**2, axis=1)
        visible_indices = np.where(d2 <= self.sense_r2)[0]
        
        for idx in visible_indices:
            # Convert voxel center to grid coordinates
            grid_pos = self.world_to_grid(self.voxel_centers[idx])
            
            # Record observed cell and its state from truth map
            observed_cells[grid_pos] = self.truth_map[grid_pos]
        
        # Update planner's shared map
        self.planner.update_map(drone_id, drone_pos, observed_cells)
        
        return observed_cells
    
    def check_collision(self, pos):
        """Check if position collides with obstacle"""
        ix, iy, iz = self.world_to_grid(pos)
        return self.truth_map[ix, iy, iz] == 1

    def get_repulsive_force(self, pos1, pos2, safe_dist):
        """Calculate repulsive force between two points"""
        diff = pos1 - pos2
        dist = np.linalg.norm(diff)
        if dist < safe_dist:
            # Force magnitude inversely proportional to distance
            magnitude = (safe_dist - dist) / safe_dist
            return (diff / dist) * magnitude * 2.0
        return np.zeros(3)
    
    def control_loop(self, frame_idx):
        """Update drone positions for one animation frame"""
        if self.simulation_start_time is None:
            self.simulation_start_time = time.time()
            
        # Add debugging output every 30 frames
        if frame_idx % 30 == 0:  # Every second (assuming 30fps)
            print(f"Frame {frame_idx}: Drone positions:")
            for i, drone in enumerate(self.drones):
                print(f"  Drone {i} ({self.drone_roles[i]}): {drone.position()}")
            
            # Print exploration statistics
            stats = self.planner.get_exploration_stats(self.truth_map)
            print(f"  Coverage: {stats['coverage_percent']:.2f}%")
            print(f"  Phase: {self.planner.phase}")
            if self.planner.target_pos is not None:
                print(f"  Target found at: {self.planner.target_pos}")
            
        if all(self.done[self.n_surveyors:]):
            # record mission time
            if self.simulation_start_time is not None:
                self.completion_time = time.time() - self.simulation_start_time
                print("\n=== Mission Complete! ===")
                print(f"All worker drones have reached the target.")
                print(f"Total mission time: {self.completion_time:.2f} seconds")
                self.print_metrics()
            return None
            
        world_frames = []
        for i in range(self.n_drones):
            if self.done[i]:
                world_frames.append(self.drones[i].world_frame())
                continue
                
            current_pos = np.array(self.drones[i].position())
            
            # Update known map based on sensing
            self.update_known_map(i, current_pos)
            
            # Get next waypoint from planner based on drone role
            if i < self.n_surveyors:
                role = 'surveyor'
                role_id = i
            else:
                role = 'worker'
                role_id = i - self.n_surveyors
                
            target_pos = self.planner.get_next_waypoint(role_id, role, current_pos)
            
            # Direction to target
            to_target = target_pos - current_pos
            dist_to_target = np.linalg.norm(to_target)
            
            # Check if drone is "stuck" (barely moving toward target)
            if dist_to_target < self.SPEED * 5 and dist_to_target > self.SPEED:
                # If worker drone is stuck when approaching target, assign new path
                if role == 'worker' and self.planner.phase == "execution":
                    # Refresh path to target
                    grid_pos = self.world_to_grid(current_pos)
                    target_grid_pos = self.planner.target_grid_pos
                    path = self.planner.find_path(grid_pos, target_grid_pos)
                    
                    if path:
                        print(f"Worker {role_id} path refreshed with {len(path)} waypoints")
                        self.planner.worker_paths[role_id] = [self.planner.grid_to_world(p) for p in path]
                        # Get the new next waypoint
                        target_pos = self.planner.worker_paths[role_id][0]
                        to_target = target_pos - current_pos
                        dist_to_target = np.linalg.norm(to_target)
                    else:
                        # If no path found, move randomly to try to find a better position
                        random_direction = np.random.uniform(-1, 1, 3)
                        random_direction = random_direction / np.linalg.norm(random_direction)
                        new_pos = current_pos + random_direction * self.SPEED * 3
                        new_pos = np.clip(new_pos, 
                                         [self.x_min, self.y_min, self.z_min],
                                         [self.x_max, self.y_max, self.z_max])
                        
                        if not self.check_collision(new_pos):
                            current_pos = new_pos
                            print(f"Worker {role_id} was stuck, moved randomly to {current_pos}")
                            self.drones[i].state[0:3] = current_pos
                            world_frames.append(self.drones[i].world_frame())
                            continue
                # For other stuck drones, just move randomly
                else:
                    random_direction = np.random.uniform(-1, 1, 3)
                    random_direction = random_direction / np.linalg.norm(random_direction)
                    new_pos = current_pos + random_direction * self.SPEED * 3
                    new_pos = np.clip(new_pos, 
                                    [self.x_min, self.y_min, self.z_min],
                                    [self.x_max, self.y_max, self.z_max])
                    
                    if not self.check_collision(new_pos):
                        current_pos = new_pos
                        print(f"Drone {i} was stuck, moved randomly to {current_pos}")
                        self.drones[i].state[0:3] = current_pos
                        world_frames.append(self.drones[i].world_frame())
                        continue
            
            if dist_to_target < self.SPEED:
                # Reached current waypoint
                if role == 'worker' and self.planner.target_pos is not None:
                    # Check if worker has reached the target
                    target_dist = np.linalg.norm(current_pos - self.planner.target_pos)
                    if target_dist < self.SPEED:
                        print(f"Worker {role_id} has reached the target!")
                        self.done[i] = True
                
                current_pos = target_pos
            else:
                # Calculate movement direction
                direction = to_target / dist_to_target
                
                # Add repulsive forces from other drones - reduce strength to prevent getting stuck
                avoidance = np.zeros(3)
                for j, other_drone in enumerate(self.drones):
                    if i != j and not self.done[j]:
                        force = self.get_repulsive_force(
                            current_pos, 
                            np.array(other_drone.position()), 
                            self.SAFE_DISTANCE
                        )
                        # Reduce the impact of avoidance forces to prevent gridlock
                        avoidance += force * 0.5
                
                # Combine goal direction with avoidance
                if np.any(avoidance):
                    movement = direction + avoidance
                    movement = movement / np.linalg.norm(movement)
                else:
                    movement = direction
                
                # Increase movement speed for worker drones heading to target
                if role == 'worker' and self.planner.phase == "execution":
                    self.SPEED = 0.5  # Higher speed for workers going to target
                else:
                    self.SPEED = 0.3  # Normal speed for exploration
                
                # Update position
                new_pos = current_pos + movement * self.SPEED
                
                # Check bounds and collisions
                new_pos = np.clip(new_pos, 
                                [self.x_min, self.y_min, self.z_min],
                                [self.x_max, self.y_max, self.z_max])
                
                if self.check_collision(new_pos):
                    print(f"Drone {i} detected collision, finding new path")
                    self.planner.record_collision()
                    self.metrics["collision_rate"] += 1
                    
                    # Actively try a different direction instead of just reporting collision
                    random_direction = np.random.uniform(-1, 1, 3)
                    random_direction = random_direction / np.linalg.norm(random_direction)
                    new_pos = current_pos + random_direction * self.SPEED * 2
                    new_pos = np.clip(new_pos, 
                                    [self.x_min, self.y_min, self.z_min],
                                    [self.x_max, self.y_max, self.z_max])
                    
                    # Double-check the new position isn't in collision
                    if not self.check_collision(new_pos):
                        current_pos = new_pos
                else:
                    current_pos = new_pos
            
            # Update drone position
            self.drones[i].state[0:3] = current_pos  # Update position in state vector
            world_frames.append(self.drones[i].world_frame())
        
        # Update metrics periodically
        if frame_idx % 10 == 0:
            stats = self.planner.get_exploration_stats(self.truth_map)
            self.metrics["coverage_rate"].append(stats["coverage_percent"])
            self.metrics["map_accuracy"].append(stats["map_accuracy"] if "map_accuracy" in stats else 0)
            
            if stats["target_found_time"] is not None and self.metrics["time_to_target"] is None:
                self.metrics["time_to_target"] = stats["target_found_time"]
            
            # Check if any worker has reached the target
            if self.planner.target_pos is not None:
                for i in range(self.n_surveyors, self.n_drones):
                    if not self.done[i]:
                        worker_pos = np.array(self.drones[i].position())
                        dist_to_target = np.linalg.norm(worker_pos - self.planner.target_pos)
                        if dist_to_target < self.SPEED * 2:
                            self.done[i] = True
                            print(f"Worker {i - self.n_surveyors} has reached the target!")
        
        return np.stack(world_frames)
    
    def simulate(self):
        """Run the simulation with animation"""
        # Initialize drones
        self.initialize_drones()
        
        # Run animation
        plot_quad_3d(
            np.zeros((1, 3)),  # Dummy waypoints, not used in our algorithm
            self.control_loop,
            lambda idx: self.planner.shared_map.grid,
            self.voxel_centers,
            self.point_cloud,
            self.limits
        )
        
        # Print final metrics
        self.print_metrics()
    
    def print_metrics(self):
        """Print the simulation performance metrics"""
        print("\n=== Simulation Performance Metrics ===")
        print(f"Time to Target Detection: {self.metrics['time_to_target']:.2f} seconds")
        print(f"Final Coverage Rate: {self.metrics['coverage_rate'][-1]:.2f}%")
        
        if self.metrics['map_accuracy']:
            print(f"Final Map Accuracy (IoU): {self.metrics['map_accuracy'][-1]:.2f}")
        
        print(f"Collision Rate: {self.metrics['collision_rate']}")
        
        # Plot metrics over time
        self.plot_metrics()
    
    def plot_metrics(self):
        """Plot the evolution of metrics over time"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        x = list(range(len(self.metrics["coverage_rate"])))
        ax1.plot(x, self.metrics["coverage_rate"], 'b-', label='Coverage Rate (%)')
        ax1.set_xlabel('Frame (x10)')
        ax1.set_ylabel('Percentage')
        ax1.set_title('Coverage Rate Over Time')
        ax1.legend()
        ax1.grid(True)
        
        if self.metrics["map_accuracy"]:
            ax2.plot(x, self.metrics["map_accuracy"], 'r-', label='Map Accuracy (IoU)')
            ax2.set_xlabel('Frame (x10)')
            ax2.set_ylabel('IoU Score')
            ax2.set_title('Map Accuracy Over Time')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
