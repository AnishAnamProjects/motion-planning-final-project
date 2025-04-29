"""
Modified Multi-Agent Drone Simulation with RRT* Path Planning
===========================================================
Updates the simulation to use RRT* for path planning instead of A*
"""

from multiagent_drone_sim import MultiAgentDroneSimulation
from rrt_integration import create_rrt_planner
import numpy as np
import random
import time
import matplotlib.pyplot as plt

class RRTStarDroneSimulation(MultiAgentDroneSimulation):
    """Extended drone simulation with RRT* path planning"""
    
    def __init__(self, n_surveyors=3, n_workers=2, space_limit=10.0):
        """Initialize simulation with RRT* path planning"""
        # Initialize parent class
        super().__init__(n_surveyors, n_workers, space_limit)
        
        # Replace the default planner with RRT* planner
        space_limits = (
            self.x_min, self.x_max, 
            self.y_min, self.y_max,
            self.z_min, self.z_max
        )
        
        # Create the RRT* planner - replaces the A* based planner
        self.planner = create_rrt_planner(
            self.planner.shared_map,  # Reuse the shared map
            space_limits,
            n_surveyors,
            n_workers
        )
        
        # Additional metrics specific to RRT*
        self.metrics.update({
            "path_smoothness": [],
            "planning_time": None,
            "replanning_count": 0
        })

        self.drone_stuck_counters = [0] * self.n_drones
        self.stuck_threshold = 30  # After this many stuck frames, take action
        self.max_simulation_frames = 1000  # Maximum frames before forced end
        self.frame_count = 0
        self.last_positions = [None] * self.n_drones
        self.position_change_threshold = 0.1 
        
        print("Initialized RRT* based drone simulation")
    
    def update_known_map(self, drone_id, drone_pos):
        """Update the known map based on drone's current position"""
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
    
    def control_loop(self, frame_idx):
        """Modified control loop with proper end condition and stuck detection"""
        if self.simulation_start_time is None:
            self.simulation_start_time = time.time()
        
        # Increment frame counter    
        self.frame_count += 1
        
        # Add debugging output every 30 frames
        if frame_idx % 30 == 0:  # Every second (assuming 30fps)
            print(f"Frame {frame_idx}: Drone positions:")
            for i, drone in enumerate(self.drones):
                print(f"  Drone {i} ({self.drone_roles[i]}): {drone.position()}")
            
            # Print exploration statistics
            stats = self.planner.get_exploration_stats(self.truth_map)
            print(f"  Coverage: {stats['coverage_percent']:.2f}%")
            print(f"  Phase: {self.planner.planner.phase}")
            if self.planner.planner.target_pos is not None:
                print(f"  Target found at: {self.planner.planner.target_pos}")
        
        # Check for simulation timeout condition
        if self.frame_count >= self.max_simulation_frames:
            print(f"\n=== Simulation reached maximum frame count: {self.frame_count} ===")
            self.completion_time = time.time() - self.simulation_start_time
            print(f"Total mission time: {self.completion_time:.2f} seconds")
            self.print_metrics()
            self.visualize_final_exploration()
            return None
        
        # Check if all worker drones have reached the target
        if self.planner.planner.target_pos is not None:
            all_workers_done = True
            for i in range(self.n_surveyors, self.n_drones):
                if not self.done[i]:
                    current_pos = np.array(self.drones[i].position())
                    target_pos = self.planner.planner.target_pos
                    dist_to_target = np.linalg.norm(current_pos - target_pos)
                    
                    # Check if drone is very close to target
                    if dist_to_target < self.SPEED * 3:
                        print(f"Worker {i - self.n_surveyors} has reached the target!")
                        self.done[i] = True
                    else:
                        all_workers_done = False
            
            # If all workers are done, end simulation
            if all_workers_done:
                if not hasattr(self, 'completion_time'):
                    self.completion_time = time.time() - self.simulation_start_time
                    print(f"\n=== Mission Complete! ===")
                    print(f"All worker drones have reached the target.")
                    print(f"Total mission time: {self.completion_time:.2f} seconds")
                    self.print_metrics()
                    self.visualize_final_exploration()
                return None
        
        # Check if simulation is already complete from previous frame
        if all(self.done):
            return None
            
        world_frames = []
        for i in range(self.n_drones):
            if self.done[i]:
                world_frames.append(self.drones[i].world_frame())
                continue
                
            current_pos = np.array(self.drones[i].position())
            
            # Check if drone is stuck by comparing with last position
            if self.last_positions[i] is not None:
                movement = np.linalg.norm(current_pos - self.last_positions[i])
                if movement < self.position_change_threshold:
                    self.drone_stuck_counters[i] += 1
                    
                    # If worker drone is stuck for too long and target is known, teleport it closer to target
                    if i >= self.n_surveyors and self.drone_stuck_counters[i] >= self.stuck_threshold and self.planner.planner.target_pos is not None:
                        target_pos = self.planner.planner.target_pos
                        
                        # Create a position between current and target
                        direction = target_pos - current_pos
                        distance = np.linalg.norm(direction)
                        
                        if distance > self.SPEED * 10:  # If significantly far from target
                            # Move 30% of the way to the target
                            new_pos = current_pos + direction * 0.3
                            print(f"Worker {i - self.n_surveyors} was stuck too long, teleporting closer to target")
                            self.drones[i].state[0:3] = new_pos
                            current_pos = new_pos
                            self.drone_stuck_counters[i] = 0  # Reset counter
                else:
                    # Reset stuck counter if drone moved
                    self.drone_stuck_counters[i] = 0
            
            # Update last position
            self.last_positions[i] = current_pos.copy()
            
            # Update known map based on sensing
            self.update_known_map(i, current_pos)
            
            # Get next waypoint from RRT* planner based on drone role
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
                # Trigger replanning in RRT* planner
                if self.planner.paths_planned:
                    self.planner.replanning_needed = True
                    self.metrics["replanning_count"] += 1
                
                # For stuck drones, move randomly to try to escape
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
                if role == 'worker' and self.planner.planner.target_pos is not None:
                    # Check if worker has reached the target
                    target_dist = np.linalg.norm(current_pos - self.planner.planner.target_pos)
                    if target_dist < self.SPEED * 3:  # Increased the threshold to detect reaching target
                        print(f"Worker {role_id} has reached the target!")
                        self.done[i] = True
                
                current_pos = target_pos
            else:
                # Regular movement code 
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
                        # Reduce the impact of avoidance forces
                        avoidance += force * 0.5
                
                # Combine goal direction with avoidance
                if np.any(avoidance):
                    movement = direction + avoidance
                    movement = movement / np.linalg.norm(movement)
                else:
                    movement = direction
                
                # Increase movement speed for worker drones heading to target
                if role == 'worker' and self.planner.planner.phase == "execution":
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
                
            # Record planning time
            if hasattr(self.planner.planner, 'rrt_planner') and \
                hasattr(self.planner.planner.rrt_planner, 'planning_time'):
                self.metrics["planning_time"] = self.planner.planner.rrt_planner.planning_time
            
            # Compute path smoothness
            path_smoothness = []
            for i in range(self.n_drones):
                if hasattr(self.planner.planner, 'rrt_planner') and \
                    hasattr(self.planner.planner.rrt_planner, 'paths') and \
                    self.planner.planner.rrt_planner.paths[i] is not None and \
                    len(self.planner.planner.rrt_planner.paths[i]) >= 3:
                    path = self.planner.planner.rrt_planner.paths[i]
                    # Calculate angle changes between segments as a measure of smoothness
                    angles = []
                    for j in range(1, len(path) - 1):
                        v1 = path[j] - path[j-1]
                        v2 = path[j+1] - path[j]
                        v1_norm = np.linalg.norm(v1)
                        v2_norm = np.linalg.norm(v2)
                        if v1_norm > 0 and v2_norm > 0:
                            cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                            # Clip to handle floating point errors
                            cos_angle = np.clip(cos_angle, -1.0, 1.0)
                            angle = np.arccos(cos_angle)
                            angles.append(angle)
                    
                    if angles:
                        # Lower angles = smoother paths
                        path_smoothness.append(np.mean(angles))
            
            if path_smoothness:
                self.metrics["path_smoothness"].append(np.mean(path_smoothness))
            
            # Check if any worker has reached the target
            if self.planner.planner.target_pos is not None:
                for i in range(self.n_surveyors, self.n_drones):
                    if not self.done[i]:
                        worker_pos = np.array(self.drones[i].position())
                        dist_to_target = np.linalg.norm(worker_pos - self.planner.planner.target_pos)
                        if dist_to_target < self.SPEED * 3:  # Increased threshold
                            self.done[i] = True
                            print(f"Worker {i - self.n_surveyors} has reached the target!")
        
        return np.stack(world_frames)
    
    def print_metrics(self):
        """Print the simulation performance metrics including RRT* specific metrics"""
        print("\n=== Simulation Performance Metrics (RRT*) ===")
        print(f"Time to Target Detection: {self.metrics['time_to_target']:.2f} seconds")
        print(f"Final Coverage Rate: {self.metrics['coverage_rate'][-1]:.2f}%")
        
        if self.metrics['map_accuracy']:
            print(f"Final Map Accuracy (IoU): {self.metrics['map_accuracy'][-1]:.2f}")
        
        print(f"Collision Rate: {self.metrics['collision_rate']}")
        print(f"Replanning Count: {self.metrics['replanning_count']}")
        
        if self.metrics['planning_time'] is not None:
            print(f"Path Planning Time: {self.metrics['planning_time']:.2f} seconds")
        
        if self.metrics['path_smoothness']:
            print(f"Path Smoothness (avg angle change): {self.metrics['path_smoothness'][-1]:.2f} radians")
        
        # Plot enhanced metrics
        self.plot_enhanced_metrics()
    
    def plot_enhanced_metrics(self):
        """Plot enhanced metrics including RRT* specific metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Coverage rate
        x = list(range(len(self.metrics["coverage_rate"])))
        ax1.plot(x, self.metrics["coverage_rate"], 'b-', label='Coverage Rate (%)')
        ax1.set_xlabel('Frame (x10)')
        ax1.set_ylabel('Percentage')
        ax1.set_title('Coverage Rate Over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Map accuracy
        if self.metrics["map_accuracy"]:
            ax2.plot(x, self.metrics["map_accuracy"], 'r-', label='Map Accuracy (IoU)')
            ax2.set_xlabel('Frame (x10)')
            ax2.set_ylabel('IoU Score')
            ax2.set_title('Map Accuracy Over Time')
            ax2.legend()
            ax2.grid(True)
        
        # Path smoothness
        if self.metrics["path_smoothness"]:
            x_smooth = list(range(len(self.metrics["path_smoothness"])))
            ax3.plot(x_smooth, self.metrics["path_smoothness"], 'g-', label='Path Smoothness')
            ax3.set_xlabel('Frame (x10)')
            ax3.set_ylabel('Average Angle (rad)')
            ax3.set_title('Path Smoothness Over Time')
            ax3.legend()
            ax3.grid(True)
        
        # Collision count histogram
        role_types = ['Surveyor'] * self.n_surveyors + ['Worker'] * self.n_workers
        role_counts = {'Surveyor': self.n_surveyors, 'Worker': self.n_workers}
        counts = [role_counts['Surveyor'], role_counts['Worker']]
        bar_positions = np.arange(len(counts))
        ax4.bar(bar_positions, counts, color=['blue', 'orange'])
        ax4.set_xticks(bar_positions)
        ax4.set_xticklabels(['Surveyors', 'Workers'])
        ax4.set_ylabel('Count')
        ax4.set_title('Drone Distribution')
        ax4.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig('rrt_star_metrics.png')
        plt.show()


    def visualize_final_exploration(self):
        """Visualize the final exploration results with multiple views"""
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D
        
        # 1. Create a 3D plot of the final exploration state
        fig = plt.figure(figsize=(15, 12))
        
        # Plot 1: 3D scatter plot of explored voxels
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Get the known map
        known_map = self.planner.planner.shared_map.grid
        
        # Plot different types of voxels with different colors
        free_mask = (known_map == 0)
        obstacle_mask = (known_map == 1)
        target_mask = (known_map == 2)
        unknown_mask = (known_map == 255)
        
        # Create grid coordinates
        X, Y, Z = np.meshgrid(
            np.linspace(self.x_min, self.x_max, known_map.shape[0]),
            np.linspace(self.y_min, self.y_max, known_map.shape[1]),
            np.linspace(self.z_min, self.z_max, known_map.shape[2]),
            indexing='ij'
        )
        
        # Plot each type - using sparse sampling for better visibility
        sample_rate = 5  # Sample every 5th voxel for clearer visualization
        
        # Free space (gray)
        ax1.scatter(
            X[free_mask][::sample_rate], 
            Y[free_mask][::sample_rate], 
            Z[free_mask][::sample_rate], 
            c='gray', marker='.', alpha=0.3, label='Free'
        )
        
        # Obstacles (red)
        ax1.scatter(
            X[obstacle_mask], 
            Y[obstacle_mask], 
            Z[obstacle_mask], 
            c='red', marker='s', alpha=0.7, label='Obstacle'
        )
        
        # Target (green)
        ax1.scatter(
            X[target_mask], 
            Y[target_mask], 
            Z[target_mask], 
            c='green', marker='*', s=100, label='Target'
        )
        
        # Plot drone paths if available
        for i in range(self.n_drones):
            if i < self.n_surveyors:
                path_color = 'blue'
                label = f'Surveyor {i} Path'
            else:
                path_color = 'orange'
                label = f'Worker {i-self.n_surveyors} Path'
                
            # Get path for this drone
            if hasattr(self.planner.planner, 'rrt_planner') and \
            hasattr(self.planner.planner.rrt_planner, 'paths') and \
            self.planner.planner.rrt_planner.paths[i] is not None:
                path = np.array(self.planner.planner.rrt_planner.paths[i])
                if path.shape[0] > 1:
                    ax1.plot(path[:, 0], path[:, 1], path[:, 2], color=path_color, linestyle='-', linewidth=2, label=label)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Exploration Map')
        ax1.legend()
        
        # Plot 2: 2D Slice - XY plane at target Z
        ax2 = fig.add_subplot(222)
        
        # Get Z slice at target position if available
        target_z = int(self.planner.planner.target_grid_pos[2]) if self.planner.planner.target_grid_pos is not None else known_map.shape[2] // 2
        
        # Create 2D slice of the known map
        z_slice = known_map[:, :, target_z]
        
        # Create custom colormap for the 2D slice
        cmap = plt.cm.colors.ListedColormap(['lightgray', 'red', 'green', 'white'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 255.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot the slice
        im = ax2.imshow(
            z_slice.T,  # Transpose for correct orientation
            origin='lower',
            extent=[self.x_min, self.x_max, self.y_min, self.y_max],
            cmap=cmap,
            norm=norm
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, ticks=[0, 1, 2, 255])
        cbar.set_ticklabels(['Free', 'Obstacle', 'Target', 'Unknown'])
        
        # If target is found, add X and Y markers on the 2D plot
        if self.planner.planner.target_pos is not None:
            ax2.plot(self.planner.planner.target_pos[0], self.planner.planner.target_pos[1], 'k*', markersize=15)
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title(f'XY Plane Slice (Z={target_z})')
        
        # Plot 3: Coverage over time
        ax3 = fig.add_subplot(223)
        x = list(range(len(self.metrics["coverage_rate"])))
        time_points = [i * 10 * self.dt for i in x]  # Convert frame indices to seconds
        
        ax3.plot(time_points, self.metrics["coverage_rate"], 'b-', linewidth=2)
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Coverage (%)')
        ax3.set_title('Exploration Coverage Over Time')
        ax3.grid(True)
        
        # Plot 4: Path smoothness over time (if available)
        ax4 = fig.add_subplot(224)
        
        if self.metrics["path_smoothness"]:
            smooth_x = list(range(len(self.metrics["path_smoothness"])))
            smooth_time = [i * 10 * self.dt for i in smooth_x]  # Convert frame indices to seconds
            
            ax4.plot(smooth_time, self.metrics["path_smoothness"], 'g-', linewidth=2)
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Average Path Angle (rad)')
            ax4.set_title('Path Smoothness Over Time')
            ax4.grid(True)
        else:
            # If no smoothness data, show exploration accuracy instead
            if self.metrics["map_accuracy"]:
                ax4.plot(time_points, self.metrics["map_accuracy"], 'r-', linewidth=2)
                ax4.set_xlabel('Time (seconds)')
                ax4.set_ylabel('Map Accuracy (IoU)')
                ax4.set_title('Map Accuracy Over Time')
                ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('final_exploration_results.png', dpi=150)
        plt.show()
        
        print("Final exploration results saved to 'final_exploration_results.png'")
        
        # Also create a separate path visualization
        self.visualize_drone_paths()

    def visualize_drone_paths(self):
        """Create a dedicated visualization of drone paths"""
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot drone paths
        for i in range(self.n_drones):
            if i < self.n_surveyors:
                path_color = 'blue'
                label = f'Surveyor {i}'
            else:
                path_color = 'orange'
                label = f'Worker {i-self.n_surveyors}'
                
            # Get path for this drone
            if hasattr(self.planner.planner, 'rrt_planner') and \
            hasattr(self.planner.planner.rrt_planner, 'paths') and \
            self.planner.planner.rrt_planner.paths[i] is not None:
                path = np.array(self.planner.planner.rrt_planner.paths[i])
                if path.shape[0] > 1:
                    ax.plot(path[:, 0], path[:, 1], path[:, 2], color=path_color, linestyle='-', linewidth=2, label=label)
                    # Add start and end markers
                    ax.scatter(path[0, 0], path[0, 1], path[0, 2], color=path_color, marker='o', s=50)
                    ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], color=path_color, marker='s', s=50)
        
        # If target is found, mark it
        if self.planner.planner.target_pos is not None:
            target_pos = self.planner.planner.target_pos
            ax.scatter(target_pos[0], target_pos[1], target_pos[2], color='green', marker='*', s=200, label='Target')
        
        # Mark obstacles
        known_map = self.planner.planner.shared_map.grid
        obstacle_mask = (known_map == 1)
        
        # Create grid coordinates
        X, Y, Z = np.meshgrid(
            np.linspace(self.x_min, self.x_max, known_map.shape[0]),
            np.linspace(self.y_min, self.y_max, known_map.shape[1]),
            np.linspace(self.z_min, self.z_max, known_map.shape[2]),
            indexing='ij'
        )
        
        # Plot obstacles
        ax.scatter(
            X[obstacle_mask], 
            Y[obstacle_mask], 
            Z[obstacle_mask], 
            c='red', marker='s', alpha=0.3, label='Obstacles'
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Drone Paths')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('drone_paths.png', dpi=150)
        plt.show()
        
        print("Drone paths visualization saved to 'drone_paths.png'")