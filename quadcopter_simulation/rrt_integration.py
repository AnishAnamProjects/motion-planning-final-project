"""
Integration of Multi-Agent RRT* with Drone Simulation
====================================================
This module integrates the RRT* path planner with the existing simulation framework.
"""

import numpy as np
import random
import time
from multiagent_rrt_star import MultiAgentRRTStar, MultiAgentPathPlanner

class RRTStarPlanner:
    """Wrapper class to integrate RRT* planning with the existing MultiAgentPlanner"""
    
    def __init__(self, shared_map, space_limits, n_surveyors, n_workers):
        """
        Initialize the RRT* planner
        
        Args:
            shared_map: SharedMap instance
            space_limits: (x_min, x_max, y_min, y_max, z_min, z_max) environment bounds
            n_surveyors: Number of surveyor drones
            n_workers: Number of worker drones
        """
        self.shared_map = shared_map
        self.space_limits = space_limits
        self.n_surveyors = n_surveyors
        self.n_workers = n_workers
        self.n_drones = n_surveyors + n_workers
        self.replanning_count = 0  
        
        # Store drone positions
        self.drone_positions = [None] * self.n_drones
        
        # Initialize RRT* planner
        self.planner = MultiAgentPathPlanner(
            shared_map,
            space_limits,
            n_surveyors,
            n_workers
        )
        
        # Internal state tracking
        self.paths_planned = False
        self.replanning_needed = False
        self.last_replanning_time = 0
        self.replanning_interval = 5.0  # seconds
        
        # Exploration behavior for surveyors
        self.exploration_goals = [None] * n_surveyors
        self.exploration_radius = 2.0
        
        print(f"RRT* planner initialized with {n_surveyors} surveyors and {n_workers} workers")
    
    def update_drone_position(self, drone_id, position):
        """Update stored drone position"""
        self.drone_positions[drone_id] = position
    
    def update_map(self, drone_id, pos, observed_cells):
        """Update shared map with observations"""
        grid_pos = self.world_to_grid(pos)
        self.planner.update_map(drone_id, pos, grid_pos, observed_cells)
        
        # If target found and paths not yet planned, or replanning needed
        if self.planner.target_pos is not None and \
           (not self.paths_planned or (self.replanning_needed and 
            time.time() - self.last_replanning_time > self.replanning_interval)):
            self.plan_paths()
    
    def world_to_grid(self, pos):
        """Convert world coordinates to grid indices"""
        return self.planner.world_to_grid(pos)
    
    def grid_to_world(self, grid_pos):
        """Convert grid indices to world coordinates"""
        return self.planner.grid_to_world(grid_pos)
    
    def get_current_positions(self):
        """Get current positions of all drones"""
        return [pos if pos is not None else np.zeros(3) for pos in self.drone_positions]
    
    def generate_exploration_goals(self):
        """Generate better exploration goals for surveyor drones"""
        x_min, x_max, y_min, y_max, z_min, z_max = self.space_limits
        
        # Get the known map to find unexplored areas
        known_map = self.planner.shared_map.grid
        
        # For each surveyor, assign a different exploration zone
        # Divide environment into sectors to ensure broad coverage
        sectors = [
            # (x_range, y_range, z_range) for each sector
            ((x_min, x_max/2), (y_min, y_max/2), (z_min, z_max/2)),  # Bottom front left
            ((x_max/2, x_max), (y_min, y_max/2), (z_min, z_max/2)),  # Bottom front right
            ((x_min, x_max/2), (y_max/2, y_max), (z_min, z_max/2)),  # Bottom back left
            ((x_max/2, x_max), (y_max/2, y_max), (z_min, z_max/2)),  # Bottom back right
            ((x_min, x_max/2), (y_min, y_max/2), (z_max/2, z_max)),  # Top front left
            ((x_max/2, x_max), (y_min, y_max/2), (z_max/2, z_max)),  # Top front right
            ((x_min, x_max/2), (y_max/2, y_max), (z_max/2, z_max)),  # Top back left
            ((x_max/2, x_max), (y_max/2, y_max), (z_max/2, z_max)),  # Top back right
        ]
        
        for i in range(self.n_surveyors):
            # If drone has reached current goal or no goal yet assigned
            if (self.exploration_goals[i] is None or 
                (self.drone_positions[i] is not None and 
                np.linalg.norm(self.drone_positions[i] - self.exploration_goals[i]) < self.exploration_radius)):
                
                # Assign each drone to different sector, cycling through them
                sector_idx = (i + self.replanning_count) % len(sectors)
                target_sector = sectors[sector_idx]
                
                # Try to find unexplored cells in this sector
                sector_x_range, sector_y_range, sector_z_range = target_sector
                
                # Convert sector bounds to grid indices
                grid_x_min = int((sector_x_range[0] - x_min) / (x_max - x_min) * (known_map.shape[0] - 1))
                grid_x_max = int((sector_x_range[1] - x_min) / (x_max - x_min) * (known_map.shape[0] - 1))
                grid_y_min = int((sector_y_range[0] - y_min) / (y_max - y_min) * (known_map.shape[1] - 1))
                grid_y_max = int((sector_y_range[1] - y_min) / (y_max - y_min) * (known_map.shape[1] - 1))
                grid_z_min = int((sector_z_range[0] - z_min) / (z_max - z_min) * (known_map.shape[2] - 1))
                grid_z_max = int((sector_z_range[1] - z_min) / (z_max - z_min) * (known_map.shape[2] - 1))
                
                # Find unexplored cells in this sector
                sector_unexplored = []
                for ix in range(grid_x_min, grid_x_max + 1):
                    for iy in range(grid_y_min, grid_y_max + 1):
                        for iz in range(grid_z_min, grid_z_max + 1):
                            if known_map[ix, iy, iz] == 255:  # Unexplored
                                sector_unexplored.append((ix, iy, iz))
                
                # If there are unexplored cells in this sector, pick one randomly
                if sector_unexplored:
                    target_cell = sector_unexplored[np.random.randint(len(sector_unexplored))]
                    self.exploration_goals[i] = self.grid_to_world(target_cell)
                    print(f"Surveyor {i} assigned to unexplored cell in sector {sector_idx}")
                    continue
                
                # If no unexplored cells in preferred sector, find frontier cells in the sector
                sector_frontiers = []
                for ix in range(grid_x_min, grid_x_max + 1):
                    for iy in range(grid_y_min, grid_y_max + 1):
                        for iz in range(grid_z_min, grid_z_max + 1):
                            if known_map[ix, iy, iz] == 0:  # Free space
                                # Check neighbors for unexplored cells
                                for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                                    nx, ny, nz = ix + dx, iy + dy, iz + dz
                                    if (0 <= nx < known_map.shape[0] and 
                                        0 <= ny < known_map.shape[1] and 
                                        0 <= nz < known_map.shape[2] and
                                        known_map[nx, ny, nz] == 255):
                                        sector_frontiers.append((ix, iy, iz))
                                        break
                
                # If there are frontier cells in this sector, pick one randomly
                if sector_frontiers:
                    target_cell = sector_frontiers[np.random.randint(len(sector_frontiers))]
                    self.exploration_goals[i] = self.grid_to_world(target_cell)
                    print(f"Surveyor {i} assigned to frontier cell in sector {sector_idx}")
                    continue
                
                # If no unexplored or frontier cells in preferred sector, pick a random point in the sector
                rand_x = np.random.uniform(sector_x_range[0], sector_x_range[1])
                rand_y = np.random.uniform(sector_y_range[0], sector_y_range[1])
                rand_z = np.random.uniform(sector_z_range[0], sector_z_range[1])
                self.exploration_goals[i] = np.array([rand_x, rand_y, rand_z])
                print(f"Surveyor {i} assigned to random point in sector {sector_idx}")
        
    def plan_paths(self):
        """Plan paths for all drones using RRT* with improved surveyor goals"""
        print("\n=== PLANNING PATHS WITH RRT* ===")
        
        # Get current positions
        start_positions = self.get_current_positions()
        
        # Generate goal positions
        goal_positions = [None] * self.n_drones
        
        # Check if target position is known
        if self.planner.target_pos is not None:
            # Set target as goal for worker drones
            for i in range(self.n_workers):
                drone_id = self.n_surveyors + i
                goal_positions[drone_id] = self.planner.target_pos
        
            # Generate better exploration goals for surveyor drones
            self.generate_exploration_goals()
            for i in range(self.n_surveyors):
                goal_positions[i] = self.exploration_goals[i]
            
            # Print info
            print("Planning with:")
            for i in range(self.n_drones):
                role = "surveyor" if i < self.n_surveyors else "worker"
                print(f"Drone {i} ({role}):")
                print(f"  From: {start_positions[i]}")
                print(f"  To: {goal_positions[i]}")
            
            # Call RRT* planner
            self.planner.rrt_planner.plan_paths(start_positions, goal_positions)
            
            # Mark planning complete
            self.paths_planned = True
            self.replanning_needed = False
            self.last_replanning_time = time.time()

            self.replanning_count += 1 
        else:
            print("Cannot plan paths: Target position not yet known")
    
    def get_next_waypoint(self, drone_id, drone_type, current_pos):
        """Get next waypoint for a drone"""
        # Update drone position
        real_drone_id = drone_id if drone_type == 'surveyor' else drone_id + self.n_surveyors
        self.update_drone_position(real_drone_id, current_pos)
        
        # Get next waypoint from planner
        return self.planner.get_next_waypoint(drone_id, drone_type, current_pos)
    
    def record_collision(self):
        """Record a collision event"""
        self.planner.record_collision()
        
        # Trigger replanning on collision
        self.replanning_needed = True
    
    def get_exploration_stats(self, truth_map=None):
        """Return exploration statistics"""
        return self.planner.get_exploration_stats(truth_map)


def create_rrt_planner(shared_map, space_limits, n_surveyors, n_workers):
    """Factory function to create an RRT* planner instance"""
    return RRTStarPlanner(shared_map, space_limits, n_surveyors, n_workers)
