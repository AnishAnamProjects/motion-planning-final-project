"""
Multi-agent Path Planning for Quadcopter Swarm
==============================================
This module implements a multi-agent exploration and path planning system with:
- Surveyor drones that explore and build a shared map
- Worker drones that use the map to reach targets
- Performance evaluation metrics
"""

import numpy as np
import random
from collections import defaultdict, deque
import heapq
from threading import Lock
import time

class SharedMap:
    """Shared environment map for all drones to contribute to and read from"""
    
    def __init__(self, nx, ny, nz):
        """Initialize the shared map with dimensions nx, ny, nz"""
        self.nx = nx
        self.ny = ny
        self.nz = nz
        # Occupancy grid: 0=free, 1=obstacle, 2=goal/target, 255=unknown
        self.grid = 255 * np.ones((nx, ny, nz), dtype=np.uint8)
        # Map access lock to prevent race conditions during updates
        self.lock = Lock()
        # Exploration record - which drone has visited which cells
        self.explored_by = {}
        # Store timestamps of when cells were first explored
        self.exploration_timestamps = {}
        # Start time of exploration
        self.start_time = time.time()
        # Time when target was found (if any)
        self.target_found_time = None
        # Collision counter
        self.collisions = 0
        
    def update(self, drone_id, pos, grid_indices, observed_cells):
        """Update the map with new observations from a drone
        
        Args:
            drone_id: ID of the reporting drone
            pos: World position of the drone
            grid_indices: (ix, iy, iz) grid cell containing the drone
            observed_cells: Dictionary mapping (ix,iy,iz) -> cell_status
        """
        with self.lock:
            # Record which drone explored each cell and when
            current_time = time.time() - self.start_time
            
            # Update the map with observed cells
            for (ix, iy, iz), status in observed_cells.items():
                cell_key = (ix, iy, iz)
                
                # Only record unexplored cells
                if self.grid[ix, iy, iz] == 255:
                    self.explored_by[cell_key] = drone_id
                    self.exploration_timestamps[cell_key] = current_time
                
                # Update the grid value
                self.grid[ix, iy, iz] = status
                
                # Mark when target is found
                if status == 2 and self.target_found_time is None:
                    self.target_found_time = current_time
                    print(f"Target found by drone {drone_id} at time {current_time:.2f}s")
    
    def record_collision(self):
        """Increment collision counter"""
        with self.lock:
            self.collisions += 1
    
    def get_explored_percentage(self):
        """Calculate the percentage of environment explored"""
        with self.lock:
            total_cells = self.nx * self.ny * self.nz
            explored_cells = np.sum(self.grid != 255)
            return (explored_cells / total_cells) * 100
    
    def get_target_found_time(self):
        """Return the time it took to find the target"""
        return self.target_found_time
    
    def get_collision_count(self):
        """Return the number of collisions"""
        return self.collisions
    
    def calculate_map_accuracy(self, truth_map):
        """Calculate map accuracy using Intersection over Union (IoU)
        
        Args:
            truth_map: Ground truth environment map
            
        Returns:
            float: IoU score (0-1 where 1 is perfect match)
        """
        with self.lock:
            # Convert to binary maps (obstacle vs not obstacle)
            known_obstacle = (self.grid == 1)
            truth_obstacle = (truth_map == 1)
            
            # Calculate intersection and union
            intersection = np.logical_and(known_obstacle, truth_obstacle).sum()
            union = np.logical_or(known_obstacle, truth_obstacle).sum()
            
            if union == 0:  # No obstacles in either map
                return 1.0
                
            return intersection / union
            
    def calculate_exploration_redundancy(self):
        """Calculate the redundancy in exploration
        
        Returns:
            float: Average number of redundant visits per explored cell
        """
        # This is a placeholder - in a real implementation, we would track
        # every visit to each cell by each drone
        return 0.0
    
    def find_path(self, start, goal):
        """Find an optimal path through the known map using A* algorithm
        
        Args:
            start: (x,y,z) start position in grid coordinates
            goal: (x,y,z) goal position in grid coordinates
            
        Returns:
            list: List of (x,y,z) positions forming the path, or None if no path
        """
        start = tuple(start)
        goal = tuple(goal)
        
        # If start or goal are in unknown or obstacle space, no path
        if self.grid[start] == 255 or self.grid[start] == 1:
            return None
        if self.grid[goal] == 255 or self.grid[goal] == 1:
            return None
        
        # A* algorithm
        open_set = []
        closed_set = set()
        g_score = {start: 0}  # Cost from start to current
        f_score = {start: self._heuristic(start, goal)}  # Estimated total cost
        came_from = {}
        
        heapq.heappush(open_set, (f_score[start], start))
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            closed_set.add(current)
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                    
                # Check if neighbor is valid
                nx, ny, nz = neighbor
                if (0 <= nx < self.nx and 0 <= ny < self.ny and 0 <= nz < self.nz):
                    # Skip obstacles and unknown cells
                    if self.grid[nx, ny, nz] == 1 or self.grid[nx, ny, nz] == 255:
                        continue
                
                tentative_g = g_score[current] + 1  # Assuming uniform cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        return None
    
    def _heuristic(self, a, b):
        """Manhattan distance heuristic for A* algorithm"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
    
    def _get_neighbors(self, pos):
        """Get valid neighboring grid cells"""
        x, y, z = pos
        potential_neighbors = [
            (x+1, y, z), (x-1, y, z),
            (x, y+1, z), (x, y-1, z),
            (x, y, z+1), (x, y, z-1)
        ]
        
        return [n for n in potential_neighbors if 
                0 <= n[0] < self.nx and 
                0 <= n[1] < self.ny and 
                0 <= n[2] < self.nz]
    
    def calculate_path_optimality(self, start, goal, truth_map):
        """Calculate path optimality ratio
        
        Args:
            start: (x,y,z) start position in grid coordinates
            goal: (x,y,z) goal position in grid coordinates
            truth_map: Ground truth map to compute optimal path
            
        Returns:
            float: Ratio of optimal path length to found path length
        """
        # Create temporary SharedMap with perfect knowledge for optimal path
        optimal_map = SharedMap(self.nx, self.ny, self.nz)
        optimal_map.grid = truth_map.copy()
        
        # Find path using current knowledge
        current_path = self.find_path(start, goal)
        
        # Find optimal path using ground truth
        optimal_path = optimal_map.find_path(start, goal)
        
        if current_path is None:
            return 0.0  # No path found with current knowledge
            
        if optimal_path is None:
            return 1.0  # No optimal path exists either (strange case)
            
        # Calculate ratio (optimal path length / current path length)
        # Lower is better - values less than 1 indicate current path is longer
        return len(optimal_path) / len(current_path)


class ExplorationPlanner:
    """Generates exploration waypoints for surveyor drones"""
    
    def __init__(self, shared_map, space_limits, drone_id):
        """Initialize exploration planner
        
        Args:
            shared_map: SharedMap instance for collaborative mapping
            space_limits: (x_min, x_max, y_min, y_max, z_min, z_max) environment bounds
            drone_id: ID of the drone using this planner
        """
        self.shared_map = shared_map
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = space_limits
        self.drone_id = drone_id
        
        # Frontier cells (unexplored cells adjacent to explored free space)
        self.frontier = set()
        
        # Probability of random exploration vs frontier-based
        self.random_exploration_prob = 0.2
        
        # Exploration pattern - can be 'frontier', 'random', or 'spiral'
        self.exploration_pattern = 'frontier'
        
        # Parameters for spiral pattern
        self.spiral_center = np.array([(self.x_max + self.x_min) / 2,
                                      (self.y_max + self.y_min) / 2,
                                      (self.z_max + self.z_min) / 2])
        self.spiral_radius = 0.0
        self.spiral_angle = 0.0
        self.spiral_height = self.z_min + 1.0
        
    def world_to_grid(self, pos):
        """Convert world coordinates to grid indices"""
        nx, ny, nz = self.shared_map.nx, self.shared_map.ny, self.shared_map.nz
        ix = int((pos[0] - self.x_min) / (self.x_max - self.x_min) * (nx - 1))
        iy = int((pos[1] - self.y_min) / (self.y_max - self.y_min) * (ny - 1))
        iz = int((pos[2] - self.z_min) / (self.z_max - self.z_min) * (nz - 1))
        return (np.clip(ix, 0, nx-1), np.clip(iy, 0, ny-1), np.clip(iz, 0, nz-1))
    
    def grid_to_world(self, grid_pos):
        """Convert grid indices to world coordinates"""
        ix, iy, iz = grid_pos
        nx, ny, nz = self.shared_map.nx, self.shared_map.ny, self.shared_map.nz
        x = self.x_min + (ix / (nx - 1)) * (self.x_max - self.x_min)
        y = self.y_min + (iy / (ny - 1)) * (self.y_max - self.y_min)
        z = self.z_min + (iz / (nz - 1)) * (self.z_max - self.z_min)
        return np.array([x, y, z])
    
    def update_frontiers(self):
        """Update the set of frontier cells based on current map"""
        grid = self.shared_map.grid
        new_frontier = set()
        
        # Check all known free cells for adjacent unknown cells
        for ix in range(self.shared_map.nx):
            for iy in range(self.shared_map.ny):
                for iz in range(self.shared_map.nz):
                    if grid[ix, iy, iz] == 0:  # Free cell
                        # Check neighbors
                        for nx, ny, nz in [(ix+1, iy, iz), (ix-1, iy, iz),
                                         (ix, iy+1, iz), (ix, iy-1, iz),
                                         (ix, iy, iz+1), (ix, iy, iz-1)]:
                            if (0 <= nx < self.shared_map.nx and
                                0 <= ny < self.shared_map.ny and
                                0 <= nz < self.shared_map.nz and
                                grid[nx, ny, nz] == 255):  # Unknown cell
                                new_frontier.add((nx, ny, nz))
        
        self.frontier = new_frontier
    
    def get_next_waypoint(self, current_pos):
        """Generate next exploration waypoint
        
        Args:
            current_pos: Current drone position in world coordinates
            
        Returns:
            numpy.ndarray: Next waypoint in world coordinates
        """
        # Update frontier cells
        self.update_frontiers()
        
        current_grid_pos = self.world_to_grid(current_pos)
        
        # Increase the random exploration chance
        self.random_exploration_prob = 0.4  # Increase from 0.2 to 0.4
        
        # Choose exploration method
        method = self.exploration_pattern
        if random.random() < self.random_exploration_prob:
            method = 'random'
        
        # Add a "jump" exploration strategy for when frontiers are limited
        if method == 'frontier' and (not self.frontier or random.random() < 0.3):
            # If no frontiers or 30% chance, do a "jump" to unexplored area
            # Find unexplored regions by checking the shared map
            unexplored_mask = self.shared_map.grid == 255  # Unknown areas
            if np.any(unexplored_mask):
                # Get indices of unexplored areas
                unexplored_indices = np.array(np.where(unexplored_mask)).T
                # Pick a random unexplored point
                if len(unexplored_indices) > 0:
                    random_idx = random.randint(0, len(unexplored_indices) - 1)
                    next_grid_pos = tuple(unexplored_indices[random_idx])
                    print(f"Drone {self.drone_id} jumping to unexplored area: {next_grid_pos}")
                    return self.grid_to_world(next_grid_pos)
        
        if method == 'frontier' and self.frontier:
            # Find closest frontier cell
            frontier_list = list(self.frontier)
            distances = [np.sum(np.abs(np.array(current_grid_pos) - np.array(f))) 
                        for f in frontier_list]
            closest_idx = np.argmin(distances)
            next_grid_pos = frontier_list[closest_idx]
            
        elif method == 'spiral':
            # Generate spiral exploration pattern
            self.spiral_angle += 0.2
            self.spiral_radius += 0.05
            
            # Calculate next point on spiral
            x = self.spiral_center[0] + self.spiral_radius * np.cos(self.spiral_angle)
            y = self.spiral_center[1] + self.spiral_radius * np.sin(self.spiral_angle)
            
            # Adjust height periodically
            if self.spiral_angle % (2 * np.pi) < 0.1:
                self.spiral_height += 1.0
                if self.spiral_height > self.z_max - 1:
                    self.spiral_height = self.z_min + 1.0
            
            next_world_pos = np.array([x, y, self.spiral_height])
            next_grid_pos = self.world_to_grid(next_world_pos)
            
        else:  # Random exploration
            # Generate random waypoint in free/unknown space
            while True:
                ix = random.randint(0, self.shared_map.nx - 1)
                iy = random.randint(0, self.shared_map.ny - 1)
                iz = random.randint(0, self.shared_map.nz - 1)
                
                # Check if cell is not an obstacle
                if self.shared_map.grid[ix, iy, iz] != 1:
                    next_grid_pos = (ix, iy, iz)
                    break
        
        # Convert grid position to world coordinates
        return self.grid_to_world(next_grid_pos)


class MultiAgentPlanner:
    """Main class for coordinating multi-agent exploration and task planning"""
    
    def __init__(self, n_surveyors, n_workers, space_limits, grid_dims=(20, 20, 20)):
        """Initialize the multi-agent planner
        
        Args:
            n_surveyors: Number of surveyor drones
            n_workers: Number of worker drones
            space_limits: (x_min, x_max, y_min, y_max, z_min, z_max) environment bounds
            grid_dims: (nx, ny, nz) grid dimensions
        """
        self.n_surveyors = n_surveyors
        self.n_workers = n_workers
        self.space_limits = space_limits
        
        # Initialize shared map
        nx, ny, nz = grid_dims
        self.shared_map = SharedMap(nx, ny, nz)
        
        # Create exploration planners for each surveyor
        self.surveyors = [ExplorationPlanner(self.shared_map, space_limits, i) 
                         for i in range(n_surveyors)]
        
        # Worker drones data
        self.worker_targets = [-1] * n_workers  # -1 means no target assigned
        self.worker_paths = [None] * n_workers
        
        # Target position (once found)
        self.target_pos = None
        self.target_grid_pos = None
        
        # State tracking
        self.phase = "exploration"  # exploration, assignment, or execution
        
        # Debug information
        print(f"Initialized MultiAgentPlanner with {n_surveyors} surveyors and {n_workers} workers")
        print(f"Grid dimensions: {grid_dims}")
        print(f"Space limits: {space_limits}")
        
    def world_to_grid(self, pos):
        """Convert world coordinates to grid indices"""
        x_min, x_max, y_min, y_max, z_min, z_max = self.space_limits
        nx, ny, nz = self.shared_map.nx, self.shared_map.ny, self.shared_map.nz
        
        ix = int((pos[0] - x_min) / (x_max - x_min) * (nx - 1))
        iy = int((pos[1] - y_min) / (y_max - y_min) * (ny - 1))
        iz = int((pos[2] - z_min) / (z_max - z_min) * (nz - 1))
        
        return (np.clip(ix, 0, nx-1), np.clip(iy, 0, ny-1), np.clip(iz, 0, nz-1))
    
    def grid_to_world(self, grid_pos):
        """Convert grid indices to world coordinates"""
        x_min, x_max, y_min, y_max, z_min, z_max = self.space_limits
        nx, ny, nz = self.shared_map.nx, self.shared_map.ny, self.shared_map.nz
        
        ix, iy, iz = grid_pos
        x = x_min + (ix / (nx - 1)) * (x_max - x_min)
        y = y_min + (iy / (ny - 1)) * (y_max - y_min)
        z = z_min + (iz / (nz - 1)) * (z_max - z_min)
        
        return np.array([x, y, z])
    
    def update_map(self, drone_id, pos, observed_cells):
        """Update shared map with observations
        
        Args:
            drone_id: ID of the reporting drone
            pos: World position of the drone
            observed_cells: Dictionary mapping (ix,iy,iz) -> cell_status
        """
        grid_pos = self.world_to_grid(pos)
        self.shared_map.update(drone_id, pos, grid_pos, observed_cells)
        
        # Check if target has been found
        for (ix, iy, iz), status in observed_cells.items():
            if status == 2 and self.target_pos is None:  # Target found
                self.target_grid_pos = (ix, iy, iz)
                self.target_pos = self.grid_to_world((ix, iy, iz))
                self.phase = "assignment"
                print(f"Target found at {self.target_pos}. Switching to assignment phase.")
    
    def record_collision(self):
        """Record a collision event"""
        self.shared_map.record_collision()
    
    def get_next_waypoint(self, drone_id, drone_type, current_pos):
        """Get next waypoint for a drone
        
        Args:
            drone_id: ID of the drone
            drone_type: 'surveyor' or 'worker'
            current_pos: Current position of the drone
            
        Returns:
            numpy.ndarray: Next waypoint in world coordinates
        """
        if drone_type == 'surveyor':
            if self.phase == "exploration":
                # During exploration phase, surveyors explore
                return self.surveyors[drone_id].get_next_waypoint(current_pos)
            else:
                # After target found, surveyors continue mapping but avoid target area
                waypoint = self.surveyors[drone_id].get_next_waypoint(current_pos)
                # If waypoint is too close to target, get a new one
                if self.target_pos is not None:
                    dist_to_target = np.linalg.norm(waypoint - self.target_pos)
                    if dist_to_target < 2.0:  # Arbitrary threshold
                        return self.get_next_waypoint(drone_id, drone_type, current_pos)
                return waypoint
                
        elif drone_type == 'worker':
            if self.phase == "exploration":
                # During exploration, workers stay in formation or holding pattern
                x_min, x_max, y_min, y_max, z_min, z_max = self.space_limits
                hold_pos = np.array([
                    (x_min + x_max) / 2 + np.cos(drone_id * 2 * np.pi / self.n_workers) * 2,
                    (y_min + y_max) / 2 + np.sin(drone_id * 2 * np.pi / self.n_workers) * 2,
                    z_max - 1.0  # Hold near the top
                ])
                return hold_pos
                
            elif self.phase == "assignment" and self.target_pos is not None:
                # Assign workers to approach target
                self.phase = "execution"
                # Generate paths for all workers
                for i in range(self.n_workers):
                    start_grid_pos = self.world_to_grid(current_pos)
                    path = self.shared_map.find_path(start_grid_pos, self.target_grid_pos)
                    if path:
                        self.worker_paths[i] = [self.grid_to_world(p) for p in path]
                        print(f"Worker {i} assigned path to target with {len(path)} waypoints")
                    else:
                        print(f"Worker {i} could not find path to target")
                
                # Fall through to execution phase
                self.phase = "execution"
                
            if self.phase == "execution":
                # Follow assigned path to target
                path = self.worker_paths[drone_id]
                if path and len(path) > 0:
                    # Get next waypoint in path
                    next_pos = path[0]
                    # Check if we've reached the current waypoint
                    dist = np.linalg.norm(current_pos - next_pos)
                    if dist < 0.5:  # Threshold for reaching waypoint
                        path.pop(0)  # Remove reached waypoint
                        if len(path) > 0:
                            return path[0]
                        else:
                            # Reached target, hover
                            return self.target_pos
                    return next_pos
                else:
                    # No path or empty path, just head toward target
                    return self.target_pos if self.target_pos is not None else current_pos
        
        # Default: stay in place
        return current_pos
    
    def get_exploration_stats(self, truth_map=None):
        """Return exploration statistics
        
        Args:
            truth_map: Optional ground truth map for accuracy calculation
            
        Returns:
            dict: Dictionary with exploration statistics
        """
        stats = {
            "coverage_percent": self.shared_map.get_explored_percentage(),
            "collisions": self.shared_map.get_collision_count(),
            "target_found_time": self.shared_map.get_target_found_time()
        }
        
        if truth_map is not None:
            stats["map_accuracy"] = self.shared_map.calculate_map_accuracy(truth_map)
            
        return stats
