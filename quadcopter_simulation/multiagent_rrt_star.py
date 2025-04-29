"""
Multi-Agent RRT* (Rapidly-exploring Random Tree Star) for Drone Path Planning
==========================================================================
This module implements a Multi-Agent RRT* algorithm optimized for:
- 3D continuous space path planning
- Coordination between surveyor and worker drones
- Dynamic obstacle avoidance
- Smooth trajectory generation
"""

import numpy as np
import random
from collections import defaultdict
import heapq
import time

class RRTNode:
    """Node in the RRT tree"""
    
    def __init__(self, position, parent=None):
        """
        Args:
            position: 3D position vector [x, y, z]
            parent: Parent node reference
        """
        self.position = np.array(position)
        self.parent = parent
        self.children = []
        self.cost = 0.0  # Cost from root to this node
    
    def __repr__(self):
        return f"RRTNode(pos={self.position}, cost={self.cost})"


class MultiAgentRRTStar:
    """Multi-Agent RRT* path planner for drone swarms"""
    
    def __init__(self, space_limits, shared_map, n_drones, drone_roles, sensing_radius=2.0, max_iterations=1000):
        """
        Args:
            space_limits: (x_min, x_max, y_min, y_max, z_min, z_max) environment bounds
            shared_map: SharedMap instance for obstacle information
            n_drones: Total number of drones
            drone_roles: List of drone roles ('surveyor' or 'worker')
            sensing_radius: Drone sensing radius
            max_iterations: Maximum iterations for RRT algorithm
        """
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = space_limits
        self.shared_map = shared_map
        self.n_drones = n_drones
        self.drone_roles = drone_roles
        self.sensing_radius = sensing_radius
        self.max_iterations = max_iterations
        
        # Parameters
        self.step_size = 0.5              # Maximum step size for extending tree
        self.goal_sample_rate = 0.15      # Probability of sampling goal
        self.search_radius = 2.0          # Radius for node rewiring
        self.drone_radius = 0.5           # Physical size of drone for collision checking
        self.time_horizon = 5.0           # Time horizon for collision prediction (seconds)
        self.obstacle_clearance = 1.0     # Minimum distance to obstacles
        
        # Trees for each drone
        self.trees = [[] for _ in range(n_drones)]
        
        # Paths for each drone
        self.paths = [None] * n_drones
        
        # Debug info
        self.iterations_used = 0
        self.planning_time = 0.0
        
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
    
    def check_collision(self, pos):
        """Check if position collides with an obstacle"""
        # Convert to grid coordinates
        grid_pos = self.world_to_grid(pos)
        
        # Check static obstacles in the shared map
        if 0 <= grid_pos[0] < self.shared_map.nx and \
           0 <= grid_pos[1] < self.shared_map.ny and \
           0 <= grid_pos[2] < self.shared_map.nz:
            cell_value = self.shared_map.grid[grid_pos]
            # Value 1 represents obstacles
            if cell_value == 1:
                return True
        
        # Check world boundaries
        if pos[0] < self.x_min + self.obstacle_clearance or pos[0] > self.x_max - self.obstacle_clearance or \
           pos[1] < self.y_min + self.obstacle_clearance or pos[1] > self.y_max - self.obstacle_clearance or \
           pos[2] < self.z_min + self.obstacle_clearance or pos[2] > self.z_max - self.obstacle_clearance:
            return True
            
        return False
    
    def check_path_collision(self, start_pos, end_pos, resolution=10):
        """Check if a path segment between two positions collides with obstacles"""
        # Create interpolated points along the path
        points = np.linspace(start_pos, end_pos, resolution)
        
        # Check each point for collision
        for point in points:
            if self.check_collision(point):
                return True
                
        return False
    
    def check_inter_agent_collision(self, pos1, pos2):
        """Check if two agent positions are too close"""
        dist = np.linalg.norm(pos1 - pos2)
        return dist < 2 * self.drone_radius  # Double the drone radius for safety
    
    def distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return np.linalg.norm(pos1 - pos2)
    
    def nearest_node(self, tree, position):
        """Find the nearest node in the tree to the given position"""
        min_dist = float('inf')
        nearest = None
        
        for node in tree:
            dist = self.distance(node.position, position)
            if dist < min_dist:
                min_dist = dist
                nearest = node
                
        return nearest
    
    def near_nodes(self, tree, position, radius):
        """Find all nodes in the tree within radius of the given position"""
        return [node for node in tree if self.distance(node.position, position) <= radius]
    
    def new_position(self, from_pos, to_pos, step_size):
        """Generate a new position by moving from from_pos toward to_pos by step_size"""
        direction = to_pos - from_pos
        # Normalize direction vector
        dist = np.linalg.norm(direction)
        if dist < step_size:
            return to_pos
        
        # Move step_size distance in the direction of to_pos
        direction = direction / dist
        return from_pos + direction * step_size
    
    def sample_free(self, goal_pos=None):
        """Sample a random collision-free position, with bias toward the goal and unexplored areas"""
        if goal_pos is not None and random.random() < self.goal_sample_rate:
            return goal_pos
        
        # Sample random position within bounds
        max_attempts = 20  # Limit attempts to avoid infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            # Generate random position
            pos = np.array([
                random.uniform(self.x_min + self.obstacle_clearance, self.x_max - self.obstacle_clearance),
                random.uniform(self.y_min + self.obstacle_clearance, self.y_max - self.obstacle_clearance),
                random.uniform(self.z_min + self.obstacle_clearance, self.z_max - self.obstacle_clearance)
            ])
            
            # Reject if in collision
            if self.check_collision(pos):
                attempts += 1
                continue
                
            # Check if in unexplored area - add a bias toward unexplored regions
            grid_pos = self.world_to_grid(pos)
            cell_value = self.shared_map.grid[grid_pos]
            
            # If cell is unexplored (255), higher chance of accepting
            if cell_value == 255 and random.random() < 0.7:  # 70% chance to accept unexplored
                return pos
                
            # If cell is free (0), lower chance of accepting
            if cell_value == 0 and random.random() < 0.3:  # 30% chance to accept already explored
                return pos
                
            attempts += 1
            
        # If we get here, just return any valid position
        while True:
            pos = np.array([
                random.uniform(self.x_min + self.obstacle_clearance, self.x_max - self.obstacle_clearance),
                random.uniform(self.y_min + self.obstacle_clearance, self.y_max - self.obstacle_clearance),
                random.uniform(self.z_min + self.obstacle_clearance, self.z_max - self.obstacle_clearance)
            ])
            
            # Only reject if in collision
            if not self.check_collision(pos):
                return pos
    
    def path_cost(self, node):
        """Calculate the total cost from root to this node"""
        cost = 0.0
        current = node
        while current.parent is not None:
            cost += self.distance(current.position, current.parent.position)
            current = current.parent
        return cost
    
    def build_path(self, node):
        """Build path from root to the given node"""
        path = []
        current = node
        while current is not None:
            path.append(current.position)
            current = current.parent
        return path[::-1]  # Reverse to get root-to-node order
    
    def plan_path(self, start_pos, goal_pos, drone_id, existing_paths=None):
        """
        Improved plan_path function with better handling of edge cases
        
        Args:
            start_pos: Starting position [x, y, z]
            goal_pos: Goal position [x, y, z]
            drone_id: ID of the drone
            existing_paths: Optional dictionary of already planned paths for other drones
            
        Returns:
            List of positions forming the path, or None if no path found
        """
        # Convert positions to numpy arrays
        start_pos = np.array(start_pos)
        goal_pos = np.array(goal_pos)
        
        # Add direct check for proximity
        direct_distance = self.distance(start_pos, goal_pos)
        if direct_distance < self.step_size * 3:
            # If already very close to goal, just return direct path
            return [start_pos, goal_pos]
        
        # Check if direct path is collision-free
        if not self.check_path_collision(start_pos, goal_pos):
            # If direct path is clear, return it
            return [start_pos, goal_pos]
        
        # Initialize tree with start node
        tree = []
        start_node = RRTNode(start_pos)
        tree.append(start_node)
        
        # Store tree for this drone
        self.trees[drone_id] = tree
        
        # Track best goal node found so far
        best_goal_node = None
        best_goal_dist = float('inf')
        
        # Record start time
        start_time = time.time()
        
        # For worker drones, increase the goal sampling rate to find the target faster
        original_goal_rate = self.goal_sample_rate
        if drone_id >= len([r for r in self.drone_roles if r == 'surveyor']):
            # This is a worker drone
            self.goal_sample_rate = 0.4  # Higher goal bias for workers
        
        # RRT* main loop
        for i in range(self.max_iterations):
            # Sample a random free position (with goal bias)
            rand_pos = self.sample_free(goal_pos)
            
            # Find nearest node in the tree
            nearest_node = self.nearest_node(tree, rand_pos)
            
            # Create new position by moving a step from nearest toward random
            new_pos = self.new_position(nearest_node.position, rand_pos, self.step_size)
            
            # Skip if new position is in collision with obstacles
            if self.check_collision(new_pos):
                continue
                
            # Skip if path to new position is in collision
            if self.check_path_collision(nearest_node.position, new_pos):
                continue
            
            # Check for collisions with other drones' planned paths
            if existing_paths:
                collision_with_other = False
                for other_id, other_path in existing_paths.items():
                    if other_path is None or len(other_path) < 2:
                        continue
                        
                    # Approximate check: just check if new position is too close to any point
                    # on other drone's path
                    for other_pos in other_path:
                        if self.check_inter_agent_collision(new_pos, other_pos):
                            collision_with_other = True
                            break
                    
                    if collision_with_other:
                        break
                        
                if collision_with_other:
                    continue
            
            # Create new node
            new_node = RRTNode(new_pos, nearest_node)
            
            # Calculate cost to new node
            new_node.cost = nearest_node.cost + self.distance(nearest_node.position, new_pos)
            
            # Add new node to parent's children
            nearest_node.children.append(new_node)
            
            # Add new node to tree
            tree.append(new_node)
            
            # Find nodes near the new node for potential rewiring
            near_nodes = self.near_nodes(tree, new_pos, self.search_radius)
            
            # Find the best parent for the new node
            for near_node in near_nodes:
                if near_node == new_node:
                    continue
                    
                # Calculate potential cost through near_node
                potential_cost = near_node.cost + self.distance(near_node.position, new_pos)
                
                # Skip if not a better cost
                if potential_cost >= new_node.cost:
                    continue
                    
                # Skip if path would collide
                if self.check_path_collision(near_node.position, new_pos):
                    continue
                
                # Rewire: Change parent of new_node
                # Remove from old parent's children
                if new_node.parent:
                    new_node.parent.children.remove(new_node)
                
                # Set new parent
                new_node.parent = near_node
                new_node.cost = potential_cost
                
                # Add to new parent's children
                near_node.children.append(new_node)
            
            # Check if we're close to the goal
            dist_to_goal = self.distance(new_pos, goal_pos)
            if dist_to_goal < self.step_size:
                # Check direct path to goal
                if not self.check_path_collision(new_pos, goal_pos):
                    # Create goal node
                    goal_node = RRTNode(goal_pos, new_node)
                    goal_node.cost = new_node.cost + dist_to_goal
                    
                    # Check if this is the best path to goal so far
                    if best_goal_node is None or goal_node.cost < best_goal_node.cost:
                        best_goal_node = goal_node
            
            # Update best goal node if we're getting closer
            if dist_to_goal < best_goal_dist:
                best_goal_dist = dist_to_goal
                
            # Optional early termination if we found a good enough path
            if best_goal_node is not None and i > self.max_iterations // 4:
                # If we've done at least 1/4 the iterations and found a path
                break
            
            # Also terminate early if taking too long (1 second)
            if time.time() - start_time > 1.0:
                break
        
        # Restore original goal sampling rate
        self.goal_sample_rate = original_goal_rate
        
        # Record planning statistics
        self.iterations_used = i + 1
        self.planning_time = time.time() - start_time
        
        # Build path if goal node found
        if best_goal_node is not None:
            path = self.build_path(best_goal_node)
            return path
        
        # If no path to goal found, see if we can at least get close
        if best_goal_dist < self.step_size * 10:  # Larger threshold
            # Find the node that got closest to the goal
            closest_node = min(tree, key=lambda node: self.distance(node.position, goal_pos))
            path = self.build_path(closest_node)
            # Append goal to get approximate path if can connect directly
            if not self.check_path_collision(closest_node.position, goal_pos):
                path.append(goal_pos)
            return path
        
        # If RRT* failed, try a simple direct path with intermediate waypoints
        print(f"RRT* failed, trying direct path with waypoints for drone {drone_id}")
        direct_path = [start_pos]
        
        # Add some intermediate points to avoid obstacles
        direction = goal_pos - start_pos
        distance = np.linalg.norm(direction)
        direction = direction / distance
        
        # Try different step sizes
        for step in range(1, 6):
            intermediate_pos = start_pos + direction * (distance * step / 5)
            if not self.check_collision(intermediate_pos):
                direct_path.append(intermediate_pos)
        
        direct_path.append(goal_pos)
        return direct_path
    
    def plan_paths(self, start_positions, goal_positions, priority_ordering=None):
        """
        Plan paths for all drones with priority ordering
        
        Args:
            start_positions: List of starting positions for each drone
            goal_positions: List of goal positions for each drone
            priority_ordering: Optional list of drone IDs in priority order
            
        Returns:
            Dictionary mapping drone IDs to paths
        """
        # Default ordering (workers first, then surveyors)
        if priority_ordering is None:
            # Find worker drones and give them higher priority
            workers = [i for i, role in enumerate(self.drone_roles) if role == 'worker']
            surveyors = [i for i, role in enumerate(self.drone_roles) if role == 'surveyor']
            priority_ordering = workers + surveyors
        
        # Initialize paths
        paths = {}
        
        # Plan paths in priority order
        for drone_id in priority_ordering:
            # Check if we have valid start and goal positions
            if start_positions[drone_id] is None or goal_positions[drone_id] is None:
                print(f"  Skipping path planning for drone {drone_id} due to missing start or goal position")
                paths[drone_id] = None
                self.paths[drone_id] = None
                continue
                
            start_pos = start_positions[drone_id]
            goal_pos = goal_positions[drone_id]
            
            print(f"Planning path for drone {drone_id} ({self.drone_roles[drone_id]})")
            print(f"  From: {start_pos}")
            print(f"  To: {goal_pos}")
            
            # Plan path for this drone, avoiding already planned paths
            path = self.plan_path(start_pos, goal_pos, drone_id, paths)
            
            # Store the path
            paths[drone_id] = path
            self.paths[drone_id] = path
            
            if path:
                print(f"  Path found with {len(path)} waypoints")
            else:
                print(f"  No path found!")
        
        return paths
    
    def post_process_paths(self, paths, smoothing_iterations=50):
        """
        Post-process paths for smoothness and uniform segment lengths
        
        Args:
            paths: Dictionary mapping drone IDs to paths
            smoothing_iterations: Number of smoothing iterations
            
        Returns:
            Dictionary of processed paths
        """
        processed_paths = {}
        
        for drone_id, path in paths.items():
            if path is None or len(path) < 3:
                processed_paths[drone_id] = path
                continue
                
            # Make a copy of the path
            smooth_path = path.copy()
            
            # Iterative path smoothing
            for _ in range(smoothing_iterations):
                # Skip endpoints
                for i in range(1, len(smooth_path) - 1):
                    # Get neighboring points
                    prev = smooth_path[i - 1]
                    curr = smooth_path[i]
                    next_pt = smooth_path[i + 1]
                    
                    # Compute smoother position (simple averaging)
                    smooth_pos = prev * 0.25 + curr * 0.5 + next_pt * 0.25
                    
                    # Check if the path with the smoother position would collide
                    if not self.check_collision(smooth_pos) and \
                       not self.check_path_collision(prev, smooth_pos) and \
                       not self.check_path_collision(smooth_pos, next_pt):
                        smooth_path[i] = smooth_pos
            
            # Resample path for uniform segment lengths
            uniform_path = [smooth_path[0]]  # Start with the first point
            total_distance = 0.0
            
            # Calculate total path length
            for i in range(1, len(smooth_path)):
                total_distance += self.distance(smooth_path[i-1], smooth_path[i])
            
            # Determine desired number of segments based on path length
            # Longer paths get more segments, with a minimum of 5
            target_segments = max(5, int(total_distance / self.step_size))
            
            # Resample path to have uniform segments
            cumulative_distance = 0.0
            segment_length = total_distance / target_segments
            next_sample_distance = segment_length
            
            for i in range(1, len(smooth_path)):
                segment_start = smooth_path[i-1]
                segment_end = smooth_path[i]
                segment_dist = self.distance(segment_start, segment_end)
                
                # If we reach the desired distance, sample a point
                while cumulative_distance + segment_dist >= next_sample_distance:
                    # Interpolate to get exact position
                    alpha = (next_sample_distance - cumulative_distance) / segment_dist
                    sample_pos = segment_start * (1 - alpha) + segment_end * alpha
                    
                    uniform_path.append(sample_pos)
                    next_sample_distance += segment_length
                
                cumulative_distance += segment_dist
            
            # Always include the goal position
            if not np.array_equal(uniform_path[-1], smooth_path[-1]):
                uniform_path.append(smooth_path[-1])
            
            processed_paths[drone_id] = uniform_path
            
        return processed_paths
    
    def get_next_waypoint(self, drone_id, current_pos, lookahead=1):
        """
        Get the next waypoint for a drone to follow
        
        Args:
            drone_id: ID of the drone
            current_pos: Current position of the drone
            lookahead: How many waypoints to look ahead (for smoother paths)
            
        Returns:
            Next position to move toward
        """
        path = self.paths[drone_id]
        if path is None or len(path) < 2:
            return current_pos
        
        # Find closest point on path
        min_dist = float('inf')
        closest_idx = 0
        
        for i, pos in enumerate(path):
            dist = self.distance(current_pos, pos)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Get waypoint with lookahead
        target_idx = min(closest_idx + lookahead, len(path) - 1)
        return path[target_idx]


class MultiAgentPathPlanner:
    """Interface to integrate RRT* with the existing multi-agent planner"""
    
    def __init__(self, shared_map, space_limits, n_surveyors, n_workers):
        """
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
        
        # Create drone roles list
        self.drone_roles = ['surveyor'] * n_surveyors + ['worker'] * n_workers
        
        # Initialize RRT* planner
        self.rrt_planner = MultiAgentRRTStar(
            space_limits,
            shared_map,
            self.n_drones,
            self.drone_roles
        )
        
        # Store target location (once found)
        self.target_pos = None
        self.target_grid_pos = None
        
        # Store drone paths
        self.paths = [None] * self.n_drones
        
        # State tracking
        self.phase = "exploration"  # exploration, planning, execution
        
    def world_to_grid(self, pos):
        """Convert world coordinates to grid indices"""
        return self.rrt_planner.world_to_grid(pos)
    
    def grid_to_world(self, grid_pos):
        """Convert grid indices to world coordinates"""
        return self.rrt_planner.grid_to_world(grid_pos)
    
    def update_map(self, drone_id, pos, grid_pos, observed_cells):
        """Update shared map with observations"""
        self.shared_map.update(drone_id, pos, grid_pos, observed_cells)
        
        # Check if target has been found
        for (ix, iy, iz), status in observed_cells.items():
            if status == 2 and self.target_pos is None:  # Target found
                self.target_grid_pos = (ix, iy, iz)
                self.target_pos = self.grid_to_world((ix, iy, iz))
                self.phase = "planning"
                print(f"Target found at {self.target_pos}. Switching to planning phase.")
    
    def get_next_waypoint(self, drone_id, drone_type, current_pos):
        """Get next waypoint for a drone"""
        real_drone_id = drone_id if drone_type == 'surveyor' else drone_id + self.n_surveyors
        
        if drone_type == 'surveyor':
            if self.phase == "exploration":
                # During exploration, use random sampling to explore space
                # This is placeholder logic - in practice you'd use existing exploration behaviors
                x_min, x_max, y_min, y_max, z_min, z_max = self.space_limits
                rand_pos = np.array([
                    random.uniform(x_min, x_max),
                    random.uniform(y_min, y_max),
                    random.uniform(z_min, z_max)
                ])
                return rand_pos
            else:
                # After target found, surveyors continue mapping but avoid target area
                if self.paths[real_drone_id] is not None:
                    return self.rrt_planner.get_next_waypoint(real_drone_id, current_pos)
                else:
                    # Just keep exploring if no path assigned
                    x_min, x_max, y_min, y_max, z_min, z_max = self.space_limits
                    # Bias exploration away from target
                    if self.target_pos is not None:
                        # Sample in opposite direction of target
                        center = np.array([
                            (x_min + x_max) / 2,
                            (y_min + y_max) / 2,
                            (z_min + z_max) / 2
                        ])
                        direction = center - self.target_pos
                        direction = direction / np.linalg.norm(direction)
                        rand_pos = center + direction * random.uniform(2.0, 5.0)
                        rand_pos = np.clip(rand_pos, 
                                          [x_min, y_min, z_min],
                                          [x_max, y_max, z_max])
                        return rand_pos
                    else:
                        rand_pos = np.array([
                            random.uniform(x_min, x_max),
                            random.uniform(y_min, y_max),
                            random.uniform(z_min, z_max)
                        ])
                        return rand_pos
        
        elif drone_type == 'worker':
            if self.phase == "exploration":
                # During exploration, workers stay in holding pattern
                x_min, x_max, y_min, y_max, z_min, z_max = self.space_limits
                hold_pos = np.array([
                    (x_min + x_max) / 2 + np.cos(drone_id * 2 * np.pi / self.n_workers) * 2,
                    (y_min + y_max) / 2 + np.sin(drone_id * 2 * np.pi / self.n_workers) * 2,
                    z_max - 1.0  # Hold near the top
                ])
                return hold_pos
            
            elif self.phase == "planning" and self.target_pos is not None:
                print(f"\n=== PLANNING PATHS TO TARGET WITH RRT* ===")
                # Get current positions of all drones
                start_positions = [None] * self.n_drones
                # Placeholder - in real code, you would get actual current positions
                
                # Set goal positions (target for workers, exploration points for surveyors)
                goal_positions = [None] * self.n_drones
                
                # Plan paths for all drones simultaneously
                self.paths = self.rrt_planner.plan_paths(start_positions, goal_positions)
                
                # Post-process paths for smoothness
                self.paths = self.rrt_planner.post_process_paths(self.paths)
                
                # Switch to execution phase
                self.phase = "execution"
                
                # Fall through to execution
                
            if self.phase == "execution":
                # Follow assigned path to target
                return self.rrt_planner.get_next_waypoint(real_drone_id, current_pos)
        
        # Default: stay in place
        return current_pos
    
    def record_collision(self):
        """Record a collision event"""
        self.shared_map.record_collision()
    
    def get_exploration_stats(self, truth_map=None):
        """Return exploration statistics"""
        stats = {
            "coverage_percent": self.shared_map.get_explored_percentage(),
            "collisions": self.shared_map.get_collision_count(),
            "target_found_time": self.shared_map.get_target_found_time()
        }
        
        if truth_map is not None:
            stats["map_accuracy"] = self.shared_map.calculate_map_accuracy(truth_map)
            
        return stats
