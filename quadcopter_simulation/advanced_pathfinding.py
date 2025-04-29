"""
Advanced Path Planning for Multi-Agent Drone Systems
===================================================
This module implements improved path planning algorithms for the drone simulation:
- RRT (Rapidly-exploring Random Trees)
- RRT* (Optimized RRT)
- PRM (Probabilistic Roadmap Method)

These algorithms offer significant advantages over basic A* for drone navigation.
"""

import numpy as np
import random
import heapq
from collections import defaultdict

class Node:
    """Node class for RRT and RRT* algorithms"""
    def __init__(self, position):
        self.position = np.array(position)
        self.parent = None
        self.cost = 0.0  # Cost from start (for RRT*)
        self.children = []  # For rewiring in RRT*

    def __eq__(self, other):
        if isinstance(other, Node):
            return np.array_equal(self.position, other.position)
        return False
        
    def __hash__(self):
        return hash(tuple(self.position))

class PathPlanner:
    """Advanced path planning algorithms for drone navigation"""
    
    def __init__(self, shared_map, space_limits, grid_dims):
        """Initialize path planner
        
        Args:
            shared_map: SharedMap instance containing occupancy grid
            space_limits: (x_min, x_max, y_min, y_max, z_min, z_max)
            grid_dims: (nx, ny, nz) grid dimensions
        """
        self.shared_map = shared_map
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = space_limits
        self.nx, self.ny, self.nz = grid_dims
        
        # Parameters
        self.step_size = 0.5  # Step size for extending RRT (world coordinates)
        self.max_iterations = 1000  # Maximum iterations for RRT
        self.goal_sample_rate = 0.1  # Probability of sampling goal directly
        self.search_radius = 2.0  # Radius for PRM and RRT* neighbor search
        
        # Default algorithm
        self.default_algorithm = "rrt"  # Options: "a_star", "rrt", "rrt_star", "prm"
    
    def find_path(self, start, goal, algorithm=None):
        """Find path from start to goal using specified algorithm
        
        Args:
            start: (x,y,z) start position in grid coordinates
            goal: (x,y,z) goal position in grid coordinates
            algorithm: Which algorithm to use (defaults to self.default_algorithm)
            
        Returns:
            list: Path from start to goal as list of grid coordinates
        """
        if algorithm is None:
            algorithm = self.default_algorithm
            
        # Convert grid coordinates to world coordinates
        start_world = self.grid_to_world(start)
        goal_world = self.grid_to_world(goal)
        
        # Choose algorithm
        if algorithm == "a_star":
            return self.a_star(start, goal)
        elif algorithm == "rrt":
            return self.rrt(start_world, goal_world)
        elif algorithm == "rrt_star":
            return self.rrt_star(start_world, goal_world)
        elif algorithm == "prm":
            return self.prm(start_world, goal_world)
        else:
            print(f"Unknown algorithm: {algorithm}, using RRT instead")
            return self.rrt(start_world, goal_world)
    
    def grid_to_world(self, grid_pos):
        """Convert grid coordinates to world coordinates"""
        ix, iy, iz = grid_pos
        x = self.x_min + (ix / (self.nx - 1)) * (self.x_max - self.x_min)
        y = self.y_min + (iy / (self.ny - 1)) * (self.y_max - self.y_min)
        z = self.z_min + (iz / (self.nz - 1)) * (self.z_max - self.z_min)
        return np.array([x, y, z])
    
    def world_to_grid(self, world_pos):
        """Convert world coordinates to grid indices"""
        x, y, z = world_pos
        ix = int((x - self.x_min) / (self.x_max - self.x_min) * (self.nx - 1))
        iy = int((y - self.y_min) / (self.y_max - self.y_min) * (self.ny - 1))
        iz = int((z - self.z_min) / (self.z_max - self.z_min) * (self.nz - 1))
        return (np.clip(ix, 0, self.nx-1), np.clip(iy, 0, self.ny-1), np.clip(iz, 0, self.nz-1))
    
    def is_collision_free(self, pos):
        """Check if a world position is collision-free
        
        Args:
            pos: (x,y,z) position in world coordinates
            
        Returns:
            bool: True if position is valid (not an obstacle or unknown)
        """
        # Check world bounds
        if (pos[0] < self.x_min or pos[0] > self.x_max or
            pos[1] < self.y_min or pos[1] > self.y_max or
            pos[2] < self.z_min or pos[2] > self.z_max):
            return False
            
        # Convert to grid coordinates
        grid_pos = self.world_to_grid(pos)
        
        # Check grid value
        grid_value = self.shared_map.grid[grid_pos]
        
        # 0 is free, 2 is target (both valid), 1 is obstacle, 255 is unknown
        return grid_value == 0 or grid_value == 2
    
    def is_path_collision_free(self, from_pos, to_pos, check_points=5):
        """Check if a straight path between two positions is collision-free
        
        Args:
            from_pos: (x,y,z) start position in world coordinates
            to_pos: (x,y,z) end position in world coordinates
            check_points: Number of intermediate points to check
            
        Returns:
            bool: True if path is collision-free
        """
        from_pos = np.array(from_pos)
        to_pos = np.array(to_pos)
        
        # Check endpoints
        if not self.is_collision_free(from_pos) or not self.is_collision_free(to_pos):
            return False
            
        # Check intermediate points
        for i in range(1, check_points):
            t = i / check_points
            interp_pos = from_pos * (1 - t) + to_pos * t
            if not self.is_collision_free(interp_pos):
                return False
                
        return True
    
    def distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return np.linalg.norm(np.array(pos2) - np.array(pos1))
    
    def random_position(self):
        """Generate a random position in the world space"""
        x = random.uniform(self.x_min, self.x_max)
        y = random.uniform(self.y_min, self.y_max)
        z = random.uniform(self.z_min, self.z_max)
        return np.array([x, y, z])
    
    def nearest_node(self, nodes, position):
        """Find the nearest node to the given position"""
        min_dist = float('inf')
        closest_node = None
        
        for node in nodes:
            dist = self.distance(node.position, position)
            if dist < min_dist:
                min_dist = dist
                closest_node = node
                
        return closest_node
    
    def steer(self, from_pos, to_pos, max_distance=None):
        """Steer from a position toward another, respecting max_distance"""
        if max_distance is None:
            max_distance = self.step_size
            
        from_pos = np.array(from_pos)
        to_pos = np.array(to_pos)
        
        dist = self.distance(from_pos, to_pos)
        
        if dist <= max_distance:
            return to_pos.copy()
        else:
            # Calculate new position by moving max_distance toward to_pos
            direction = (to_pos - from_pos) / dist
            return from_pos + direction * max_distance
    
    def near_nodes(self, nodes, position, radius):
        """Find all nodes within a radius of the position"""
        return [node for node in nodes if self.distance(node.position, position) <= radius]
    
    #######################
    # A* IMPLEMENTATION
    #######################
    
    def a_star(self, start, goal):
        """A* algorithm (based on existing implementation but with improvements)
        
        Args:
            start: (x,y,z) start position in grid coordinates
            goal: (x,y,z) goal position in grid coordinates
            
        Returns:
            list: Path from start to goal as list of grid coordinates
        """
        start = tuple(start)
        goal = tuple(goal)
        
        # If start or goal are in unknown or obstacle space, no path
        if self.shared_map.grid[start] in (255, 1):
            return None
        if self.shared_map.grid[goal] in (255, 1):
            return None
        
        # A* algorithm
        open_set = []
        closed_set = set()
        g_score = {start: 0}  # Cost from start to current
        f_score = {start: self.heuristic(start, goal)}  # Estimated total cost
        came_from = {}
        
        heapq.heappush(open_set, (f_score[start], start))
        
        # Add timeout to prevent infinite loops
        max_iterations = self.nx * self.ny * self.nz
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
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
            
            # Explore neighbors - including diagonal moves
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                    
                # Check if neighbor is valid
                nx, ny, nz = neighbor
                if not (0 <= nx < self.nx and 0 <= ny < self.ny and 0 <= nz < self.nz):
                    continue
                
                # Skip obstacles and unknown cells
                if self.shared_map.grid[nx, ny, nz] in (1, 255):
                    continue
                
                # Calculate cost - higher cost for diagonal moves
                diagonal_move = sum(abs(a - b) for a, b in zip(current, neighbor)) > 1
                move_cost = 1.4 if diagonal_move else 1.0
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        return None
    
    def heuristic(self, a, b):
        """Heuristic for A* - using Euclidean distance for 3D space"""
        return np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(a, b)))
    
    def get_neighbors(self, pos):
        """Get neighboring grid cells, including diagonal moves"""
        x, y, z = pos
        neighbors = []
        
        # 6-connected neighbors (cardinal directions)
        cardinal = [
            (x+1, y, z), (x-1, y, z),
            (x, y+1, z), (x, y-1, z),
            (x, y, z+1), (x, y, z-1)
        ]
        
        # Add cardinal neighbors
        for nx, ny, nz in cardinal:
            if 0 <= nx < self.nx and 0 <= ny < self.ny and 0 <= nz < self.nz:
                neighbors.append((nx, ny, nz))
        
        return neighbors
    
    #######################
    # RRT IMPLEMENTATION
    #######################
    
    def rrt(self, start, goal, max_iterations=None):
        """RRT algorithm for path planning
        
        Args:
            start: (x,y,z) start position in world coordinates
            goal: (x,y,z) goal position in world coordinates
            max_iterations: Maximum number of iterations (default: self.max_iterations)
            
        Returns:
            list: Path from start to goal as list of grid coordinates
        """
        if max_iterations is None:
            max_iterations = min(self.max_iterations, 500)  # Limit to prevent freezing
            
        # Safety check - if goal or start are in collision, return direct path
        if not self.is_collision_free(goal) or not self.is_collision_free(start):
            print(f"Warning: Start or goal position is in collision. Returning direct path.")
            return [self.world_to_grid(start), self.world_to_grid(goal)]
            
        # Check if direct path is possible (faster check first)
        if self.is_path_collision_free(start, goal):
            return [self.world_to_grid(start), self.world_to_grid(goal)]
            
        # Initialize tree with start node
        start_node = Node(start)
        nodes = [start_node]
        
        # Goal region parameters
        goal_threshold = self.step_size * 0.1  # Increased threshold for faster convergence
        
        # For early termination if no progress is made
        best_dist_to_goal = self.distance(start, goal)
        stall_iterations = 0
        max_stall = 50  # Reduced to prevent freezing
        
        # Main RRT loop with timeout protection
        for i in range(max_iterations):
            # Bias towards goal with higher probability
            if random.random() < self.goal_sample_rate * 1.5:  # Increased goal bias
                random_pos = goal
            else:
                random_pos = self.random_position()
                
            # Find nearest node in tree
            nearest = self.nearest_node(nodes, random_pos)
            
            # Steer toward random position with step size
            new_pos = self.steer(nearest.position, random_pos)
            
            # Check if new position is collision-free (with fewer checks)
            if self.is_path_collision_free(nearest.position, new_pos, check_points=3):
                # Create new node
                new_node = Node(new_pos)
                new_node.parent = nearest
                nodes.append(new_node)
                nearest.children.append(new_node)
                
                # Check if we've reached the goal
                dist_to_goal = self.distance(new_pos, goal)
                
                # Track progress toward goal
                if dist_to_goal < best_dist_to_goal:
                    best_dist_to_goal = dist_to_goal
                    stall_iterations = 0
                else:
                    stall_iterations += 1
                
                # If we're close enough to goal and can connect to it directly
                if dist_to_goal < goal_threshold or self.is_path_collision_free(new_pos, goal, check_points=3):
                    # Create goal node and connect
                    goal_node = Node(goal)
                    goal_node.parent = new_node
                    nodes.append(goal_node)
                    new_node.children.append(goal_node)
                    
                    # Extract path
                    return self.extract_path(goal_node)
            
            # Early termination if no progress
            if stall_iterations > max_stall:
                print(f"RRT stalled after {i} iterations, returning best path.")
                # Try to connect the closest node to goal
                closest_node = self.nearest_node(nodes, goal)
                if self.is_path_collision_free(closest_node.position, goal, check_points=3):
                    goal_node = Node(goal)
                    goal_node.parent = closest_node
                    nodes.append(goal_node)
                    closest_node.children.append(goal_node)
                    return self.extract_path(goal_node)
                
                # If we can't connect to goal, return path to closest node
                return self.extract_path(closest_node)
        
        print(f"RRT reached max iterations ({max_iterations}), returning best path.")
        # If max iterations reached without finding path
        # Return best partial path
        closest_node = self.nearest_node(nodes, goal)
        return self.extract_path(closest_node)
    
    def extract_path(self, end_node):
        """Extract path from end node back to start node"""
        path = []
        current = end_node
        
        # Traverse parent links to get path
        while current is not None:
            # Add to path and convert to grid coordinates
            path.append(self.world_to_grid(current.position))
            current = current.parent
            
        path.reverse()  # Reverse to get start-to-goal path
        return path
    
    #######################
    # RRT* IMPLEMENTATION
    #######################
    
    def rrt_star(self, start, goal, max_iterations=None):
        """RRT* algorithm - an optimized version of RRT
        
        Args:
            start: (x,y,z) start position in world coordinates
            goal: (x,y,z) goal position in world coordinates
            max_iterations: Maximum number of iterations
        
        Returns:
            list: Path from start to goal as list of grid coordinates
        """
        if max_iterations is None:
            max_iterations = self.max_iterations
            
        # Initialize tree with start node
        start_node = Node(start)
        nodes = [start_node]
        
        # Goal node and best path tracking
        goal_node = None
        best_goal_cost = float('inf')
        
        # Goal region parameters
        goal_threshold = self.step_size * 0.5
        
        # Main RRT* loop
        for i in range(max_iterations):
            # Sample goal with higher probability as iterations increase
            adaptive_goal_rate = self.goal_sample_rate * (1 + i / max_iterations)
            if random.random() < adaptive_goal_rate:
                random_pos = goal
            else:
                random_pos = self.random_position()
                
            # Find nearest node
            nearest = self.nearest_node(nodes, random_pos)
            
            # Steer toward random position
            new_pos = self.steer(nearest.position, random_pos)
            
            # Skip if collision or already very close to existing node
            if not self.is_path_collision_free(nearest.position, new_pos):
                continue
                
            # Create new node
            new_node = Node(new_pos)
            
            # Find nearest neighbors for rewiring
            # Fixed radius for simplicity (can be dynamic based on tree size)
            neighbors = self.near_nodes(nodes, new_pos, self.search_radius)
            
            # Connect to the neighbor that provides lowest cost path
            min_cost = nearest.cost + self.distance(nearest.position, new_pos)
            min_node = nearest
            
            # Check if any neighbor provides a better path
            for neighbor in neighbors:
                cost = neighbor.cost + self.distance(neighbor.position, new_pos)
                if cost < min_cost and self.is_path_collision_free(neighbor.position, new_pos):
                    min_cost = cost
                    min_node = neighbor
            
            # Set parent to lowest-cost neighbor
            new_node.parent = min_node
            new_node.cost = min_cost
            new_node.parent.children.append(new_node)
            nodes.append(new_node)
            
            # Rewire the tree - update children's costs if new node provides better path
            for neighbor in neighbors:
                # Skip the parent node
                if neighbor == min_node:
                    continue
                    
                potential_cost = new_node.cost + self.distance(new_node.position, neighbor.position)
                
                if potential_cost < neighbor.cost and self.is_path_collision_free(new_node.position, neighbor.position):
                    # Update parent
                    old_parent = neighbor.parent
                    if old_parent:
                        old_parent.children.remove(neighbor)
                    
                    neighbor.parent = new_node
                    new_node.children.append(neighbor)
                    
                    # Update cost for this node and all descendants
                    cost_diff = neighbor.cost - potential_cost
                    self.update_descendant_costs(neighbor, cost_diff)
            
            # Check if we can connect to goal
            dist_to_goal = self.distance(new_pos, goal)
            if dist_to_goal < goal_threshold and self.is_path_collision_free(new_pos, goal):
                # Create goal node
                potential_goal_node = Node(goal)
                potential_goal_node.parent = new_node
                potential_goal_node.cost = new_node.cost + dist_to_goal
                
                # Update best path to goal if this is better
                if goal_node is None or potential_goal_node.cost < best_goal_cost:
                    if goal_node:
                        # Remove from previous parent's children
                        goal_node.parent.children.remove(goal_node)
                    
                    goal_node = potential_goal_node
                    best_goal_cost = goal_node.cost
                    new_node.children.append(goal_node)
        
        # Return best path if found
        if goal_node:
            return self.extract_path(goal_node)
        
        # If no path to goal, return path to closest node
        closest_node = self.nearest_node(nodes, goal)
        return self.extract_path(closest_node)
    
    def update_descendant_costs(self, node, cost_diff):
        """Update costs for all descendants after rewiring"""
        node.cost -= cost_diff
        
        for child in node.children:
            self.update_descendant_costs(child, cost_diff)
    
    #######################
    # PRM IMPLEMENTATION
    #######################
    
    def prm(self, start, goal, num_samples=100):
        """Probabilistic Roadmap (PRM) algorithm
        
        Args:
            start: (x,y,z) start position in world coordinates
            goal: (x,y,z) goal position in world coordinates
            num_samples: Number of random samples for the roadmap
            
        Returns:
            list: Path from start to goal as list of grid coordinates
        """
        # Check if direct path is possible
        if self.is_path_collision_free(start, goal):
            return [self.world_to_grid(start), self.world_to_grid(goal)]
        
        # Initialize roadmap
        graph = defaultdict(list)  # node -> [(neighbor, distance), ...]
        nodes = [start, goal]  # Include start and goal in roadmap
        
        # Sample random points for the roadmap
        for _ in range(num_samples):
            sample = self.random_position()
            
            # Skip if in collision
            if not self.is_collision_free(sample):
                continue
                
            nodes.append(sample)
        
        # Connect each node to nearest neighbors
        k_neighbors = min(10, len(nodes) - 1)  # Maximum number of connections per node
        
        for i, node in enumerate(nodes):
            # Find k nearest neighbors
            distances = [(j, self.distance(node, other_node)) 
                        for j, other_node in enumerate(nodes) if i != j]
            distances.sort(key=lambda x: x[1])
            
            # Connect to nearest neighbors if collision-free
            for j, dist in distances[:k_neighbors]:
                if self.is_path_collision_free(node, nodes[j]):
                    graph[i].append((j, dist))
                    graph[j].append((i, dist))  # Undirected graph
        
        # Run Dijkstra's algorithm to find shortest path in roadmap
        path_indices = self.dijkstra(graph, 0, 1)  # 0=start, 1=goal
        
        if path_indices:
            # Convert indices to world positions then to grid coordinates
            return [self.world_to_grid(nodes[idx]) for idx in path_indices]
        else:
            # If no path found through PRM, try falling back to RRT
            return self.rrt(start, goal)
    
    def dijkstra(self, graph, start_idx, goal_idx):
        """Dijkstra's algorithm for shortest path in graph
        
        Args:
            graph: Adjacency list representation of roadmap
            start_idx: Index of start node
            goal_idx: Index of goal node
            
        Returns:
            list: Path from start to goal as list of node indices
        """
        # Priority queue for Dijkstra's algorithm
        queue = [(0, start_idx)]
        visited = set()
        costs = {start_idx: 0}
        came_from = {}
        
        while queue:
            cost, current = heapq.heappop(queue)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == goal_idx:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_idx)
                path.reverse()
                return path
            
            # Check neighbors
            for neighbor, edge_cost in graph[current]:
                if neighbor in visited:
                    continue
                    
                new_cost = cost + edge_cost
                
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    came_from[neighbor] = current
                    heapq.heappush(queue, (new_cost, neighbor))
        
        # No path found
        return None


# Helper function to simplify a path by removing redundant waypoints
def simplify_path(path, planner, max_distance=1.0):
    """Simplify a path by removing unnecessary waypoints
    
    Args:
        path: List of waypoints in grid coordinates
        planner: PathPlanner instance for collision checking
        max_distance: Maximum distance for path segments
        
    Returns:
        list: Simplified path
    """
    # Safety check
    if not path or len(path) < 3:
        return path
    
    # For extremely long paths, use a faster simplification method
    if len(path) > 20:
        return simplify_path_fast(path, planner)
        
    # Convert to world coordinates for better simplification
    world_path = [planner.grid_to_world(point) for point in path]
    
    simplified = [world_path[0]]  # Start with first point
    current_idx = 0
    
    # Limit number of iterations for safety
    max_iterations = min(len(world_path) * 2, 100)
    iteration = 0
    
    while current_idx < len(world_path) - 1 and iteration < max_iterations:
        iteration += 1
        next_idx = current_idx + 1
        
        # Look for furthest safe point to jump to
        for i in range(min(current_idx + 8, len(world_path) - 1), current_idx, -1):
            # Only check collision if we're skipping points
            if i > next_idx and planner.is_path_collision_free(world_path[current_idx], world_path[i], check_points=3):
                next_idx = i
                break
        
        # Add the next point and continue
        if next_idx < len(world_path):
            simplified.append(world_path[next_idx])
            current_idx = next_idx
        else:
            break
    
    # Make sure the last point is included
    if simplified[-1] is not world_path[-1]:
        simplified.append(world_path[-1])
    
    # Convert back to grid coordinates
    return [planner.world_to_grid(point) for point in simplified]

def simplify_path_fast(path, planner):
    """A faster path simplification that just keeps every N-th point
    
    Args:
        path: List of waypoints in grid coordinates
        planner: PathPlanner instance
        
    Returns:
        list: Simplified path
    """
    if len(path) <= 3:
        return path
    
    # Always keep start and end points
    simplified = [path[0]]
    
    # Determine sampling rate based on path length
    n = max(len(path) // 8, 2)  # Keep at most 8 intermediate points
    
    # Add intermediate points
    for i in range(n, len(path) - n, n):
        simplified.append(path[i])
    
    # Add end point
    simplified.append(path[-1])
    
    return simplified