import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import heapq
import time
from IPython.display import display, clear_output
import random
import math

'''
Methods to redo from scratch using methods from class, as they are the main focus of our problem:
Drone.sense
Drone.find_frontiers_and_paths
Drone.choose_target_frontier
Drone.plan_path_to_target

Methods to create:
Drone.deliver
'''

# --- Configuration ---
GRID_WIDTH = 100 # Make smaller grid for faster coverage testing
GRID_HEIGHT = 100
NUM_DRONES = 3
NUM_GOALS = 8 # Define number of goals
OBSTACLE_DENSITY = 0.2
DRONE_SENSOR_RANGE = 5
DRONE_START_POS = (1, 1)
DELIVERY_START_POS = DRONE_START_POS # Same as drones eventually
MAX_SIMULATION_STEPS = 1000 # Increase max steps for coverage goal
SPREADING_WEIGHT = 0.6
MIN_OTHER_DRONE_DIST_THRESHOLD = 3
COVERAGE_THRESHOLD = 1.0 # Target coverage (1.0 for 100%) (0.95 is probably good enough, 
# but the one guy in the 5% will be pretty mad)

# --- Cell States ---
# Used for coloring in the grid later on
UNKNOWN = -1
FREE = 0
OBSTACLE = 1
GOAL = 2 # Represents GOAL location on visualization map
DRONE = 3
# PATH = 4 - Implement in a second
# DELIVERY_ROBOT = 5 - Implement in a second

# --- Visualization ---
# Add a color for found goals maybe?
# Using Red for Goals, Blue for Drones
cmap = mcolors.ListedColormap(['grey', 'white', 'black', 'red', 'blue'])
bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# --- Helper Functions ---
def euclidean_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# --- Environment Class (Modified for multiple goals) ---
class Environment:
    def __init__(self, width, height, obstacle_density, num_goals):
        self.width = width
        self.height = height
        self.obstacle_density = obstacle_density
        self.num_goals = num_goals
        self.ground_truth_map = np.full((height, width), FREE)
        self.discovered_map = np.full((height, width), UNKNOWN)
        self.goal_positions = set() # Store multiple goals
        self._place_obstacles_and_goals()
        # Ensure drone start is not obstacle
        self.ground_truth_map[DRONE_START_POS[1], DRONE_START_POS[0]] = FREE
        self.initial_sense() # Perform initial sensing around start

    def _place_obstacles_and_goals(self):
        # Add borders
        self.ground_truth_map[0, :] = OBSTACLE
        self.ground_truth_map[-1, :] = OBSTACLE
        self.ground_truth_map[:, 0] = OBSTACLE
        self.ground_truth_map[:, -1] = OBSTACLE

        # Add random obstacles
        num_obstacles = int((self.width - 2) * (self.height - 2) * self.obstacle_density)
        placed_obstacles = 0
        while placed_obstacles < num_obstacles:
            x, y = random.randint(1, self.width - 2), random.randint(1, self.height - 2)
            if (x, y) != DRONE_START_POS: # Avoid start pos
                 if self.ground_truth_map[y, x] == FREE:
                     self.ground_truth_map[y, x] = OBSTACLE
                     placed_obstacles += 1

        # Place goals in free spaces, ensure they don't overlap
        placed_goals = 0
        attempts = 0
        max_attempts = (self.width * self.height) * 2 # Safety break
        while placed_goals < self.num_goals and attempts < max_attempts:
            attempts+=1
            gx, gy = random.randint(1, self.width - 2), random.randint(1, self.height - 2)
            goal_candidate = (gx, gy)
            if self.ground_truth_map[gy, gx] == FREE and \
               goal_candidate != DRONE_START_POS and \
               goal_candidate not in self.goal_positions: # Check against existing goals
                self.goal_positions.add(goal_candidate)
                # Goal positions are inherently FREE in the ground truth
                placed_goals += 1
        if placed_goals < self.num_goals:
            print(f"Warning: Only able to place {placed_goals}/{self.num_goals} goals.")


    def initial_sense(self):
        # Perform initial sensing around the start position
        cx, cy = DRONE_START_POS
        radius = 1 # Sense a small area initially
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                 x, y = cx + dx, cy + dy
                 if self.is_valid(x, y):
                    value = self.ground_truth_map[y, x]
                    self.update_discovered_map(x, y, value)


    def is_valid(self, x, y):
        return 0 <= y < self.height and 0 <= x < self.width

    def is_obstacle_truth(self, x, y):
        # Check if coordinates are valid before accessing map
        return self.is_valid(x, y) and self.ground_truth_map[y, x] == OBSTACLE

    def is_obstacle_discovered(self, x, y):
         # Check if coordinates are valid before accessing map
         return self.is_valid(x, y) and self.discovered_map[y, x] == OBSTACLE

    def update_discovered_map(self, x, y, value):
        if self.is_valid(x, y):
            # Allow updating UNKNOWN or FREE cells. Don't overwrite OBSTACLE.
            if self.discovered_map[y, x] != OBSTACLE:
                 self.discovered_map[y, x] = value

    def get_neighbors(self, x, y, diagonal=False):
        neighbors = []
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if diagonal:
             moves += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def calculate_coverage(self):
        """Calculates the fraction of non-obstacle cells discovered."""
        known_mask = self.discovered_map != UNKNOWN
        ground_truth_non_obstacle_mask = self.ground_truth_map != OBSTACLE

        correctly_known_non_obstacle = np.sum(known_mask & ground_truth_non_obstacle_mask)
        total_non_obstacle = np.sum(ground_truth_non_obstacle_mask)

        return correctly_known_non_obstacle / total_non_obstacle if total_non_obstacle > 0 else 1.0


# --- Drone Class ---
class Drone:
    def __init__(self, id, start_pos, sensor_range):
        self.id = id
        self.pos = start_pos
        self.sensor_range = sensor_range
        self.target_frontier = None
        self.current_path = []

    def sense(self, environment):
        """Updates discovered map, returns list of *newly discovered* goal positions."""
        newly_discovered_goals = []
        cx, cy = self.pos
        min_r, max_r = -self.sensor_range, self.sensor_range + 1
        min_c, max_c = -self.sensor_range, self.sensor_range + 1

        y_start, y_end = max(0, cy + min_r), min(environment.height, cy + max_r)
        x_start, x_end = max(0, cx + min_c), min(environment.width, cx + max_c)

        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                sensed_pos = (x, y)
                # Optional: circular sensor range check - but life is hard so it is what it is
                # if euclidean_distance((cx, cy), sensed_pos) > self.sensor_range:
                #      continue

                # Update map first
                value = environment.ground_truth_map[y, x]
                environment.update_discovered_map(x, y, value)

                # Check if it's a goal location
                if sensed_pos in environment.goal_positions:
                    # We don't need to check if it was *already* found here,
                    # the main loop handles the found_goals set.
                    newly_discovered_goals.append(sensed_pos)

        return newly_discovered_goals

    def find_frontiers_and_paths(self, environment):
        """Finds reachable frontiers and paths/distances using BFS from current pos."""
        q = [(0, self.pos, [self.pos])] # (distance, current_pos, path_list)
        visited = {self.pos}
        # Frontier data stores the KNOWN cell bordering the unknown frontier
        frontier_data = {} # {known_neighbor_pos: {'distance': d, 'path': p, 'frontier':(fx,fy)}}

        queue_map = {self.pos: 0} # Track positions in queue to avoid cycles in BFS

        while q:
            # Using simple list pop(0) for BFS correctness, heapq was for Dijkstra-like explore
            # For finding *all* frontiers, BFS is standard.
            if not q: break # Should not happen if logic is right, but safe check
            dist, curr_pos, path = q.pop(0)

            found_frontier_here = False
            for nx, ny in environment.get_neighbors(curr_pos[0], curr_pos[1]):
                 neighbor_pos = (nx, ny)
                 if environment.is_valid(nx, ny):
                    neighbor_state = environment.discovered_map[ny, nx]

                    if neighbor_state == UNKNOWN: # Found an unknown cell
                        # The frontier is the UNKNOWN cell (nx, ny)
                        # The path leads to the KNOWN cell curr_pos
                        frontier_cell = neighbor_pos
                        if curr_pos not in frontier_data: # Store based on known neighbor
                           frontier_data[curr_pos] = {'distance': dist, 'path': path, 'frontier': frontier_cell}
                           # print(f"Drone {self.id} found frontier {frontier_cell} via {curr_pos}")
                        found_frontier_here = True
                        # Don't continue from here into the unknown

                    elif neighbor_state == FREE and neighbor_pos not in visited:
                         if neighbor_pos not in queue_map or queue_map[neighbor_pos] > dist + 1:
                              visited.add(neighbor_pos)
                              new_path = list(path)
                              new_path.append(neighbor_pos)
                              q.append((dist + 1, neighbor_pos, new_path))
                              queue_map[neighbor_pos] = dist + 1

            # Optimization was removed - explore all free space

        return frontier_data # Returns dict where key is the KNOWN cell bordering the unknown frontier

    # TODO: REFACTOR THIS - DELETE AND RECREATE IF YOU MUST - THIS IS THE FRONTIER PLANNING METHOD
    def choose_target_frontier(self, available_frontiers_data, other_drone_positions, assigned_frontiers_set, weight, dist_threshold):
        """Chooses the best frontier based on distance and spacing."""
        # available_frontiers_data = {known_pos: {'distance': d, 'path': p, 'frontier': (fx,fy)}}
        best_frontier_known_neighbor = None # The known cell we path towards
        best_score = -float('inf') # Maximize score = spacing - path_cost

        scored_frontiers = []

        for known_pos, data in available_frontiers_data.items():
            actual_frontier_pos = data['frontier']
            if actual_frontier_pos in assigned_frontiers_set: # Check if the UNKNOWN cell is assigned
                continue # Skip already assigned frontiers

            path_dist = data['distance']
            # Path distance is to the known cell adjacent to the frontier
            # Add 1 to estimate cost to reach the frontier cell itself
            path_dist_to_frontier_cell = path_dist + 1

            min_dist_to_others = float('inf')
            target_pos_for_dist_calc = actual_frontier_pos # Use the unknown cell for spacing

            if not other_drone_positions:
                 min_dist_to_others = float('inf')
            else:
                 for other_pos in other_drone_positions:
                      dist = euclidean_distance(target_pos_for_dist_calc, other_pos)
                      min_dist_to_others = min(min_dist_to_others, dist)

            # Score calculation
            score = (weight * min_dist_to_others) - path_dist_to_frontier_cell
            scored_frontiers.append((score, known_pos)) # Store score and the KNOWN neighbor

        if not scored_frontiers:
            return None

        scored_frontiers.sort(key=lambda x: x[0], reverse=True)
        best_known_neighbor = scored_frontiers[0][1]

        # We return the KNOWN neighbor as the target to pathfind to
        return best_known_neighbor


    def plan_path_to_target(self, target_pos, environment):
        """Finds path from current pos to target using BFS on discovered map."""
        if not target_pos or self.pos == target_pos:
            return []

        q = [(self.pos, [self.pos])] # (current_pos, path_list)
        visited = {self.pos}

        while q:
            curr, path = q.pop(0)
            if curr == target_pos:
                return path # Return the full path

            for nx, ny in environment.get_neighbors(curr[0], curr[1]):
                neighbor = (nx, ny)
                # Move only through known free space
                # Ensure neighbour is valid before checking state
                if environment.is_valid(nx, ny) and \
                   neighbor not in visited and \
                   environment.discovered_map[ny, nx] == FREE:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    q.append((neighbor, new_path))
        return [] # No path found

    def get_next_move_along_path(self):
         """Gets the next position from the current path."""
         if self.current_path and len(self.current_path) > 1:
             return self.current_path[1]
         return self.pos

    def move(self, next_pos, environment):
        """Moves the drone and updates its path"""
        moved = (next_pos != self.pos)
        self.pos = next_pos

        # Update path: remove the step just taken if move was successful and according to plan
        if moved and self.current_path and len(self.current_path) > 1 and self.current_path[1] == next_pos:
             self.current_path.pop(0)
             # If path is now just the current location, clear target to force replan
             if len(self.current_path) <= 1:
                   #print(f"Drone {self.id} reached target {self.target_frontier}")
                   self.target_frontier = None
                   self.current_path = []

        elif not moved or (self.current_path and len(self.current_path)>1 and self.current_path[1] != next_pos) :
             # If we didn't move, or move wasn't the planned one (collision avoid)
             # Clear the path to force replan next step
             self.target_frontier = None
             self.current_path = []
             # print(f"Drone {self.id}: Path cleared due to non-movement or deviation.")


# --- A* Function (Not simulated, create later) ---
# def astar(graph, heuristic, ):
#    ...

# --- Simulation (Single Phase: Exploration until Coverage) ---
def run_simulation(env_config, drone_config, sim_config, coverage_target, spread_weight, dist_threshold):
    # --- Initialization ---
    env = Environment(env_config['width'], env_config['height'],
                      env_config['obstacle_density'], env_config['num_goals'])
    drones = [Drone(i, drone_config['start_pos'], drone_config['sensor_range']) for i in range(drone_config['num_drones'])]

    steps = 0
    found_goals = set() # Store positions of goals found
    first_goal_found_time = -1
    current_coverage = 0.0

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    metrics = {} # Store final metrics

    # --- Simulation Loop ---
    while current_coverage < coverage_target and steps < sim_config['max_steps']:
        if steps % 20 == 0: # Print status periodically
             print(f"Step: {steps}, Coverage: {current_coverage:.3f}, Goals Found: {len(found_goals)}/{env.num_goals}")
        # TODO: If no path exists for exploration then stop stop even if coverage != 100%

        all_drone_positions = {d.id: d.pos for d in drones}
        assigned_frontiers_this_step = set() # Track the actual UNKNOWN cells assigned

        # 1. Sense & Find Goals
        for drone in drones:
            newly_sensed_goals = drone.sense(env)
            for goal_pos in newly_sensed_goals:
                if goal_pos not in found_goals:
                    print(f"\n!!! Goal {goal_pos} found by Drone {drone.id} at step {steps} !!!\n")
                    found_goals.add(goal_pos)
                    if first_goal_found_time == -1:
                        first_goal_found_time = steps

        # 2. Plan Moves
        planned_moves = {} # {drone_id: next_pos}
        drone_frontier_options = {} # {drone_id: frontier_data}

        # Identify which drones need replanning
        for drone in drones:
            # Replan if path is empty/finished or target is no longer valid (e.g., explored)
            needs_replan = False
            if not drone.current_path or len(drone.current_path) <= 1:
                 needs_replan = True
            elif drone.target_frontier and env.discovered_map[drone.target_frontier[1], drone.target_frontier[0]] != FREE :
                 # If the KNOWN cell we are pathing towards is no longer FREE (e.g. became obstacle - unlikely here, or drone landed)
                 # Or more relevant: if the associated UNKNOWN cell is now known
                 frontier_cell_coords = None
                 # Need to retrieve the actual frontier cell associated with target_frontier
                 # This requires finding it again or storing it better. Let's assume replan is safer.
                 # Simplified: Always replan if path is short.
                 needs_replan = True # Force replan if path short for simplicity

            if needs_replan:
                 drone.target_frontier = None
                 drone.current_path = []
                 drone_frontier_options[drone.id] = drone.find_frontiers_and_paths(env)
            else:
                 # Keep following existing path
                 planned_moves[drone.id] = drone.get_next_move_along_path()


        # Assign targets based on spreading score (for drones needing replanning)
        available_frontier_data_combined = {} # Combine all frontier options {known_pos: data}
        for data_dict in drone_frontier_options.values():
             available_frontier_data_combined.update(data_dict)

        # Sort drones needing replanning (e.g., by ID) for deterministic assignment order
        drones_needing_replan_ids = sorted(drone_frontier_options.keys())

        for drone_id in drones_needing_replan_ids:
             drone = next(d for d in drones if d.id == drone_id)
             frontier_data = drone_frontier_options[drone_id]
             if not frontier_data:
                 planned_moves[drone.id] = drone.pos # No frontiers found, stay put
                 continue

             other_drone_positions = {pos for id, pos in all_drone_positions.items() if id != drone_id}

             # Choose the best KNOWN neighbor cell leading to an unassigned frontier
             chosen_known_neighbor = drone.choose_target_frontier(
                 frontier_data,
                 other_drone_positions,
                 assigned_frontiers_this_step, # Pass the set of assigned UNKNOWN cells
                 spread_weight,
                 dist_threshold
             )

             if chosen_known_neighbor and chosen_known_neighbor in frontier_data:
                 # Mark the actual frontier cell (the UNKNOWN one) as assigned
                 actual_frontier_cell = frontier_data[chosen_known_neighbor]['frontier']
                 if actual_frontier_cell not in assigned_frontiers_this_step:
                     assigned_frontiers_this_step.add(actual_frontier_cell)
                     drone.target_frontier = chosen_known_neighbor # Target the known cell
                     drone.current_path = drone.plan_path_to_target(chosen_known_neighbor, env)
                     if drone.current_path:
                           planned_moves[drone.id] = drone.get_next_move_along_path()
                     else:
                           planned_moves[drone.id] = drone.pos # Path not found
                 else:
                      # Best choice was already taken this step, stay put this turn
                      planned_moves[drone.id] = drone.pos
             else:
                 # No suitable frontier assigned
                 planned_moves[drone.id] = drone.pos


        # 3. Collision Resolution & Movement
        final_positions = {}
        occupied_next_steps = {}
        drone_order = list(drones) # random.shuffle?

        for drone in drone_order:
            target_pos = planned_moves.get(drone.id, drone.pos)
            collision = False
            if target_pos != drone.pos:
                if target_pos in occupied_next_steps:
                    collision = True
                elif env.is_obstacle_discovered(target_pos[0], target_pos[1]):
                     print(f"Warning: Drone {drone.id} planned move into discovered obstacle {target_pos}. Waiting.")
                     collision = True

            if not collision:
                final_positions[drone.id] = target_pos
                occupied_next_steps[target_pos] = drone.id
            else:
                final_positions[drone.id] = drone.pos
                if drone.pos not in occupied_next_steps:
                     occupied_next_steps[drone.pos] = drone.id

        # Execute Moves
        for drone in drones:
             next_p = final_positions[drone.id]
             drone.move(next_p, env)

        # 4. Update Coverage & Check Termination
        current_coverage = env.calculate_coverage()
        steps += 1

        # 5. Visualization
        if steps % 5 == 0 or current_coverage >= coverage_target: # Update more often near end
             ax.clear()
             display_map = env.discovered_map.copy()

             # Overlay ALL Goals (maybe mark found ones differently later)
             for gx, gy in env.goal_positions:
                 # Ensure goal drawing doesn't overwrite drone/obstacle if cell is known
                 if display_map[gy, gx] == FREE or display_map[gy, gx] == UNKNOWN :
                      display_map[gy, gx] = GOAL
                 # If goal is found and drone is on it, drone color takes precedence below

             # Overlay Drones
             for drone in drones:
                 if env.is_valid(drone.pos[0], drone.pos[1]):
                     # Make drone visible even if on a goal cell for visualization
                     display_map[drone.pos[1], drone.pos[0]] = DRONE


             ax.imshow(display_map, cmap=cmap, norm=norm)
             title = f"Step: {steps}, Coverage: {current_coverage:.2%}, Goals Found: {len(found_goals)}"
             ax.set_title(title)
             ax.set_xticks(np.arange(-.5, env.width, 1), minor=True)
             ax.set_yticks(np.arange(-.5, env.height, 1), minor=True)
             ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
             ax.tick_params(which='minor', size=0)
             ax.set_xticks([])
             ax.set_yticks([])

             clear_output(wait=True)
             plt.draw()
             plt.pause(0.01) 

    # --- End of Simulation Loop ---
    plt.close(fig)
    plt.ioff()
    print("\n--- Simulation Ended ---")

    # Final Metrics
    metrics['total_steps'] = steps
    metrics['final_coverage'] = current_coverage
    metrics['goals_found_count'] = len(found_goals)
    metrics['all_goals_found'] = (len(found_goals) == env.num_goals)
    metrics['first_goal_time'] = first_goal_found_time if first_goal_found_time != -1 else 'N/A'

    # Map Accuracy (Obstacle IoU)
    discovered_obstacles = (env.discovered_map == OBSTACLE)
    true_obstacles = (env.ground_truth_map == OBSTACLE)
    intersection = np.sum(discovered_obstacles & true_obstacles)
    union = np.sum(discovered_obstacles | true_obstacles)
    metrics['map_accuracy_iou'] = intersection / union if union > 0 else 1.0

    # Determine end reason
    if current_coverage >= coverage_target:
         print(f"Reason: Coverage target ({coverage_target:.0%}) reached at step {steps}.")
         metrics['termination_reason'] = 'Coverage Met'
    else:
         print(f"Reason: Max steps ({sim_config['max_steps']}) reached.")
         metrics['termination_reason'] = 'Max Steps Reached'

    print(f"\n--- Final State ---")
    print(f"Found Goals ({metrics['goals_found_count']}/{env.num_goals}): {found_goals if found_goals else 'None'}")
    print(f"Final Coverage: {metrics['final_coverage']:.3%}")
    print(f"Map Accuracy (Obstacle IoU): {metrics['map_accuracy_iou']:.3f}")
    print(f"Time to Find First Goal: {metrics['first_goal_time']}")
    print(f"Total Steps: {metrics['total_steps']}")


    return metrics, found_goals


# --- Run ---
env_settings = {
    'width': GRID_WIDTH,
    'height': GRID_HEIGHT,
    'obstacle_density': OBSTACLE_DENSITY,
    'num_goals': NUM_GOALS # Pass number of goals
}

drone_settings = {
    'num_drones': NUM_DRONES,
    'start_pos': DRONE_START_POS,
    'sensor_range': DRONE_SENSOR_RANGE
}

sim_settings = {
    'max_steps': MAX_SIMULATION_STEPS
}

final_metrics, list_of_found_goals = run_simulation(
    env_settings,
    drone_settings,
    sim_settings,
    coverage_target=COVERAGE_THRESHOLD,
    spread_weight=SPREADING_WEIGHT,
    dist_threshold=MIN_OTHER_DRONE_DIST_THRESHOLD
)

print("\nList of found goal coordinates:", list_of_found_goals)