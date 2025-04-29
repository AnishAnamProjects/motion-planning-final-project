
"""
Path Planning Integration
========================
This script patches the existing MultiAgentPlanner class to use advanced
path planning algorithms without modifying your original files.

Simply import and run apply_path_planning_patch() at the start of your 
run_multiagent_simulation.py script.
"""

from advanced_pathfinding import PathPlanner, simplify_path
import types

def apply_path_planning_patch(multiagent_planner_class):
    """Patch the MultiAgentPlanner class to use advanced path planning
    
    Args:
        multiagent_planner_class: The MultiAgentPlanner class to patch
    """
    original_init = multiagent_planner_class.__init__
    original_find_path = multiagent_planner_class.find_path
    
    def patched_init(self, n_surveyors, n_workers, space_limits, grid_dims=(20, 20, 20)):
        # Call original __init__
        original_init(self, n_surveyors, n_workers, space_limits, grid_dims)
        
        # Add path planner
        self.path_planner = PathPlanner(self.shared_map, space_limits, grid_dims)
        self.path_planner.default_algorithm = "rrt"  # Use RRT by default
        
        print("Enhanced path planning enabled - using RRT algorithm")
    
    def patched_find_path(self, start, goal, algorithm=None):
        """Enhanced path finding method using advanced algorithms"""
        try:
            # Use the advanced path planner
            if not algorithm:
                algorithm = self.path_planner.default_algorithm
            
            # Add safety timeout
            import time
            start_time = time.time()
            max_time = 3.0  # Maximum 3 seconds for path planning
            
            print(f"Planning path from {start} to {goal} using {algorithm}")
            
            # If direct path is possible, just return it (faster)
            start_world = self.path_planner.grid_to_world(start)
            goal_world = self.path_planner.grid_to_world(goal)

            if self.path_planner.is_path_collision_free(start_world, goal_world):
                print(f"Direct path possible - skipping complex planning")
                return [start, goal]
            
            # Use a simpler algorithm for assignment phase which needs to be fast
            if hasattr(self, 'phase') and self.phase == "assignment":
                algorithm = "rrt"  # Use faster RRT for assignment
                print(f"Assignment phase detected - using faster RRT algorithm")
                
                # Set up a very direct path with minimal waypoints
                self.path_planner.max_iterations = 200
                self.path_planner.goal_sample_rate = 0.5  # Very high goal bias
                
            # Perform path planning with timeout protection
            path = self.path_planner.find_path(start, goal, algorithm)
            
            # Check timeout
            if time.time() - start_time > max_time:
                print(f"Path planning taking too long (>{max_time}s), simplifying approach")
                # Fall back to direct path if too slow
                return [start, goal]
            
            if path and len(path) > 3:
                # Simplify path if it exists and has sufficient waypoints
                # But limit time spent on simplification
                original_length = len(path)
                if time.time() - start_time < max_time * 0.8:  # Only if we have time left
                    try:
                        path = simplify_path(path, self.path_planner, max_distance=2.0)
                        print(f"Path simplified from {original_length} to {len(path)} waypoints")
                    except Exception as e:
                        print(f"Error during path simplification: {e}, using original path")
                else:
                    print(f"Skipping path simplification due to time constraints")
            
            return path
        except Exception as e:
            # Fall back to original method if there's any error
            print(f"Error in advanced path planning: {e}, falling back to A*")
            return original_find_path(self, start, goal)
    
    # Apply patches
    multiagent_planner_class.__init__ = patched_init
    multiagent_planner_class.find_path = patched_find_path