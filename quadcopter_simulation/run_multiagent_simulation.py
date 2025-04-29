"""
Run Multi-Agent Drone Simulation
================================
Main entry point to run the multi-agent drone simulation with:
- Exploration by surveyor drones
- Target detection and handling by worker drones
- Performance evaluation

This script integrates with the existing codebase while adding multi-agent functionality.
"""

from multiagent_drone_sim import MultiAgentDroneSimulation
from debug_helpers import monitor_simulation, analyze_exploration_pattern
import matplotlib.pyplot as plt
import numpy as np
import time

# Import the path planning integration
from path_planning_integration import apply_path_planning_patch
from multiagent_planner import MultiAgentPlanner

lowres_filename = "resources/Industrial_full_scene_2.0lowres_lessfloor.ply"
medres_filename = "resources/Industrial_main_part_1.0medres.ply"
# medres_filename = "resources/Industrial_full_scene_1.0medres_lessfloor.ply"
hires_filename = "resources/Industrial_full_scene_0.5hires_lessfloor.ply"

# Apply the advanced path planning patch
apply_path_planning_patch(MultiAgentPlanner)

# Add a safety monkey patch to the MultiAgentPlanner.get_next_waypoint method
original_get_next_waypoint = MultiAgentPlanner.get_next_waypoint

def safe_get_next_waypoint(self, drone_id, drone_type, current_pos):
    # Add a timeout protection for the assignment phase
    import time
    start_time = time.time()
    max_time = 5.0  # Maximum 5 seconds for waypoint calculation
    
    try:
        # Call original method
        if self.phase == "assignment":
            print("Using direct path assignment for safety")
            # Switch to execution phase immediately
            self.phase = "execution"
            
            # For workers, assign direct paths to target
            if drone_type == 'worker':
                worker_id = drone_id
                # Simple direct path to target
                if self.target_pos is not None:
                    self.worker_paths[worker_id] = [current_pos, self.target_pos]
                    print(f"Worker {worker_id} assigned direct path to target")
                    return self.target_pos
            
            # For surveyors, continue normal operation
            return original_get_next_waypoint(self, drone_id, drone_type, current_pos)
        else:
            # Normal operation for other phases
            return original_get_next_waypoint(self, drone_id, drone_type, current_pos)
    except Exception as e:
        print(f"Error in get_next_waypoint: {e}")
        # Return current position as a failsafe
        return current_pos
    finally:
        # Check for timeout
        elapsed = time.time() - start_time
        if elapsed > 1.0:  # Log slow operations
            print(f"Waypoint calculation took {elapsed:.2f}s")

# Apply the monkey patch
MultiAgentPlanner.get_next_waypoint = safe_get_next_waypoint

def run_simulation(filename = "", cloud_res=1.0, n_surveyors=3, n_workers=2, space_limit=10.0, debug=True):
    """Run simulation with the given parameters and collect metrics"""
    sim = MultiAgentDroneSimulation(
        filename,
        cloud_res,
        n_surveyors=n_surveyors,
        n_workers=n_workers,
        space_limit=space_limit
    )
    
    print(f"Starting simulation with {n_surveyors} surveyors and {n_workers} workers...")
    
    # Enable debugging if requested
    if debug:
        sim = monitor_simulation(sim)
    
    # Run simulation
    sim.simulate()
    
    return sim.metrics

def run_parameter_study():
    """Run multiple simulations with different parameters to analyze performance"""
    print("Running parameter study...")
    
    # Different configurations to test
    configs = [
        {"n_surveyors": 2, "n_workers": 1, "label": "2S-1W"},
        {"n_surveyors": 3, "n_workers": 1, "label": "3S-1W"},
        {"n_surveyors": 3, "n_workers": 2, "label": "3S-2W"},
        {"n_surveyors": 4, "n_workers": 2, "label": "4S-2W"}
    ]
    
    # Metrics to collect
    results = {
        "time_to_target": [],
        "final_coverage": [],
        "collisions": [],
        "labels": []
    }
    
    # Run each configuration
    for config in configs:
        print(f"\nTesting configuration: {config['label']}")
        metrics = run_simulation(
            n_surveyors=config["n_surveyors"],
            n_workers=config["n_workers"]
        )
        
        # Collect metrics
        results["labels"].append(config["label"])
        results["time_to_target"].append(metrics["time_to_target"] or 0)
        results["final_coverage"].append(metrics["coverage_rate"][-1] if metrics["coverage_rate"] else 0)
        results["collisions"].append(metrics["collision_rate"])
        
        # Add a delay between runs to make sure we can see each one
        time.sleep(2)
    
    # Plot comparative results
    plot_comparative_results(results)

def plot_comparative_results(results):
    """Plot comparative metrics across different configurations"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(results["labels"]))
    width = 0.3
    
    # Time to target
    ax1.bar(x, results["time_to_target"], width, label='Time (s)')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Time to Target Detection')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results["labels"])
    ax1.grid(True, axis='y')
    
    # Coverage
    ax2.bar(x, results["final_coverage"], width, label='Coverage (%)')
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Coverage (%)')
    ax2.set_title('Final Coverage Percentage')
    ax2.set_xticks(x)
    ax2.set_xticklabels(results["labels"])
    ax2.grid(True, axis='y')
    
    # Collisions
    ax3.bar(x, results["collisions"], width, label='Collisions')
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Count')
    ax3.set_title('Number of Collisions')
    ax3.set_xticks(x)
    ax3.set_xticklabels(results["labels"])
    ax3.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('multiagent_results.png')
    plt.show()

if __name__ == "__main__":
    # Uncomment to run a parameter study across multiple configurations
    # run_parameter_study()
    
    # Run a single simulation
    run_simulation(medres_filename, cloud_res = 1.0,n_surveyors=3, n_workers=2)
