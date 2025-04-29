"""
Run Multi-Agent Drone Simulation with RRT* Path Planning
=======================================================
Main entry point to run the drone simulation with RRT* path planning.
"""

from rrt_drone_sim import RRTStarDroneSimulation
from debug_helpers import monitor_simulation
import argparse
import time

lowres_filename = "resources/Industrial_full_scene_2.0lowres_lessfloor.ply"
medres_filename = "resources/Industrial_main_part_1.0medres.ply"
# medres_filename = "resources/Industrial_full_scene_1.0medres_lessfloor.ply"
hires_filename = "resources/Industrial_full_scene_0.5hires_lessfloor.ply"

def run_simulation(filename = "", cloud_res=1.0, n_surveyors=3, n_workers=2, space_limit=10.0, debug=True):
    """Run RRT*-based simulation with the given parameters"""

    print(f"Starting RRT* simulation with {n_surveyors} surveyors and {n_workers} workers...")
    
    # Create simulation
    sim = RRTStarDroneSimulation(
        filename,
        cloud_res,
        n_surveyors=n_surveyors,
        n_workers=n_workers,
        space_limit=space_limit
    )
    
    # Enable debugging if requested
    if debug:
        sim = monitor_simulation(sim)
    
    # Run simulation
    sim.simulate()
    
    return sim.metrics

def compare_with_astar():
    """Run simulations with both A* and RRT* and compare results"""
    from multiagent_drone_sim import MultiAgentDroneSimulation
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("Running comparison between A* and RRT* planners...")
    
    # Run A* simulation
    print("\n=== Running A* Simulation ===")
    astar_sim = MultiAgentDroneSimulation(n_surveyors=3, n_workers=2)
    astar_sim.simulate()
    astar_metrics = astar_sim.metrics
    
    # Allow some time between simulations
    time.sleep(2)
    
    # Run RRT* simulation
    print("\n=== Running RRT* Simulation ===")
    rrt_sim = RRTStarDroneSimulation(n_surveyors=3, n_workers=2)
    rrt_sim.simulate()
    rrt_metrics = rrt_sim.metrics
    
    # Compare results
    print("\n=== Comparison Results ===")
    print(f"Time to Target Detection:")
    print(f"  A*: {astar_metrics['time_to_target']:.2f}s")
    print(f"  RRT*: {rrt_metrics['time_to_target']:.2f}s")
    
    print(f"Final Coverage Rate:")
    print(f"  A*: {astar_metrics['coverage_rate'][-1]:.2f}%")
    print(f"  RRT*: {rrt_metrics['coverage_rate'][-1]:.2f}%")
    
    print(f"Collision Rate:")
    print(f"  A*: {astar_metrics['collision_rate']}")
    print(f"  RRT*: {rrt_metrics['collision_rate']}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Normalize time series lengths
    min_len = min(len(astar_metrics["coverage_rate"]), len(rrt_metrics["coverage_rate"]))
    astar_coverage = astar_metrics["coverage_rate"][:min_len]
    rrt_coverage = rrt_metrics["coverage_rate"][:min_len]
    x = list(range(min_len))
    
    # Coverage rate comparison
    ax1.plot(x, astar_coverage, 'b-', label='A* Coverage')
    ax1.plot(x, rrt_coverage, 'r-', label='RRT* Coverage')
    ax1.set_xlabel('Frame (x10)')
    ax1.set_ylabel('Coverage Rate (%)')
    ax1.set_title('Coverage Rate Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Metrics comparison
    metrics = ['Time to Target', 'Collision Rate']
    astar_values = [astar_metrics['time_to_target'] or 0, astar_metrics['collision_rate']]
    rrt_values = [rrt_metrics['time_to_target'] or 0, rrt_metrics['collision_rate']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x - width/2, astar_values, width, label='A*')
    ax2.bar(x + width/2, rrt_values, width, label='RRT*')
    
    ax2.set_ylabel('Value')
    ax2.set_title('Performance Metrics Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('astar_vs_rrtstar.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multi-agent drone simulation with RRT* path planning')
    parser.add_argument('--surveyors', type=int, default=3, help='Number of surveyor drones')
    parser.add_argument('--workers', type=int, default=2, help='Number of worker drones')
    parser.add_argument('--space', type=float, default=10.0, help='Size of environment')
    parser.add_argument('--compare', action='store_true', help='Compare with A* planner')
    parser.add_argument('--no-debug', action='store_true', help='Disable debugging output')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_with_astar()
    else:
        run_simulation(
            medres_filename,
            cloud_res = 1.0,
            n_surveyors=args.surveyors,
            n_workers=args.workers,
            space_limit=args.space,
            debug=not args.no_debug
        )
