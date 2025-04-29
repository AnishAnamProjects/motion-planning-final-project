"""
Debug Helpers for Multi-Agent Drone Simulation
=============================================
Provides utility functions to debug and visualize the simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_drone_positions(drones, drone_roles, planner):
    """
    Create a static visualization of current drone positions and map state
    
    Args:
        drones: List of drone objects
        drone_roles: List of drone roles ('surveyor' or 'worker')
        planner: MultiAgentPlanner or RRTStarPlanner instance
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot drone positions
    colors = ['b', 'g', 'r', 'y', 'c', 'm']
    for i, drone in enumerate(drones):
        pos = drone.position()
        color = colors[i % len(colors)]
        marker = 'o' if drone_roles[i] == 'surveyor' else 's'
        ax.scatter(pos[0], pos[1], pos[2], c=color, marker=marker, s=100, 
                  label=f"{drone_roles[i].capitalize()} {i}")
    
    # Check what kind of planner we have (original or RRT*)
    if hasattr(planner, 'planner'):
        # This is an RRTStarPlanner
        target_pos = planner.planner.target_pos
        frontier = None  # RRT* planner doesn't have frontier attribute
    else:
        # This is the original planner
        target_pos = planner.target_pos
        # Check if surveyors exist and have frontiers
        frontier = planner.surveyors[0].frontier if hasattr(planner, 'surveyors') and planner.surveyors else None
    
    # Plot target location if found
    if target_pos is not None:
        ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
                  c='k', marker='*', s=200, label='Target')
    
    # Plot frontier cells if available
    if frontier:
        frontier_cells = list(frontier)
        frontier_world = np.array([planner.grid_to_world(cell) for cell in frontier_cells])
        ax.scatter(frontier_world[:, 0], frontier_world[:, 1], frontier_world[:, 2], 
                  c='grey', marker='.', alpha=0.3, label='Frontier')
    
    # Set axis limits
    if hasattr(planner, 'space_limits'):
        x_min, x_max, y_min, y_max, z_min, z_max = planner.space_limits
    else:
        # Get from planner.planner if available
        x_min, x_max, y_min, y_max, z_min, z_max = planner.planner.space_limits if hasattr(planner, 'planner') else (0, 10, 0, 10, 0, 10)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Drone Positions and Map State')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('drone_positions.png')
    plt.close()
    
    print("Drone positions visualization saved to 'drone_positions.png'")

def analyze_exploration_pattern(planner, n_steps=10):
    """
    Analyze exploration pattern for a surveyor drone
    
    Args:
        planner: MultiAgentPlanner instance
        n_steps: Number of steps to simulate
    """
    # Choose the first surveyor for analysis
    if not planner.surveyors:
        print("No surveyors available for analysis")
        return
        
    surveyor = planner.surveyors[0]
    
    # Simulate drone movement with different exploration strategies
    start_pos = np.array([
        (planner.space_limits[0] + planner.space_limits[1]) / 2,
        (planner.space_limits[2] + planner.space_limits[3]) / 2,
        (planner.space_limits[4] + planner.space_limits[5]) / 2
    ])
    
    # Store paths for different strategies
    paths = {
        'frontier': [start_pos],
        'random': [start_pos],
        'spiral': [start_pos]
    }
    
    # Original strategy
    original_pattern = surveyor.exploration_pattern
    
    # Simulate movement for each strategy
    for strategy in paths.keys():
        # Set strategy
        surveyor.exploration_pattern = strategy
        current_pos = start_pos.copy()
        
        # Generate path
        for _ in range(n_steps):
            next_pos = surveyor.get_next_waypoint(current_pos)
            paths[strategy].append(next_pos)
            current_pos = next_pos
    
    # Restore original strategy
    surveyor.exploration_pattern = original_pattern
    
    # Visualize paths
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {'frontier': 'b', 'random': 'r', 'spiral': 'g'}
    for strategy, path in paths.items():
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], c=colors[strategy], marker='o', 
                label=f"{strategy.capitalize()} Strategy")
    
    # Set axis limits
    x_min, x_max, y_min, y_max, z_min, z_max = planner.space_limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Exploration Strategies Comparison')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('exploration_strategies.png')
    plt.close()
    
    print("Exploration strategies visualization saved to 'exploration_strategies.png'")

def visualize_map_coverage(planner, truth_map):
    """
    Visualize coverage of the environment map
    
    Args:
        planner: MultiAgentPlanner instance
        truth_map: Ground truth map
    """
    # Get shared map and dimensions
    shared_map = planner.shared_map.grid
    nx, ny, nz = shared_map.shape
    
    # Create 2D slices at different heights
    n_slices = min(4, nz)
    slice_heights = np.linspace(0, nz-1, n_slices).astype(int)
    
    fig, axes = plt.subplots(2, n_slices, figsize=(16, 8))
    
    # Plot shared map slices (top row)
    for i, z in enumerate(slice_heights):
        ax = axes[0, i]
        map_slice = shared_map[:, :, z]
        im = ax.imshow(map_slice.T, origin='lower', cmap='viridis', 
                     vmin=0, vmax=255, interpolation='none')
        ax.set_title(f'Known Map (z={z})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    # Plot truth map slices (bottom row)
    for i, z in enumerate(slice_heights):
        ax = axes[1, i]
        truth_slice = truth_map[:, :, z]
        im = ax.imshow(truth_slice.T, origin='lower', cmap='viridis', 
                     vmin=0, vmax=255, interpolation='none')
        ax.set_title(f'Truth Map (z={z})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0, 1, 2, 255])
    cbar.set_ticklabels(['Free', 'Obstacle', 'Target', 'Unknown'])
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig('map_coverage.png')
    plt.close()
    
    print("Map coverage visualization saved to 'map_coverage.png'")

def print_drone_status(drones, drone_roles, planner):
    """
    Print detailed status of all drones
    
    Args:
        drones: List of drone objects
        drone_roles: List of drone roles ('surveyor' or 'worker')
        planner: MultiAgentPlanner or RRTStarPlanner instance
    """
    print("\n=== DRONE STATUS REPORT ===")
    
    # Check what kind of planner we have (original or RRT*)
    if hasattr(planner, 'planner'):
        # This is an RRTStarPlanner
        current_phase = planner.planner.phase
        target_found = planner.planner.target_pos is not None
        target_pos = planner.planner.target_pos
    else:
        # This is the original planner
        current_phase = planner.phase
        target_found = planner.target_pos is not None
        target_pos = planner.target_pos
    
    print(f"Current phase: {current_phase}")
    print(f"Target found: {'Yes' if target_found else 'No'}")
    
    if target_found:
        print(f"Target position: {target_pos}")
        
    print("\nIndividual drone status:")
    for i, drone in enumerate(drones):
        role = drone_roles[i]
        pos = drone.position()
        
        # Get grid position using the appropriate planner
        grid_pos = planner.world_to_grid(pos)
        
        print(f"\nDrone {i} ({role.upper()}):")
        print(f"  Position: {pos}")
        print(f"  Grid position: {grid_pos}")
        
        # Role-specific info with appropriate checks
        if role == 'surveyor':
            print("  Exploration status: Active")
        else:  # worker
            if hasattr(planner, 'planner') and hasattr(planner.planner, 'rrt_planner'):
                path = planner.planner.rrt_planner.paths[i] if i < len(planner.planner.rrt_planner.paths) else None
            else:
                path = None
                
            print(f"  Path to target: {'Yes' if path else 'No'}")
            print(f"  Remaining waypoints: {len(path) if path else 0}")
    
    # Coverage information
    stats = planner.get_exploration_stats()
    print("\nExploration statistics:")
    print(f"  Coverage: {stats['coverage_percent']:.2f}%")
    print(f"  Collisions: {stats['collisions']}")
    if stats["target_found_time"] is not None:
        print(f"  Time to find target: {stats['target_found_time']:.2f}s")

def monitor_simulation(simulation, interval=30):
    """
    Monitor the simulation and generate debug information at intervals
    
    Args:
        simulation: MultiAgentDroneSimulation instance
        interval: Number of frames between debug outputs
    """
    # Add hook to simulation's control_loop
    original_control_loop = simulation.control_loop
    
    def monitored_control_loop(frame_idx):
        # Call original control loop
        result = original_control_loop(frame_idx)
        
        # Generate debug info at intervals
        if frame_idx % interval == 0 and frame_idx > 0:
            print(f"\n=== DEBUG INFO (Frame {frame_idx}) ===")
            print_drone_status(simulation.drones, simulation.drone_roles, simulation.planner)
            
            # Generate visualizations less frequently
            if frame_idx % (interval * 5) == 0:
                visualize_drone_positions(
                    simulation.drones, 
                    simulation.drone_roles, 
                    simulation.planner
                )
                visualize_map_coverage(simulation.planner, simulation.truth_map)
        
        return result
    
    # Replace control loop with monitored version
    simulation.control_loop = monitored_control_loop
    
    print("Simulation monitoring enabled")
    
    return simulation