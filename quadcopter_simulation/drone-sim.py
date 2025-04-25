from drone import Drone
import trajGen3D
import controller
import pointcloud
from runsim import *
import numpy as np
import random
from quadPlot import plot_quad_3d

lowres_filename = "resources/Industrial_full_scene_2.0lowres_lessfloor.ply"
medres_filename = "resources/Industrial_main_part_1.0medres.ply"
hires_filename = "resources/Industrial_full_scene_0.5hires_lessfloor.ply"

class DroneSimulation:
    def __init__(self, filename = "", cloud_resolution = 1.0, n_drones=4, space_limit=10.0):
        '''
        cloud_resolution = how far apart the cloud points are
        '''

        # Simulation parameters
        self.cloud_res = 1/cloud_resolution
        self.animation_frequency = 30
        self.dt = 5.0 / self.animation_frequency
        self.space_limit = space_limit
        self.n_drones = n_drones
        self.SAFE_DISTANCE = 1.5  # Minimum distance between drones
        self.SPEED = 0.1  # Units per frame

        # # World bounds
        # self.x_min, self.x_max = 0.0, space_limit
        # self.y_min, self.y_max = 0.0, space_limit
        # self.z_min, self.z_max = 0.0, space_limit

        # Get world limits (mins and maxes) from point cloud
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = pointcloud.get_min_max(filename)
        # World limits to be sent to quadPlot.py
        self.limits = (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)
        
        # Voxel grid for sensing
        values = (self.x_max, self.y_max, self.z_max)
        self.nx, self.ny, self.nz = tuple(values * self.cloud_res for values in values)
        self.nx = int(self.nx)
        self.ny = int(self.ny)
        self.nz = int(self.nz)
        self.xs = np.linspace(self.x_min, self.x_max, self.nx)
        self.ys = np.linspace(self.y_min, self.y_max, self.ny)
        self.zs = np.linspace(self.z_min, self.z_max, self.nz)
        Xc, Yc, Zc = np.meshgrid(self.xs, self.ys, self.zs, indexing='ij')
        self.voxel_centers = np.stack([Xc.ravel(), Yc.ravel(), Zc.ravel()], axis=1)
        
        # Occupancy maps: 0=free, 1=obs, 2=goal, 255=unknown
        self.truth_map = np.zeros((self.nx, self.ny, self.nz), dtype=np.uint8)
        self.known_map = 255 * np.ones_like(self.truth_map)
        
        # Initialize drones and their states
        self.drones = [None] * n_drones
        self.waypoints = [None] * n_drones
        self.current_waypoint_idx = [0] * n_drones
        self.done = [False] * n_drones
        
        # Sensing radius squared (for efficiency)
        self.sense_r2 = 2.0**2
        
        # Initialize the environment
        self.setup_environment()
        
    def setup_environment(self):
        """Add random obstacles to the environment"""
        num_obstacles = 3
        for _ in range(num_obstacles):
            box_min = np.array([
                random.randint(2, self.nx-4),
                random.randint(2, self.ny-4),
                random.randint(1, self.nz-4)
            ])
            box_size = np.array([3, 3, 3])
            box_max = box_min + box_size
            self.truth_map[box_min[0]:box_max[0], 
                          box_min[1]:box_max[1], 
                          box_min[2]:box_max[2]] = 1
    
    def world_to_grid(self, pos):
        """Convert world coordinates to grid indices"""
        ix = int((pos[0]-self.x_min)/(self.x_max-self.x_min)*(self.nx-1))
        iy = int((pos[1]-self.y_min)/(self.y_max-self.y_min)*(self.ny-1))
        iz = int((pos[2]-self.z_min)/(self.z_max-self.z_min)*(self.nz-1))
        return np.clip(ix,0,self.nx-1), np.clip(iy,0,self.ny-1), np.clip(iz,0,self.nz-1)
    
    def initialize_drones(self, start_positions=None):
        """Initialize drones with given or default start positions"""
        if start_positions is None:
            # Create a grid of start positions
            grid = np.meshgrid(np.arange(0, self.n_drones, 1), np.arange(0, self.n_drones, 1))
            start_positions = np.array(grid).T.reshape(-1, 2)
            start_positions = np.hstack((start_positions, np.zeros((start_positions.shape[0], 1))))
            # Scale positions to world coordinates
            start_positions = start_positions * (self.space_limit / self.n_drones)
        
        for i in range(self.n_drones):
            pos = start_positions[i]
            attitude = (0, 0, 0)  # Initial attitude (roll, pitch, yaw)
            start_state = [pos, attitude]
            self.drones[i] = Drone(start_state)
    
    def set_waypoints(self, waypoints_list):
        """Set waypoints for each drone"""
        assert len(waypoints_list) == self.n_drones, "Must provide waypoints for each drone"
        self.waypoints = waypoints_list
        self.current_waypoint_idx = [0] * self.n_drones
        self.done = [False] * self.n_drones
    
    def update_known_map(self, drone_pos):
        """Update the known map based on drone's current position"""
        d2 = np.sum((self.voxel_centers - drone_pos)**2, axis=1)
        self.known_map.ravel()[d2 <= self.sense_r2] = self.truth_map.ravel()[d2 <= self.sense_r2]
    
    def check_collision(self, pos):
        """Check if position collides with obstacle"""
        ix, iy, iz = self.world_to_grid(pos)
        return self.truth_map[ix, iy, iz] == 1

    def get_repulsive_force(self, pos1, pos2, safe_dist):
        """Calculate repulsive force between two points"""
        diff = pos1 - pos2
        dist = np.linalg.norm(diff)
        if dist < safe_dist:
            # Force magnitude inversely proportional to distance
            magnitude = (safe_dist - dist) / safe_dist
            return (diff / dist) * magnitude * 2.0
        return np.zeros(3)
    
    def control_loop(self, frame_idx):
        """Update drone positions for one animation frame"""
        if all(self.done):
            return None
            
        world_frames = []
        for i in range(self.n_drones):
            if self.done[i]:
                world_frames.append(self.drones[i].world_frame())
                continue
                
            current_pos = np.array(self.drones[i].position())
            target_pos = self.waypoints[i][self.current_waypoint_idx[i]]
            
            # Direction to current waypoint
            to_target = target_pos - current_pos
            dist_to_target = np.linalg.norm(to_target)
            
            if dist_to_target < self.SPEED:
                # Reached current waypoint
                self.current_waypoint_idx[i] += 1
                if self.current_waypoint_idx[i] >= len(self.waypoints[i]):
                    self.done[i] = True
                    print(f"Drone {i} completed its path!")
                current_pos = target_pos
            else:
                # Calculate movement direction
                direction = to_target / dist_to_target
                
                # Add repulsive forces from other drones
                avoidance = np.zeros(3)
                for j, other_drone in enumerate(self.drones):
                    if i != j and not self.done[j]:
                        force = self.get_repulsive_force(
                            current_pos, 
                            np.array(other_drone.position()), 
                            self.SAFE_DISTANCE
                        )
                        avoidance += force
                
                # Combine goal direction with avoidance
                if np.any(avoidance):
                    movement = direction + avoidance
                    movement = movement / np.linalg.norm(movement)
                else:
                    movement = direction
                
                # Update position
                new_pos = current_pos + movement * self.SPEED
                
                # Check bounds and collisions
                new_pos = np.clip(new_pos, 
                                [self.x_min, self.y_min, self.z_min],
                                [self.x_max, self.y_max, self.z_max])
                
                if not self.check_collision(new_pos):
                    current_pos = new_pos
                else:
                    print(f"Drone {i} detected collision, stopping")
                    self.done[i] = True
            
            # Update drone position and known map
            self.drones[i].state[0:3] = current_pos  # Update position in state vector
            self.update_known_map(current_pos)
            world_frames.append(self.drones[i].world_frame())
        
        return np.stack(world_frames)
    
    def simulate(self):
        """Run the simulation with animation"""
        # Combine all waypoints for visualization
        all_waypoints = np.vstack([wp for wp in self.waypoints])
        
        # Run animation
        plot_quad_3d(
            all_waypoints,
            self.control_loop,
            lambda idx: self.known_map,
            self.voxel_centers,
            self.limits
        )

    # def generate_waypoints(self, num_drones, space_limit=10.0):
    #     waypoints = [
    #         np.array([[0, 0, 8], [self.space_limit, 0, 8]]),  # Drone 0: front left to front right
    #         np.array([[self.space_limit, 0, 8], [self.space_limit, self.space_limit, 8]]),  # Drone 1: front right to back right
    #         np.array([[0, self.space_limit, 8], [0, 0, 8]]),  # Drone 2: back left to front left
    #         np.array([[self.space_limit, self.space_limit, 8], [0, self.space_limit, 8]])   # Drone 3: back right to back left
    #     ]
    #     return waypoints
    
    def generate_waypoints(self, num_drones, limits):

        x_min, x_max, y_min, y_max, z_min, z_max = limits

        waypoints = [
            np.array([[0, 0, 8], [x_max, 0, 8]]),  # Drone 0: front left to front right
            np.array([[x_max, 0, 8], [x_max, y_max, 8]]),  # Drone 1: front right to back right
            np.array([[0, y_max, 8], [0, 0, 8]]),  # Drone 2: back left to front left
            np.array([[x_max, y_max, 8], [0, y_max, 8]])   # Drone 3: back right to back left
        ]
        return waypoints

# Example usage
if __name__ == "__main__":

    # # Visualize the point cloud being used
    # pointcloud.Visualize.open3D(hires_filename)

    # Create simulation
    sim = DroneSimulation(medres_filename, cloud_resolution = 1.0, n_drones=4, space_limit=10.0)
    
    # Initialize drones
    sim.initialize_drones()
    
    # Set waypoints (example: each drone goes to a corner)
    waypoints = sim.generate_waypoints(4, sim.limits)
    
    sim.set_waypoints(waypoints)
    
    # Run simulation
    sim.simulate()