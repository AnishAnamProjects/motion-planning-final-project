import open3d as o3d
import numpy as np
from pointcloud import *

class Environment:
    def __init__(self, goal_position, filename, resolution=5):
        '''
        Initialize the environment with a point cloud
        '''
        self.file = filename

        # Read the point cloud file
        # Create a coordinate frame

        self.point_cloud = o3d.io.read_point_cloud(self.file)
        self.point_cloud.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * len(self.point_cloud.points))  # Set color to gray
        
        # extract the limits of the point cloud
        self.x_min, self.x_max = np.min(np.asarray(self.point_cloud.points)[:, 0]), np.max(np.asarray(self.point_cloud.points)[:, 0])
        self.y_min, self.y_max = np.min(np.asarray(self.point_cloud.points)[:, 1]), np.max(np.asarray(self.point_cloud.points)[:, 1])
        self.z_min, self.z_max = np.min(np.asarray(self.point_cloud.points)[:, 2]), np.max(np.asarray(self.point_cloud.points)[:, 2])

        # define the goal position
        self.goal = goal_position

        # define the grid resolution
        self.resolution = resolution

        self.knowledge_map = o3d.geometry.VoxelGrid()
        self.knowledge_map.voxel_size = resolution
        self.obstacle_map = self.generate_obstacle_map()
        self.obstacle_map2 = self.generate_obstacle_map_voxels()
        self.obstacle_voxel_list = self.obstacle_map2.get_voxels()
        self.obstacle_voxel_list = [tuple(voxel.grid_index) for voxel in self.obstacle_voxel_list]
        self.obstacle_voxel_list = set(self.obstacle_voxel_list)  # Convert to a set for faster lookup
        self.detected_obstacles = o3d.geometry.VoxelGrid()
        self.detected_obstacles.voxel_size = resolution
       
        self.detected_free_space = o3d.geometry.VoxelGrid()
        self.detected_free_space.voxel_size = resolution

    def show(self, drone_trajectories=None):
        '''
        Show the point cloud in Open3D
        '''

        # Make it a Voxel grid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.point_cloud, voxel_size=2)

        # Add a new green voxel to the grid above the environment
        goal_voxel = o3d.geometry.Voxel()
        goal_voxel.grid_index = np.array([0, 0, 100])
        goal_voxel.color = [0, 1, 0]  # Set color to green
        voxel_grid.add_voxel(goal_voxel)
        # Visualize the point cloud
        o3d.visualization.draw_geometries([self.point_cloud, voxel_grid])
        
        

    def generate_obstacle_map(self):

        '''
        Generate a truth map from the point cloud
        '''
        # Create a 3D grid based on the limits of the point cloud
        x_bins = np.linspace(self.x_min, self.x_max, self.resolution + 1)
        y_bins = np.linspace(self.y_min, self.y_max, self.resolution + 1)
        z_bins = np.linspace(self.z_min, self.z_max, self.resolution + 1)

        # Create a 3D histogram to count points in each voxel
        hist, edges = np.histogramdd(np.asarray(self.point_cloud.points), bins=(x_bins, y_bins, z_bins))

        obstacle_map = hist > 0
        return obstacle_map
    
    def generate_obstacle_map_voxels(self):
        return o3d.geometry.VoxelGrid.create_from_point_cloud(self.point_cloud, voxel_size=self.resolution)
    
    def query(self, point):
        '''
        Query the obstacle map for a given position
        '''
        voxel_index = self.obstacle_map2.get_voxel(point)
        if voxel_index is not None:
            if tuple(voxel_index) in self.obstacle_voxel_list:
            #if any((voxel.grid_index == voxel_index).all() for voxel in self.obstacle_voxel_list):
                return "Obstacle"
            else:
                return "Free" 
        else:
            return "Out of bounds"
        
        
    def world_to_grid(self, pos):
        """Convert world coordinates to grid indices"""
        x = int((pos[0] - self.x_min) / (self.x_max - self.x_min) * self.resolution)
        y = int((pos[1] - self.y_min) / (self.y_max - self.y_min) * self.resolution)
        z = int((pos[2] - self.z_min) / (self.z_max - self.z_min) * self.resolution)
        return (x, y, z)

#industrial = Environment(goal_position=(10, 10, 10), filename="resources/Industrial_full_scene_0.5hires_lessfloor.ply", resolution=50)
#industrial.show()