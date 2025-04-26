import open3d as o3d
import numpy as np
from pointcloud import *

class Environment:
    def __init__(self, goal_position, filename, resolution=0.1):
        '''
        Initialize the environment with a point cloud
        '''
        self.file = filename

        # Read the point cloud file
        # Create a coordinate frame

        self.point_cloud = o3d.io.read_point_cloud(self.file)
        self.point_cloud.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(self.point_cloud.points))  # Set color to gray
        
        # extract the limits of the point cloud
        self.x_min, self.x_max = np.min(np.asarray(self.point_cloud.points)[:, 0]), np.max(np.asarray(self.point_cloud.points)[:, 0])
        self.y_min, self.y_max = np.min(np.asarray(self.point_cloud.points)[:, 1]), np.max(np.asarray(self.point_cloud.points)[:, 1])
        self.z_min, self.z_max = np.min(np.asarray(self.point_cloud.points)[:, 2]), np.max(np.asarray(self.point_cloud.points)[:, 2])

        # define the goal position
        self.goal = goal_position

        # define the grid resolution
        self.resolution = int(1/resolution)

        self.obstacle_map = self.generate_obstacle_map()
    def show(self):
        '''
        Show the point cloud in Open3D
        '''
        # Visualize the point cloud
        o3d.visualization.draw_geometries([self.point_cloud], mesh_show_wireframe=True)
        
        #Visualize.plyfile3D(self.file)

    def generate_obstacle_map(self, resolution=0.1):

        '''
        Generate a truth map from the point cloud
        '''
        # Create a 3D grid based on the limits of the point cloud
        x_bins = np.linspace(self.x_min, self.x_max, self.resolution)
        y_bins = np.linspace(self.y_min, self.y_max, self.resolution)
        z_bins = np.linspace(self.z_min, self.z_max, self.resolution)

        # Create a 3D histogram to count points in each voxel
        hist, edges = np.histogramdd(np.asarray(self.point_cloud.points), bins=(x_bins, y_bins, z_bins))
        obstacle_map = hist > 0
        return obstacle_map
        
    def query(self, voxel):
        '''
        Query the obstacle map for a given position
        '''

        # Check if the voxel is within the bounds of the grid
        if (voxel[0] < 0 or voxel[0] >= self.resolution or
            voxel[1] < 0 or voxel[1] >= self.resolution or
            voxel[2] < 0 or voxel[2] >= self.resolution):
            print("Voxel out of bounds")
            return "Out of bounds"
        # Check if the voxel is the goal position
        elif voxel == self.world_to_grid(self.goal):
            return "Goal"
        # check if the voxel is in the obstacle map
        elif self.obstacle_map[voxel]:
            return "Obstacle"
        # otherwise the voxel is free
        else:
            return "Free"    
    
    def world_to_grid(self, pos):
        """Convert world coordinates to grid indices"""
        x = int((pos[0] - self.x_min) / (self.x_max - self.x_min) * self.resolution)
        y = int((pos[1] - self.y_min) / (self.y_max - self.y_min) * self.resolution)
        z = int((pos[2] - self.z_min) / (self.z_max - self.z_min) * self.resolution)
        return (x, y, z)

industrial = Environment(goal_position=(10, 10, 10), filename="resources/Industrial_full_scene_0.5hires_lessfloor.ply", resolution=0.01)
industrial.show()