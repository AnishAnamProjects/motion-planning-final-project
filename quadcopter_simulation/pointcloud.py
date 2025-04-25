import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

def get_points(filename = ""):
    '''
    Reads in a .ply point cloud file and puts the points
      into a, y, and z arrays, and zeros the lowest 
      data number to to start at the zero of each axis. 
    '''

    mesh = trimesh.load(filename)
    points = mesh.vertices

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2] 

    # shift the data to start at axes zeros
    x, y, z = zero_data(x, y, z)

    return (x, y, z)

def get_min_max(filename = ""):
    '''
    Gets the x, y, z, minimums and maximums
      from a .ply point cloud file.
    '''

    x, y, z, = get_points(filename)

    # x_min, x_max = int(x.min()), int(x.max())
    # y_min, y_max = int(y.min()), int(y.max())
    # z_min, z_max = int(z.min()), int(z.max())
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    return x_min, x_max, y_min, y_max, z_min, z_max

class Visualize():

    def __init__(self, filename = ""):

        self.pc_file = filename
        
        return None
    
    # You will need to install `python3 -m pip install trimesh`
    # Render point cloud in a pyPlot scatter plot
    def trimesh3D(filename = ""):
            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        x, y, z, = get_points(filename)
        
        # some colormap options = viridis, plasma, inferno, magma, cividis
        ax.scatter(x, y, z, marker='.', cmap='viridis', c=z)
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_aspect('equal')
        
        plt.show()
    
        return None
    
    # You will need to install `python3 -m pip install open3d`
    # render point cloud in Open3D
    def open3D(filename = ""):
    
        cloud = o3d.io.read_point_cloud(filename) # Read point cloud
        o3d.visualization.draw_geometries([cloud])    # Visualize point cloud
    
        return None
    
    # You will need to install `python3 -m pip install plyfile`
    # Uses plyfile to draw the point cloud in 2D top down
    def plyfile2D(filename = ""):
    
        import matplotlib.pyplot as plt
        from plyfile import PlyData, PlyElement
        
        # Read the .ply file
        plydata = PlyData.read(filename)
        
        # Extract the coordinates
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
    
        # shift the data to start at axes zeros
        x, y, z = zero_data(x, y, z)
        
        # Create a scatter plot
        plt.scatter(x, y, c=z)  # Use z for color mapping
        plt.show()
    
        return None
    
    # You will need to install `python3 -m pip install plyfile`
    # Uses plyfile to draw the point cloud in 3D top down
    def plyfile3D(filename = ""):
    
        import matplotlib.pyplot as plt
        from plyfile import PlyData, PlyElement
        
        # Read the .ply file
        plydata = PlyData.read(filename)
        
        # Extract the coordinates
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
    
        # shift the data to start at axes zeros
        x, y, z = zero_data(x, y, z)
    
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a scatter plot
        # some colormap options = viridis, plasma, inferno, magma, cividis
        ax.scatter(x, y, z, marker='.', cmap='viridis', c=z)
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_aspect('equal')
    
        plt.show()
    
        return None
    
# shifts the point cloud data to start at the zero of each axis
def zero_data(x_data, y_data, z_data):

    x_diff = 0 - x_data.min()
    x = x_data + x_diff
    y_diff = 0 - y_data.min()
    y = y_data + y_diff
    z_diff = 0 - z_data.min()
    z  = z_data + z_diff

    return x, y, z

def main():

    lowres_filename = "resources/Industrial_full_scene_2.0lowres_lessfloor.ply"
    medres_filename = "resources/Industrial_full_scene_1.0medres_lessfloor.ply"
    hires_filename = "resources/Industrial_full_scene_0.5hires_lessfloor.ply"

    # Visualize.plyfile2D(lowres_filename)
    Visualize.plyfile3D(lowres_filename)
    Visualize.trimesh3D(medres_filename)
    Visualize.open3D(hires_filename)

    return None

if __name__ == "__main__":
    main()


