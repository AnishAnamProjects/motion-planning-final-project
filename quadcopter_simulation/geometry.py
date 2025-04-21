import numpy as np
from typing import List, Tuple

class Polygon3D:
    def __init__(self, vertices: np.ndarray):
        """
        Initialize a 3D polygon with vertices
        
        Args:
            vertices: numpy array of shape (n, 3) containing vertex coordinates
        """
        self.vertices = vertices
        self.center = np.mean(vertices, axis=0)
        # Calculate bounding box for sweep algorithm
        self.bounds = {
            'x_min': np.min(vertices[:, 0]),
            'x_max': np.max(vertices[:, 0]),
            'y_min': np.min(vertices[:, 1]),
            'y_max': np.max(vertices[:, 1]),
            'z_min': np.min(vertices[:, 2]),
            'z_max': np.max(vertices[:, 2])
        }
        
    def get_faces(self) -> List[np.ndarray]:
        """Returns list of polygon faces as triangles"""
        # For simplicity, we'll triangulate the polygon
        # Assuming convex polygon, we can triangulate from first vertex
        faces = []
        v0 = self.vertices[0]
        for i in range(1, len(self.vertices) - 1):
            face = np.array([v0, self.vertices[i], self.vertices[i + 1]])
            faces.append(face)
        return faces
    
    def point_inside_bounds(self, point: np.ndarray) -> bool:
        """Check if point is inside bounding box"""
        return (self.bounds['x_min'] <= point[0] <= self.bounds['x_max'] and
                self.bounds['y_min'] <= point[1] <= self.bounds['y_max'] and
                self.bounds['z_min'] <= point[2] <= self.bounds['z_max'])

def point_to_polygon_distance(point: np.ndarray, polygon: Polygon3D) -> float:
    """
    Calculate minimum distance from point to polygon
    
    Args:
        point: numpy array [x, y, z]
        polygon: Polygon3D object
        
    Returns:
        float: Minimum distance from point to polygon
    """
    # First check bounding box for early exit
    if not polygon.point_inside_bounds(point):
        # If point is outside bounding box, return distance to closest vertex
        distances = np.linalg.norm(polygon.vertices - point, axis=1)
        return np.min(distances)
    
    # If point is inside bounding box, check distance to each face
    min_distance = float('inf')
    for face in polygon.get_faces():
        # Calculate distance from point to triangle face
        # Using method from Real-Time Collision Detection by Christer Ericson
        
        # Get triangle vertices
        a, b, c = face
        
        # Calculate triangle normal
        normal = np.cross(b - a, c - a)
        normal = normal / np.linalg.norm(normal)
        
        # Calculate distance from point to triangle plane
        plane_dist = abs(np.dot(point - a, normal))
        
        min_distance = min(min_distance, plane_dist)
    
    return min_distance

def create_box(center: np.ndarray, size: float) -> Polygon3D:
    """
    Create a 3D box centered at given point
    
    Args:
        center: numpy array [x, y, z]
        size: length of box sides
        
    Returns:
        Polygon3D: Box shaped polygon
    """
    half = size / 2
    vertices = np.array([
        [center[0] - half, center[1] - half, center[2] - half],  # bottom face
        [center[0] + half, center[1] - half, center[2] - half],
        [center[0] + half, center[1] + half, center[2] - half],
        [center[0] - half, center[1] + half, center[2] - half],
        [center[0] - half, center[1] - half, center[2] + half],  # top face
        [center[0] + half, center[1] - half, center[2] + half],
        [center[0] + half, center[1] + half, center[2] + half],
        [center[0] - half, center[1] + half, center[2] + half],
    ])
    return Polygon3D(vertices) 