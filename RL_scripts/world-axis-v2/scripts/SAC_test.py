import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time
import random
from scipy.spatial.distance import euclidean
import threading
import queue

class RRTStar3D:
    def __init__(self, start, goal, bounds, obstacles, step_size=0.5, max_iter=1000, goal_radius=0.5):
        self.start = np.array(start, dtype=np.float64)
        self.goal = np.array(goal, dtype=np.float64)
        self.bounds = bounds  # (min_x, max_x, min_y, max_y, min_z, max_z)
        self.obstacles = obstacles  # list of (center, size) tuples
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_radius = goal_radius
        self.gamma = 10.0  # RRT* parameter
        
        # Tree structure
        self.vertices = [self.start.copy()]
        self.edges = []  # list of (parent_idx, child_idx)
        self.costs = [0.0]  # cost from start to each vertex
        
        # Animation data
        self.path = []
        self.exploration_lines = []
        self.path_lines = []
        self.drone_position = self.start.copy()
        self.current_path_idx = 0
        self.navigation_complete = False
        self.planning_complete = False
        
    def is_collision_free(self, point):
        """Check if a point is collision-free"""
        x, y, z = point
        min_x, max_x, min_y, max_y, min_z, max_z = self.bounds
        
        # Check bounds
        if not (min_x <= x <= max_x and min_y <= y <= max_y and min_z <= z <= max_z):
            return False
            
        # Check obstacles
        for obs_center, obs_size in self.obstacles:
            if euclidean(point, obs_center) < obs_size/2 + 0.2:  # 0.2 is drone radius
                return False
        return True
    
    def is_segment_collision_free(self, p1, p2):
        """Check if a line segment is collision-free"""
        dist = euclidean(p1, p2)
        if dist < 0.001:  # Prevent division by zero
            return self.is_collision_free(p1)
        
        steps = max(int(dist / 0.1), 1)  # 0.1 step size for collision checking, minimum 1 step
        for i in range(steps + 1):
            t = i / steps
            point = p1 * (1 - t) + p2 * t
            if not self.is_collision_free(point):
                return False
        return True
    
    def sample_free_space(self):
        """Sample a random point in free space"""
        if random.random() < 0.05:  # 5% chance to sample goal
            return self.goal.copy()
        
        while True:
            x = random.uniform(self.bounds[0], self.bounds[1])
            y = random.uniform(self.bounds[2], self.bounds[3])
            z = random.uniform(self.bounds[4], self.bounds[5])
            point = np.array([x, y, z], dtype=np.float64)
            if self.is_collision_free(point):
                return point
    
    def nearest_vertex(self, point):
        """Find the nearest vertex in the tree"""
        min_dist = float('inf')
        nearest_idx = 0
        for i, vertex in enumerate(self.vertices):
            dist = euclidean(point, vertex)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx
    
    def new_config(self, nearest_idx, target):
        """Generate a new configuration in the direction of target"""
        nearest_point = self.vertices[nearest_idx]
        direction = target - nearest_point
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm <= self.step_size:
            return target.copy()
        else:
            return nearest_point + (direction / direction_norm) * self.step_size
    
    def find_near_vertices(self, new_vertex):
        """Find vertices near the new vertex for rewiring"""
        n = len(self.vertices)
        radius = min(
            self.gamma * (np.log(n) / n) ** (1.0/3.0) if n > 1 else self.step_size * 3,
            self.step_size * 3
        )
        
        near_indices = []
        for i, vertex in enumerate(self.vertices):
            if euclidean(new_vertex, vertex) < radius:
                near_indices.append(i)
        return near_indices
    
    def plan_path(self):
        """Execute RRT* algorithm"""
        print("Starting RRT* path planning...")
        
        for iteration in range(self.max_iter):
            if self.navigation_complete or self.planning_complete:
                break
                
            # Sample random point
            random_point = self.sample_free_space()
            
            # Find nearest vertex
            nearest_idx = self.nearest_vertex(random_point)
            
            # Generate new configuration
            new_vertex = self.new_config(nearest_idx, random_point)
            
            # Check collision
            if not self.is_segment_collision_free(self.vertices[nearest_idx], new_vertex):
                continue
            
            # Find best parent
            near_indices = self.find_near_vertices(new_vertex)
            best_parent_idx = nearest_idx
            min_cost = self.costs[nearest_idx] + euclidean(self.vertices[nearest_idx], new_vertex)
            
            for near_idx in near_indices:
                cost = self.costs[near_idx] + euclidean(self.vertices[near_idx], new_vertex)
                if cost < min_cost and self.is_segment_collision_free(self.vertices[near_idx], new_vertex):
                    min_cost = cost
                    best_parent_idx = near_idx
            
            # Add new vertex
            new_vertex_idx = len(self.vertices)
            self.vertices.append(new_vertex.copy())
            self.edges.append((best_parent_idx, new_vertex_idx))
            self.costs.append(min_cost)
            
            # Store exploration line
            self.exploration_lines.append((self.vertices[best_parent_idx].copy(), new_vertex.copy()))
            
            # Rewire
            for near_idx in near_indices:
                if near_idx == best_parent_idx:
                    continue
                
                new_cost = min_cost + euclidean(new_vertex, self.vertices[near_idx])
                if new_cost < self.costs[near_idx] and self.is_segment_collision_free(new_vertex, self.vertices[near_idx]):
                    # Update parent
                    old_edge_idx = -1
                    for i, (parent, child) in enumerate(self.edges):
                        if child == near_idx:
                            old_edge_idx = i
                            break
                    
                    if old_edge_idx != -1:
                        # Remove old edge
                        self.edges.pop(old_edge_idx)
                    
                    # Add new edge
                    self.edges.append((new_vertex_idx, near_idx))
                    self.costs[near_idx] = new_cost
            
            # Check if goal is reached
            if euclidean(new_vertex, self.goal) < self.goal_radius:
                print(f"Goal reached at iteration {iteration}")
                self.extract_path()
                self.planning_complete = True
                break
        
        print("Path planning completed")
    
    def extract_path(self):
        """Extract the path from start to goal"""
        # Find vertices within goal radius
        goal_vertices = []
        for i, vertex in enumerate(self.vertices):
            if euclidean(vertex, self.goal) < self.goal_radius:
                goal_vertices.append(i)
        
        if not goal_vertices:
            print("No vertex reached the goal")
            return
        
        # Find the vertex with minimum cost among goal vertices
        min_cost_idx = min(goal_vertices, key=lambda i: self.costs[i])
        
        # Backtrack to find path
        path = [self.vertices[min_cost_idx].copy()]
        current_idx = min_cost_idx
        
        while current_idx != 0:  # Not the start vertex
            parent_idx = None
            for parent, child in self.edges:
                if child == current_idx:
                    parent_idx = parent
                    break
            
            if parent_idx is not None:
                path.append(self.vertices[parent_idx].copy())
                current_idx = parent_idx
            else:
                break
        
        path.reverse()
        self.path = path
        print(f"Path extracted with {len(path)} waypoints")
        
        # Store path lines
        for i in range(1, len(path)):
            self.path_lines.append((path[i-1].copy(), path[i].copy()))
    
    def navigate(self, speed=0.1):
        """Navigate the drone along the path"""
        if not self.path or self.navigation_complete:
            return
            
        if self.current_path_idx >= len(self.path):
            self.navigation_complete = True
            print("Navigation completed!")
            return
        
        target = self.path[self.current_path_idx]
        direction = target - self.drone_position
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:  # Reached current waypoint
            self.current_path_idx += 1
            if self.current_path_idx >= len(self.path):
                self.navigation_complete = True
                print("Navigation completed!")
        else:
            # Move towards target
            move_direction = direction / distance
            self.drone_position += move_direction * min(speed, distance)

class DroneNavigationSimulator:
    def __init__(self):
        self.rrt_star = None
        self.fig = None
        self.ax = None
        self.line_drone = None
        self.scatter_path = None
        self.scatter_tree = None
        self.scatter_obstacles = None
        self.animation_running = True
        self.planning_thread = None
        
    def setup_scene(self):
        """Setup the 3D scene"""
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set bounds
        bounds = (-10, 10, -5, 5, -10, 10)
        self.ax.set_xlim(bounds[0], bounds[1])
        self.ax.set_ylim(bounds[2], bounds[3])
        self.ax.set_zlim(bounds[4], bounds[5])
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D RRT* Drone Navigation Simulation')
        
        # Create RRT* planner
        start = np.array([-8, 0, -8], dtype=np.float64)
        goal = np.array([8, 0, 8], dtype=np.float64)
        
        # Generate random obstacles
        obstacles = []
        for _ in range(15):
            center = np.array([
                random.uniform(-7, 7),
                random.uniform(-3, 3),
                random.uniform(-7, 7)
            ], dtype=np.float64)
            size = random.uniform(1.0, 2.0)
            obstacles.append((center, size))
        
        self.rrt_star = RRTStar3D(start, goal, bounds, obstacles, 
                                 step_size=0.5, max_iter=1000, goal_radius=0.5)
        
        # Plot start and goal
        self.ax.scatter(*start, color='red', s=100, label='Start', depthshade=False)
        self.ax.scatter(*goal, color='green', s=100, label='Goal', depthshade=False)
        
        # Plot obstacles
        for center, size in obstacles:
            self.plot_cube(center, size)
        
        plt.legend()
    
    def plot_cube(self, center, size):
        """Plot a cube obstacle"""
        r = size / 2
        x, y, z = center
        # Define cube vertices
        vertices = np.array([
            [x-r, y-r, z-r],
            [x+r, y-r, z-r],
            [x+r, y+r, z-r],
            [x-r, y+r, z-r],
            [x-r, y-r, z+r],
            [x+r, y-r, z+r],
            [x+r, y+r, z+r],
            [x-r, y+r, z+r]
        ])
        
        # Define cube faces
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # connecting edges
        ]
        
        for edge in edges:
            points = vertices[edge]
            self.ax.plot3D(*points.T, color='red', alpha=0.6)
    
    def animate(self, frame):
        """Animation update function"""
        if frame == 0:
            # Start planning in a separate thread
            self.planning_thread = threading.Thread(target=self.rrt_star.plan_path)
            self.planning_thread.daemon = True
            self.planning_thread.start()
        
        # Update drone position
        self.rrt_star.navigate(speed=0.1)
        
        # Clear previous drone position
        if self.line_drone:
            self.line_drone.remove()
        
        # Plot current drone position
        pos = self.rrt_star.drone_position
        self.line_drone, = self.ax.plot([pos[0]], [pos[1]], [pos[2]], 
                                       color='white', marker='o', markersize=8, 
                                       markeredgecolor='black', markeredgewidth=1)
        
        # Plot exploration tree (sample lines)
        if frame % 5 == 0:  # Update tree visualization every 5 frames
            # Remove previous tree lines except start, goal, and axis
            lines_to_remove = []
            for line in self.ax.lines:
                # Check if it's not the start, goal, or drone position lines
                if len(line.get_xdata()) == 1 and line.get_color() in ['red', 'green', 'white']:
                    continue
                lines_to_remove.append(line)
            
            for line in lines_to_remove:
                line.remove()
            
            # Plot exploration lines (limit to last 50 for performance)
            for p1, p2 in self.rrt_star.exploration_lines[-50:]:
                self.ax.plot3D(*np.array([p1, p2]).T, color='yellow', alpha=0.3, linewidth=0.8)
            
            # Plot path lines
            for p1, p2 in self.rrt_star.path_lines:
                self.ax.plot3D(*np.array([p1, p2]).T, color='blue', linewidth=2)
        
        # Check if navigation is complete
        if self.rrt_star.navigation_complete:
            print("Navigation completed successfully!")
            self.animation_running = False
        
        return [self.line_drone]
    
    def run_simulation(self):
        """Run the simulation"""
        self.setup_scene()
        
        # Create animation
        ani = animation.FuncAnimation(
            self.fig, 
            self.animate, 
            frames=range(2000), 
            interval=50, 
            blit=False,
            repeat=False
        )
        
        plt.show()

def main():
    print("Starting 3D RRT* Drone Navigation Simulation...")
    print("Red cubes: obstacles")
    print("Red point: start position")
    print("Green point: goal position")
    print("White circle: drone")
    print("Yellow lines: exploration tree")
    print("Blue lines: final path")
    print("\nStarting simulation...")
    
    simulator = DroneNavigationSimulator()
    simulator.run_simulation()

if __name__ == "__main__":
    main()



