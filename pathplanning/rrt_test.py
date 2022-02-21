"""
This script tests the rrt algorithm for collision avoidance
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-17
"""

import numpy as np
import matplotlib.pyplot as plt
MAXNUM = 3000
XLIM = [0, 1]
YLIM = [0, 1]
GOAL_SAMPLE_RATE = .01
STEP = .01
DISTANCE_TOLERANCE = .05


class Location:

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        pass


class TreeNode:

    def __init__(self, location=None, parent=None):
        self.location = location
        self.parent = parent
        pass


class RRTConfig:

    def __init__(self, starting_location=None, ending_location=None, goal_sample_rate=None, step=None):
        self.starting_location = starting_location
        self.ending_location = ending_location
        self.goal_sample_rate = goal_sample_rate
        self.step = step
        pass


class RRT:

    nodes = []

    def __init__(self, config=None):
        self.config = config
        self.path = []
        self.starting_node = TreeNode(self.config.starting_location, None)
        self.ending_node = TreeNode(self.config.ending_location, None)
        pass

    def expand_trees(self):
        self.nodes.append(self.starting_node)
        for i in range(MAXNUM):
            print(i)
            if np.random.rand() <= self.config.goal_sample_rate:
                new_location = self.config.ending_location
            else:
                new_location = self.get_new_location()

            nearest_node = self.get_nearest_node(self.nodes, new_location)
            next_node = self.get_next_node(nearest_node, new_location)
            if self.isarrived(next_node):
                self.ending_node.parent = next_node
                self.nodes.append(self.ending_node)
                break
            else:
                self.nodes.append(next_node)
            pass
        pass

    @staticmethod
    def get_new_location():
        x = np.random.uniform(XLIM[0], XLIM[1])
        y = np.random.uniform(YLIM[0], YLIM[1])
        location = Location(x, y)
        return location

    def get_nearest_node(self, nodes, location):
        dist = []
        node_new = TreeNode(location)
        for node in nodes:
            dist.append(self.get_distance_between_nodes(node, node_new))
        return self.nodes[dist.index(min(dist))]

    @staticmethod
    def get_distance_between_nodes(node1, node2):
        dist_x = node1.location.x - node2.location.x
        dist_y = node1.location.y - node2.location.y
        dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
        return dist

    def get_next_node(self, node, location):
        angle = np.math.atan2(location.y - node.location.y, location.x - node.location.x)
        x = node.location.x + self.config.step * np.cos(angle)
        y = node.location.y + self.config.step * np.sin(angle)
        location_next = Location(x, y)
        return TreeNode(location_next, node)

    def isarrived(self, node):
        dist = self.get_distance_between_nodes(self.ending_node, node)
        if dist < DISTANCE_TOLERANCE:
            return True
        else:
            return False

    def iscollided(self):
        #TODO: use shapely to add intersect collision avoidance
        """
        - Check if point is inside polygon
        - check line intersect

        """
        pass

    def get_shortest_path(self):
        self.path.append([self.ending_node.location.x, self.ending_node.location.y])
        pointer_node = self.ending_node
        while pointer_node.parent is not None:

            node = pointer_node.parent
            self.path.append([node.location.x, node.location.y])
            pointer_node = node

        self.path.append([self.starting_node.location.x, self.starting_node.location.y])
        pass

    def plot_tree(self):

        plt.clf()
        for node in self.nodes:
            if node.parent is not None:
                plt.plot([node.location.x, node.parent.location.x],
                         [node.location.y, node.parent.location.y], "-g")
        path = np.array(self.path)
        plt.plot(path[:, 0], path[:, 1], "-r")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    starting_loc = Location(0, 0)
    ending_loc = Location(.8, .5)
    rrtconfig = RRTConfig(starting_location=starting_loc, ending_location=ending_loc, goal_sample_rate=GOAL_SAMPLE_RATE,
                          step=STEP)
    rrt = RRT(rrtconfig)
    rrt.expand_trees()
    rrt.get_shortest_path()
    rrt.plot_tree()

    pass

#%%
# import Point, Polygon
from sympy import Point, Polygon

# creating points using Point()
p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
p5, p6, p7 = map(Point, [(3, 2), (1, -1), (0, 2)])

# creating polygons using Polygon()
poly1 = Polygon(p1, p2, p3, p4)
poly2 = Polygon(p5, p6, p7)

# using intersection()
isIntersection = poly1.intersection(poly2)

print(isIntersection)
#%%
p1, p2, p3, p4 = map(Point, [(0, 0), (0, 1), (1, 1), (1, 0)])
P1 = Polygon(p1, p2, p3, p4)
p1, p2, p3, p4 = map(Point, [(2, 2), (3, 2), (3, 3), (2, 3)])
P2 = Polygon(p1, p2, p3, p4)
isIntersection = P1.intersection(P2)
print(isIntersection)

#%%

import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry

# poly = shapely.geometry.Polygon()
circle = shapely.geometry.Point(5.0, 0.0).buffer(10.0)
clip_poly = shapely.geometry.Polygon([[-9.5, -2], [2, 2], [3, 4], [-1, 3]])
clipped_shape = circle.difference(clip_poly)

line = shapely.geometry.LineString([[-10, -5], [15, 5]])
line2 = shapely.geometry.LineString([[-10, -5], [-5, 0], [2, 3]])
print(line.intersects(clip_poly))

print(line2.intersects(clip_poly))


# print 'Blue line intersects clipped shape:', line.intersects(clipped_shape)
# print 'Green line intersects clipped shape:', line2.intersects(clipped_shape)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.plot(*np.array(line).T, color='blue', linewidth=3, solid_capstyle='round')
# ax.plot(*np.array(line2).T, color='green', linewidth=3, solid_capstyle='round')
# ax.add_patch(descartes.PolygonPatch(clipped_shape, fc='blue', alpha=0.5))
# ax.axis('equal')
#
# plt.show()




