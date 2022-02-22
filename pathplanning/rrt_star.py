"""
This script tests the rrt* algorithm for collision avoidance
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-22
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString

MAXNUM = 1000
XLIM = [0, 1]
YLIM = [0, 1]
GOAL_SAMPLE_RATE = .01
STEP = .05
RADIUS_NEIGHBOUR = .1
DISTANCE_TOLERANCE = .05
OBSTACLES = [[[.1, .1], [.2, .1], [.2, .2], [.1, .2]],
             [[.4, .4], [.6, .5], [.5, .6], [.3, .4]],
             [[.8, .8], [.95, .8], [.95, .95], [.8, .95]]]

FIGPATH = "/Users/yaoling/OneDrive - NTNU/Self-improvements/LearnedAlgorithms/pathplanning/fig/rrt_star/"


class Location:

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        pass


class TreeNode:

    def __init__(self, location=None, parent=None, cost=None):
        self.location = location
        self.parent = parent
        self.cost = cost
        pass


class RRTConfig:

    def __init__(self, starting_location=None, ending_location=None, goal_sample_rate=None, step=None):
        self.starting_location = starting_location
        self.ending_location = ending_location
        self.goal_sample_rate = goal_sample_rate
        self.step = step
        pass


class RRTStar:

    nodes = []
    obstacles = np.array(OBSTACLES)
    polygon_obstacles = []

    def __init__(self, config=None):
        self.config = config
        self.path = []
        self.starting_node = TreeNode(self.config.starting_location, None, 0)
        self.ending_node = TreeNode(self.config.ending_location, None, 0)
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
            next_node, nearest_node = self.rewire_tree(next_node, nearest_node)

            if self.iscollided(next_node):
                continue

            if self.isarrived(next_node):
                self.ending_node.parent = next_node
                self.nodes.append(self.ending_node)
                break
            else:
                self.nodes.append(next_node)
            pass

            # plt.clf()
            # for j in range(self.obstacles.shape[0]):
            #     obstacle = np.append(self.obstacles[j], self.obstacles[j][0, :].reshape(1, -1), axis=0)
            #     plt.plot(obstacle[:, 0], obstacle[:, 1], 'r-.')
            #
            # for node in self.nodes:
            #     if node.parent is not None:
            #         plt.plot([node.location.x, node.parent.location.x],
            #                  [node.location.y, node.parent.location.y], "-g")
            # # path = np.array(self.path)
            # # plt.plot(path[:, 0], path[:, 1], "-r")
            # plt.plot(self.config.starting_location.x, self.config.starting_location.y, 'k*', ms=10)
            # plt.plot(self.config.ending_location.x, self.config.ending_location.y, 'g*', ms=10)
            # plt.grid()
            # plt.savefig(FIGPATH + "P_{:05d}.png".format(i))
            # plt.show()
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
        return nodes[dist.index(min(dist))]

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

    def rewire_tree(self, node_new, node_nearest):
        for i in range(len(self.nodes)):
            if RRTStar.get_distance_between_nodes(self.nodes[i], node_new) <= RADIUS_NEIGHBOUR:
                if self.nodes[i].cost + RRTStar.get_distance_between_nodes(self.nodes[i], node_new) < \
                        node_nearest.cost + RRTStar.get_distance_between_nodes(node_nearest, node_new):
                    node_nearest = self.nodes[i]
        node_new.cost = node_nearest.cost + RRTStar.get_distance_between_nodes(node_nearest, node_new)
        node_new.parent = node_nearest
        return node_new, node_nearest

    def isarrived(self, node):
        dist = self.get_distance_between_nodes(self.ending_node, node)
        if dist < DISTANCE_TOLERANCE:
            return True
        else:
            return False

    def set_obstacles(self):
        for i in range(self.obstacles.shape[0]):
            self.polygon_obstacles.append(Polygon(list(map(tuple, self.obstacles[i]))))

    def iscollided(self, node):
        point = Point(node.location.x, node.location.y)
        line = LineString([(node.parent.location.x, node.parent.location.y),
                           (node.location.x, node.location.y)])
        collision = False
        for i in range(len(self.polygon_obstacles)):
            if self.polygon_obstacles[i].contains(point) or self.polygon_obstacles[i].intersects(line):
                # print("Collision detected")
                collision = True
        return collision

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
        for i in range(self.obstacles.shape[0]):
            obstacle = np.append(self.obstacles[i], self.obstacles[i][0, :].reshape(1, -1), axis=0)
            plt.plot(obstacle[:, 0], obstacle[:, 1], 'r-.')

        for node in self.nodes:
            if node.parent is not None:
                plt.plot([node.location.x, node.parent.location.x],
                         [node.location.y, node.parent.location.y], "-g")
        path = np.array(self.path)
        plt.plot(path[:, 0], path[:, 1], "-r")
        plt.plot(self.config.starting_location.x, self.config.starting_location.y, 'k*', ms=10)
        plt.plot(self.config.ending_location.x, self.config.ending_location.y, 'g*', ms=10)
        plt.grid()
        plt.show()


if __name__ == "__main__":
    starting_loc = Location(0, 0)
    ending_loc = Location(1, 1)
    rrtconfig = RRTConfig(starting_location=starting_loc, ending_location=ending_loc, goal_sample_rate=GOAL_SAMPLE_RATE,
                          step=STEP)
    rrt = RRTStar(rrtconfig)
    rrt.set_obstacles()
    rrt.expand_trees()
    rrt.get_shortest_path()
    rrt.plot_tree()
    pass





