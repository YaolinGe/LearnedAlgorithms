"""
This script tests the probabilistic road map construction for path planning
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-24
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from shapely.geometry import Point, Polygon, LineString

MAXNUM = 100
XLIM = [0, 1]
YLIM = [0, 1]
GOAL_SAMPLE_RATE = .01
STEP = .1
RADIUS_NEIGHBOUR = .15
DISTANCE_TOLERANCE = .11
# OBSTACLES = [[[.1, .1], [.2, .1], [.2, .2], [.1, .2]],
#              [[.4, .4], [.6, .5], [.5, .6], [.3, .4]],
#              [[.8, .8], [.95, .8], [.95, .95], [.8, .95]]]
OBSTACLES = [[[.1, .0], [.2, .0], [.2, .5], [.1, .5]],
             [[.0, .6], [.6, .6], [.6, 1.], [.0, 1.]],
             [[.8, .0], [1., .0], [1., .9], [.8, .9]],
             [[.3, .1], [.4, .1], [.4, .6], [.3, .6]],
             [[.5, .0], [.6, .0], [.6, .4], [.5, .4]]]

FIGPATH = "/Users/yaoling/OneDrive - NTNU/Self-improvements/LearnedAlgorithms/pathplanning/fig/prm/"


class Location:

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        pass


class Node:

    def __init__(self, location=None, cost=None):
        self.location = location
        self.cost = cost
        self.parent = None
        self.neighbours = []
        pass


class PRMConfig:

    def __init__(self, starting_location=None, ending_location=None, num_nodes=None, num_neighbours=None):
        self.starting_location = starting_location
        self.ending_location = ending_location
        self.num_nodes = num_nodes
        self.num_neighbours = num_neighbours
        pass


class PRM:

    nodes = []
    obstacles = np.array(OBSTACLES)
    polygon_obstacles = []

    def __init__(self, config=None):
        self.config = config
        self.path = []
        self.starting_node = Node(self.config.starting_location)
        self.ending_node = Node(self.config.ending_location)
        pass

    def get_all_random_nodes(self):
        self.nodes.append(self.starting_node)
        counter_nodes = 0
        while counter_nodes < self.config.num_nodes:
            new_location = self.get_new_location()
            if not self.inRedZone(new_location):
                self.nodes.append(Node(new_location))
                counter_nodes += 1
        self.nodes.append(self.ending_node)

    def get_road_maps(self):
        for i in range(len(self.nodes)):
            dist = []
            node_now = self.nodes[i]
            for j in range(len(self.nodes)):
                node_next = self.nodes[j]
                dist.append(PRM.get_distance_between_nodes(node_now, node_next))
            ind_sort = np.argsort(dist)
            # print(ind_sort[:self.config.num_neighbours])
            for k in range(self.config.num_neighbours):
                node_neighbour = self.nodes[ind_sort[:self.config.num_neighbours][k]]
                if not self.iscollided(node_now, node_neighbour):
                    node_now.neighbours.append(node_neighbour)
            # print(node_now.neighbours)
            # print(self.nodes[ind_sort[:self.config.num_neighbours][0]])
            # neighbouring_nodes = self.nodes[ind_sort[:self.config.num_neighbours]]
            # print(neighbouring_nodes)
            # print(i)


    def expand_trees(self):
        self.nodes.append(self.starting_node)
        for i in range(MAXNUM):
            # print(i)
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

    def inRedZone(self, location):
        point = Point(location.x, location.y)
        collision = False
        for i in range(len(self.polygon_obstacles)):
            if self.polygon_obstacles[i].contains(point):
                collision = True
        return collision

    @staticmethod
    def get_distance_between_nodes(node1, node2):
        dist_x = node1.location.x - node2.location.x
        dist_y = node1.location.y - node2.location.y
        dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
        return dist

    def set_obstacles(self):
        for i in range(self.obstacles.shape[0]):
            self.polygon_obstacles.append(Polygon(list(map(tuple, self.obstacles[i]))))

    def iscollided(self, node1, node2):
        line = LineString([(node1.location.x, node1.location.y),
                           (node2.location.x, node2.location.y)])
        collision = False
        for i in range(len(self.polygon_obstacles)):
            if self.polygon_obstacles[i].intersects(line):
                collision = True
        return collision

    def get_shortest_path_using_dijkstra(self):
        self.unvisited_nodes = []
        for node in self.nodes:
            node.cost = np.inf
            node.parent = None
            self.unvisited_nodes.append(node)
            pass

        current_node = self.unvisited_nodes[0]
        current_node.cost = 0
        pointer_node = current_node

        while self.unvisited_nodes:
            ind_min_cost = PRM.get_ind_min_cost(self.unvisited_nodes)
            current_node = self.unvisited_nodes[ind_min_cost]

            for neighbour_node in current_node.neighbours:
                if neighbour_node in self.unvisited_nodes:
                    cost = current_node.cost + PRM.get_distance_between_nodes(current_node, neighbour_node)
                    if cost < neighbour_node.cost:
                        neighbour_node.cost = cost
                        neighbour_node.parent = current_node
            pointer_node = current_node
            self.unvisited_nodes.pop(ind_min_cost)

        self.path.append([pointer_node.location.x, pointer_node.location.y])

        while pointer_node.parent is not None:
            node = pointer_node.parent
            self.path.append([node.location.x, node.location.y])
            pointer_node = node

        self.path.append([self.starting_node.location.x, self.starting_node.location.y])
        pass

    def get_shortest_path_using_astar(self):
        current_node = self.nodes[0]
        current_node.parent = current_node
        self.get_total_cost(current_node)
        self.counter = 0
        self.find_next_nodes_with_min_cost(current_node)
        pass

    def find_next_nodes_with_min_cost(self, node):
        cost = []
        #TODO: Needs to fix AStar
        if node != self.ending_node and node.neighbours is not None:
            self.counter += 1
            if self.counter>10:
                return None
            for neighbour_node in node.neighbours:

                neighbour_node.parent = node
                self.get_total_cost(neighbour_node)
                cost.append(neighbour_node.cost)
            # print(cost)
            # print(cost.index(min(cost)))
            # print(node.neighbours)
            return self.find_next_nodes_with_min_cost(node.neighbours[cost.index(min(cost))])
        else:
            return node
        pass

    def get_total_cost(self, node):
        node.heuristic_cost = self.get_manhattan_distance(node)
        node.distance_cost = prm.get_distance_between_nodes(node, node.parent)
        node.cost = node.heuristic_cost + node.distance_cost

    def get_manhattan_distance(self, node):
        manhattan_distance = np.abs(node.location.x - self.ending_node.location.x) + \
                             np.abs(node.location.y - self.ending_node.location.y)
        return manhattan_distance

    @staticmethod
    def get_ind_min_cost(nodes):
        cost = []
        for node in nodes:
            cost.append(node.cost)
        return cost.index(min(cost))

    def plot_prm(self):
        plt.clf()
        for i in range(self.obstacles.shape[0]):
            obstacle = np.append(self.obstacles[i], self.obstacles[i][0, :].reshape(1, -1), axis=0)
            plt.plot(obstacle[:, 0], obstacle[:, 1], 'r-.')

        for node in self.nodes:
            if node.neighbours is not None:
                for i in range(len(node.neighbours)):
                    plt.plot([node.location.x, node.neighbours[i].location.x],
                             [node.location.y, node.neighbours[i].location.y], "-g")
        path = np.array(self.path)
        plt.plot(path[:, 0], path[:, 1], "-r")
        plt.plot(self.config.starting_location.x, self.config.starting_location.y, 'k*', ms=10)
        plt.plot(self.config.ending_location.x, self.config.ending_location.y, 'b*', ms=10)
        plt.grid()
        plt.title("prm")
        plt.show()


if __name__ == "__main__":
    starting_loc = Location(0, 0)
    ending_loc = Location(1, 1)
    rrtconfig = PRMConfig(starting_location=starting_loc, ending_location=ending_loc, num_nodes=100, num_neighbours=10)
    prm = PRM(rrtconfig)
    prm.set_obstacles()
    prm.get_all_random_nodes()
    prm.get_road_maps()
    prm.get_shortest_path_using_dijkstra()
    # prm.get_shortest_path_using_astar()
    prm.plot_prm()
    pass




