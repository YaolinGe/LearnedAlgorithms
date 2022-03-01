"""
This script tests the rrt* algorithm for collision avoidance
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-22
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from scipy.spatial.distance import cdist
from scipy.stats import mvn


#== GP kernel
SIGMA = .08
LATERAL_RANGE = .7
NUGGET = .01
np.random.seed(0)
#==


MAXNUM = 500
XLIM = [0, 1]
YLIM = [0, 1]
GOAL_SAMPLE_RATE = .0001
STEP = .1
RADIUS_NEIGHBOUR = .15
DISTANCE_TOLERANCE = .18
# OBSTACLES = [[[.1, .1], [.2, .1], [.2, .2], [.1, .2]],
#              [[.4, .4], [.6, .5], [.5, .6], [.3, .4]],
#              [[.8, .8], [.95, .8], [.95, .95], [.8, .95]]]
# OBSTACLES = [[[.1, .0], [.2, .0], [.2, .5], [.1, .5]],
#              [[.0, .6], [.6, .6], [.6, 1.], [.0, 1.]],
#              [[.8, .0], [1., .0], [1., .9], [.8, .9]],
#              [[.3, .1], [.4, .1], [.4, .6], [.3, .6]],
#              [[.5, .0], [.6, .0], [.6, .4], [.5, .4]]]
# OBSTACLES = [[[1.2, 1.2], [1.4, 1.2], [1.4, 1.4], [1.2, 1.4]]]
OBSTACLES = [[]]


FIGPATH = "/Users/yaoling/OneDrive - NTNU/Self-improvements/LearnedAlgorithms/pathplanning/fig/rrt_star/"


class GP:

    def __init__(self):
        self.getGrid()
        self.getMean()
        self.getCov()
        self.getGroundTruth()

        pass

    def getGrid(self):
        x = np.linspace(XLIM[0], XLIM[1], 40)
        y = np.linspace(YLIM[0], YLIM[1], 40)
        xx, yy = np.meshgrid(x, y)
        self.grid_x = xx.reshape(-1, 1)
        self.grid_y = yy.reshape(-1, 1)
        self.grid = np.hstack((self.grid_x, self.grid_y))
        pass

    def setCoef(self):
        self.sigma = SIGMA
        self.eta = 4.5 / LATERAL_RANGE
        self.tau = NUGGET
        self.R = np.diagflat(self.tau ** 2)

    def getMean(self):
        self.mu_prior = 1 - np.exp(- ((self.grid[:, 0] - 1.) ** 2 + (self.grid[:, 1] - .5) ** 2))
        pass

    def getCov(self):
        self.setCoef()
        DistanceMatrix = cdist(self.grid, self.grid)
        self.Sigma_prior = self.sigma ** 2 * (1 + self.eta * DistanceMatrix) * np.exp(-self.eta * DistanceMatrix)
        pass

    def getGroundTruth(self):
        self.mu_truth = self.mu_prior.reshape(-1, 1) + np.linalg.cholesky(self.Sigma_prior) @ np.random.randn(len(self.mu_prior)).reshape(-1, 1)
        plt.scatter(self.grid[:, 0], self.grid[:, 1], c=self.mu_truth, cmap="Paired", vmin=.0, vmax=1)
        plt.colorbar()
        plt.show()
        pass

    def getF(self, location):
        x_loc = location.x.reshape(-1, 1)
        y_loc = location.y.reshape(-1, 1)

        DM_x = x_loc @ np.ones([1, len(self.grid_x)]) - np.ones([len(x_loc), 1]) @ self.grid_x
        DM_y = y_loc @ np.ones([1, len(self.grid_y)]) - np.ones([len(y_loc), 1]) @ self.grid_y
        DM = DM_x ** 2 + DM_y ** 2
        ind_F = np.argmin(DM, axis = 1) # interpolated vectorised indices
        F = np.zeros([1, len(self.grid_x)])
        F[ind_F] = True
        return F

    @staticmethod
    def GPupd(mu, Sigma, F, R, measurement):
        C = F @ Sigma @ F.T + R
        mu = mu + Sigma @ F.T @ np.linalg.solve(C, (measurement - F @ mu))
        Sigma = Sigma - Sigma @ F.T @ np.linalg.solve(C, F @ Sigma)
        return mu, Sigma

    @staticmethod
    def getEIBV(mu, Sigma, F, R, threshold):
        Sigma_updated = Sigma - Sigma @ F.T @ np.linalg.solve(F @ Sigma @ F.T + R, F @ Sigma)
        Variance = np.diag(Sigma_updated).reshape(-1, 1)
        EIBV = 0
        for i in range(mu.shape[0]):
            EIBV += (mvn.mvnun(-np.inf, threshold, mu[i], Variance[i])[0] -
                     mvn.mvnun(-np.inf, threshold, mu[i], Variance[i])[0] ** 2)
        return EIBV


class Location:

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        pass


class TreeNode:

    def __init__(self, location=None, parent=None, cost=None, mu=None, Sigma=None):
        self.location = location
        self.parent = parent
        self.cost = cost
        self.mu = mu
        self.Sigma = Sigma
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
                # self.nodes.append(self.ending_node)
                # break
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
    def get_cost_from_field(location):
        return 1 - np.exp(-((location.x - .5) ** 2 + (location.y - .5) ** 2) / .1)

    @staticmethod
    def get_cost_along_path(location1, location2):
        N = 10
        x = np.linspace(location1.x, location2.x, N)
        y = np.linspace(location1.y, location2.y, N)
        cost = []
        for i in range(N):
            cost.append(RRTStar.get_cost_from_field(Location(x[i], y[i])))
        cost_total = np.trapz(cost) / N
        # cost_total = np.sum(cost) / RRTStar.get_distance_between_nodes(TreeNode(location1), TreeNode(location2))
        print("cost total: ", cost_total)
        return cost_total

    @staticmethod
    def get_distance_between_nodes(node1, node2):
        dist_x = node1.location.x - node2.location.x
        dist_y = node1.location.y - node2.location.y
        dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
        return dist

    def get_next_node(self, node, location):
        node_temp = TreeNode(location)
        if RRTStar.get_distance_between_nodes(node, node_temp) <= self.config.step:
            return TreeNode(location, node)
        else:
            angle = np.math.atan2(location.y - node.location.y, location.x - node.location.x)
            x = node.location.x + self.config.step * np.cos(angle)
            y = node.location.y + self.config.step * np.sin(angle)
            location_next = Location(x, y)
        return TreeNode(location_next, node)

    def rewire_tree(self, node_current, node_nearest):
        ind_neighbour_nodes = self.get_neighbour_nodes(node_current)

        for i in range(len(ind_neighbour_nodes)):
            # print(i)
            if self.nodes[ind_neighbour_nodes[i]].cost + \
                    RRTStar.get_distance_between_nodes(self.nodes[ind_neighbour_nodes[i]], node_current) + \
                    RRTStar.get_cost_along_path(self.nodes[ind_neighbour_nodes[i]].location, node_current.location) < \
                    node_nearest.cost + \
                    RRTStar.get_distance_between_nodes(node_nearest, node_current) + \
                    RRTStar.get_cost_along_path(node_nearest.location, node_current.location):
                node_nearest = self.nodes[ind_neighbour_nodes[i]]
        node_current.cost = node_nearest.cost + \
                            RRTStar.get_distance_between_nodes(node_nearest, node_current) + \
                            RRTStar.get_cost_along_path(node_nearest.location, node_current.location)
        node_current.parent = node_nearest

        print("Distance: ", RRTStar.get_distance_between_nodes(node_nearest, node_current))
        for i in range(len(ind_neighbour_nodes)):
            # print(i)
            cost_current_neighbour = node_current.cost + \
                                     RRTStar.get_distance_between_nodes(node_current, self.nodes[ind_neighbour_nodes[i]]) + \
                                     RRTStar.get_cost_along_path(node_current.location, self.nodes[ind_neighbour_nodes[i]].location)
            if cost_current_neighbour < self.nodes[ind_neighbour_nodes[i]].cost:
                self.nodes[ind_neighbour_nodes[i]].cost = cost_current_neighbour
                self.nodes[ind_neighbour_nodes[i]].parent = node_current

        return node_current, node_nearest
        # for i in range(len(self.nodes)):
            # distance_between_nodes = RRTStar.get_distance_between_nodes(self.nodes[i], node_current)


            # if distance_between_nodes <= RADIUS_NEIGHBOUR:
            #     if self.nodes[i].cost + distance_between_nodes < \
            #             node_nearest.cost + RRTStar.get_distance_between_nodes(node_nearest, node_current):
            #         node_nearest = self.nodes[i]
        # node_current.cost = node_nearest.cost + RRTStar.get_distance_between_nodes(node_nearest, node_current)
        # node_current.parent = node_nearest
        # return node_current, node_nearest
        pass

    def get_neighbour_nodes(self, node_current):
        distance_between_nodes = []
        for i in range(len(self.nodes)):
            distance_between_nodes.append(RRTStar.get_distance_between_nodes(self.nodes[i], node_current))
        print(distance_between_nodes)
        ind_neighbours = np.where(np.array(distance_between_nodes) <= RADIUS_NEIGHBOUR)[0]
        print(ind_neighbours)
        return ind_neighbours

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
            if i > 0:
                obstacle = np.append(self.obstacles[i], self.obstacles[i][0, :].reshape(1, -1), axis=0)
                plt.plot(obstacle[:, 0], obstacle[:, 1], 'r-.')
            else:
                pass

        for node in self.nodes:
            if node.parent is not None:
                plt.plot([node.location.x, node.parent.location.x],
                         [node.location.y, node.parent.location.y], "-g")
        path = np.array(self.path)
        plt.plot(path[:, 0], path[:, 1], "-r")
        plt.plot(self.config.starting_location.x, self.config.starting_location.y, 'k*', ms=10)
        plt.plot(self.config.ending_location.x, self.config.ending_location.y, 'g*', ms=10)

        plt.grid()
        plt.title("rrt_star")
        plt.show()


if __name__ == "__main__":
    starting_loc = Location(0., 0.)
    ending_loc = Location(.0, 1.)
    rrtconfig = RRTConfig(starting_location=starting_loc, ending_location=ending_loc, goal_sample_rate=GOAL_SAMPLE_RATE,
                          step=STEP)
    rrt = RRTStar(rrtconfig)
    rrt.set_obstacles()
    rrt.expand_trees()
    rrt.get_shortest_path()
    rrt.plot_tree()
    pass





