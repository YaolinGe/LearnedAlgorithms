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
import time
from scipy.interpolate import griddata
from matplotlib.gridspec import GridSpec
from matplotlib.cm import get_cmap


#== GP kernel
SIGMA = .15
LATERAL_RANGE = .7
NUGGET = .01
np.random.seed(0)
THRESHOLD = .6
#==


NUM_STEPS = 50
BUDGET = 7

MAXNUM = 300
XLIM = [0, 1]
YLIM = [0, 1]
NX = 25
NY = 25
GOAL_SAMPLE_RATE = .01
STEP = .15
RADIUS_NEIGHBOUR = .2
DISTANCE_TOLERANCE = .05
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


def plotf_vector(grid, values, title, alpha=None, cmap="Paired"):
    x = grid[:, 0]
    y = grid[:, 1]
    nx = 100
    ny = 100

    xmin, ymin = map(np.amin, [x, y])
    xmax, ymax = map(np.amax, [x, y])

    xv = np.linspace(xmin, xmax, nx)
    yv = np.linspace(ymin, ymax, ny)
    grid_x, grid_y = np.meshgrid(xv, yv)

    grid_values = griddata(grid, values, (grid_x, grid_y))
    # plt.figure()
    plt.scatter(grid_x, grid_y, c=grid_values, cmap=cmap, alpha=alpha)
    plt.xlim(XLIM)
    plt.ylim(YLIM)
    plt.colorbar()
    plt.title(title)
    # plt.show()


def plotf_budget_radar(centre, radius):
    plt.plot(centre[0], centre[1], 'yp')
    radar = plt.Circle((centre[0], centre[1]), radius, color='b', fill=False)
    plt.gca().add_patch(radar)
    if radius >= 1:
        plt.xlim([radius * XLIM[0], radius * XLIM[1]])
        plt.ylim([radius * YLIM[0], radius * YLIM[1]])


def plotf_trajectory(trajectory):
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'k.-')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'k-')


def plotf_matrix(values, title):
    # grid_values = griddata(grid, values, (grid_x, grid_y))
    plt.figure()
    plt.imshow(values, cmap="Paired", extent=(XLIM[0], XLIM[1], YLIM[0], YLIM[1]), origin="lower")
    plt.colorbar()
    plt.title(title)
    plt.show()


def vectorise(x):
    return np.array(x).reshape(-1, 1)


def normalise(x):
    return (x - np.amin(x)) / (np.amax(x) - np.amin(x))


class GPKernel:

    def __init__(self):
        pass
        # self.getEIBVField()
        # self.getVRField()
        # self.getGradientField()
        # self.getTotalCost()

    def setup(self):
        self.getGrid()
        self.getMean()
        self.getCov()
        self.getGroundTruth()

        self.mu_cond = self.mu_prior_vector
        self.Sigma_cond = self.Sigma_prior

    def getGrid(self):
        self.x = np.linspace(XLIM[0], XLIM[1], NX)
        self.y = np.linspace(YLIM[0], YLIM[1], NY)
        self.x_matrix, self.y_matrix = np.meshgrid(self.x, self.y)
        self.x_vector = self.x_matrix.reshape(-1, 1)
        self.y_vector = self.y_matrix.reshape(-1, 1)
        self.grid_vector = np.hstack((self.x_vector, self.y_vector))
        pass

    def setCoef(self):
        self.sigma = SIGMA
        self.eta = 4.5 / LATERAL_RANGE
        self.tau = NUGGET
        self.R = np.diagflat(self.tau ** 2)
        self.threshold = THRESHOLD

    @staticmethod
    def getPrior(x, y):
        return (1 - np.exp(- ((x - 1.) ** 2 + (y - .5) ** 2) / .07))
                # 1 - np.exp(- ((x - 1.) ** 2 + (y - .5) ** 2) / .05))
                # 1 - np.exp(- ((x - .5) ** 2 + (y - .0) ** 2) / .004) +
                # 1 - np.exp(- ((x - .99) ** 2 + (y - .1) ** 2) / .1))


    def getMean(self):
        self.mu_prior_vector = vectorise(self.getPrior(self.x_vector, self.y_vector))
        self.mu_prior_matrix = self.getPrior(self.x_matrix, self.y_matrix)
        # plotf_matrix(self.mu_prior_matrix, "Prior")

    def getCov(self):
        self.setCoef()
        DistanceMatrix = cdist(self.grid_vector, self.grid_vector)
        self.Sigma_prior = self.sigma ** 2 * (1 + self.eta * DistanceMatrix) * np.exp(-self.eta * DistanceMatrix)

    def getGroundTruth(self):
        self.mu_truth = (self.mu_prior_vector.reshape(-1, 1) +
                         np.linalg.cholesky(self.Sigma_prior) @
                         np.random.randn(len(self.mu_prior_vector)).reshape(-1, 1))
        # plotf_vector(self.grid_vector, self.mu_truth, "Truth")

    def getIndF(self, x, y):
        x, y = map(vectorise, [x, y])
        DM_x = x @ np.ones([1, len(self.x_vector)]) - np.ones([len(x), 1]) @ self.x_vector.T
        DM_y = y @ np.ones([1, len(self.y_vector)]) - np.ones([len(y), 1]) @ self.y_vector.T
        DM = DM_x ** 2 + DM_y ** 2
        ind_F = np.argmin(DM, axis = 1)
        return ind_F

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

    @staticmethod
    def getVarianceReduction(Sigma, F, R):
        Reduction = Sigma @ F.T @ np.linalg.solve(F @ Sigma @ F.T + R, F @ Sigma)
        vr = np.sum(np.diag(Reduction))
        return vr

    @staticmethod
    def getGradient(field):
        gradient_x, gradient_y = np.gradient(field)
        gradient_norm = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        return gradient_norm

    def getEIBVField(self):
        self.eibv = []
        t1 = time.time()
        for i in range(self.grid_vector.shape[0]):
            F = np.zeros([1, self.grid_vector.shape[0]])
            F[0, i] = True
            self.eibv.append(GPKernel.getEIBV(self.mu_cond, self.Sigma_cond, F, self.R, THRESHOLD))
        self.eibv = normalise(np.array(self.eibv))
        # print("EIBV: ", self.eibv)
        t2 = time.time()
        print("EIBV field time consumed: ", t2 - t1)
        # plotf_vector(self.grid_vector, self.eibv, "EIBV")

    def getBudgetField(self, goal):
        t1 = time.time()
        self.budget_field = np.sqrt((self.grid_vector[:, 0] - goal.x) ** 2 +
                                    (self.grid_vector[:, 1] - goal.y) ** 2)
        t2 = time.time()
        print("Budget field time consumed: ", t2 - t1)

    def getVRField(self):
        self.vr = np.zeros_like(self.x_matrix)
        t1 = time.time()
        for i in range(self.x_matrix.shape[0]):
            for j in range(self.x_matrix.shape[1]):
                ind_F = self.getIndF(self.x_matrix[i, j], self.y_matrix[i, j])
                F = np.zeros([1, self.grid_vector.shape[0]])
                F[0, ind_F] = True
                self.vr[i, j] = GPKernel.getVarianceReduction(self.Sigma_prior, F, self.R)
        self.vr = normalise(self.vr)
        t2 = time.time()
        print("Time consumed: ", t2 - t1)
        # plotf_vector(self.grid_vector, self.vr, "VR")
        plotf_matrix(self.vr, "VR")
        pass

    def getGradientField(self):
        self.gradient_prior = normalise(self.getGradient(self.mu_prior_matrix))
        plotf_matrix(self.gradient_prior, "Gradient Prior")
        pass

    def getTotalCost(self):
        self.cost_total = normalise(self.vr + self.gradient_prior)
        plotf_matrix(self.cost_total, "Cost total")


class Location:

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        pass


class TreeNode:

    def __init__(self, location=None, parent=None, cost=None, mu=None, Sigma=None, F=None, EIBV=None):
        self.location = location
        self.parent = parent
        self.cost = cost
        self.mu = mu
        self.Sigma = Sigma
        self.F = F
        self.EIBV = EIBV
        pass


class RRTConfig:

    def __init__(self, starting_location=None, ending_location=None,
                 goal_sample_rate=None, step=None, GPKernel=None,
                 budget=None, goal_location=None):
        self.starting_location = starting_location
        self.ending_location = ending_location
        self.goal_sample_rate = goal_sample_rate
        self.step = step
        self.GPKernel = GPKernel
        self.budget = budget
        self.goal_location = goal_location
        pass


class RRTStar:

    obstacles = np.array(OBSTACLES)
    polygon_obstacles = []

    def __init__(self, config=None):
        self.nodes = []
        self.config = config
        self.path = []
        self.starting_node = TreeNode(self.config.starting_location, None, 0,
                                      self.config.GPKernel.mu_prior_vector, self.config.GPKernel.Sigma_prior)
        self.ending_node = TreeNode(self.config.ending_location, None, 0)
        self.starting_node.EIBV = 0
        self.setF(self.starting_node)
        self.setF(self.ending_node)
        self.counter_fig = 0
        pass

    def setF(self, node):
        node.F = self.config.GPKernel.getIndF(node.location.x, node.location.y)

    def expand_trees(self):
        self.nodes.append(self.starting_node)
        t1 = time.time()
        for i in range(MAXNUM):
            # print("Iteration: ", i)
            if np.random.rand() <= self.config.goal_sample_rate:
                new_location = self.config.ending_location
            else:
                new_location = self.get_new_location()

            nearest_node = self.get_nearest_node(self.nodes, new_location)
            next_node = self.get_next_node(nearest_node, new_location)

            if self.isWithin(next_node):
                # print("Collision zone")
                continue

            next_node, nearest_node = self.rewire_tree(next_node, nearest_node)

            if self.isIntersect(nearest_node, next_node):
                continue

            if self.isarrived(next_node):
                self.ending_node.parent = next_node
            else:
                self.nodes.append(next_node)
            pass

            # plt.figure()
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
            # plt.savefig(FIGPATH + "T_{:04d}.png".format(self.counter_fig))
            # self.counter_fig += 1
            # # plt.show()
            # plt.close("all")
        t2 = time.time()
        print("Time consumed: ", t2 - t1)
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

    @staticmethod
    def get_distance_between_nodes(node1, node2):
        dist_x = node1.location.x - node2.location.x
        dist_y = node1.location.y - node2.location.y
        dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
        return dist

    def rewire_tree(self, node_current, node_nearest):
        ind_neighbour_nodes = self.get_neighbour_nodes(node_current)

        for i in range(len(ind_neighbour_nodes)):
            if self.get_cost_between_nodes(self.nodes[ind_neighbour_nodes[i]], node_current) < \
                    self.get_cost_between_nodes(node_nearest, node_current):
                node_nearest = self.nodes[ind_neighbour_nodes[i]]

            node_current.parent = node_nearest
            node_current.cost = self.get_cost_between_nodes(node_nearest, node_current)

        for i in range(len(ind_neighbour_nodes)):
            cost_current_neighbour = self.get_cost_between_nodes(node_current, self.nodes[ind_neighbour_nodes[i]])
            if cost_current_neighbour < self.nodes[ind_neighbour_nodes[i]].cost:
                self.nodes[ind_neighbour_nodes[i]].cost = cost_current_neighbour
                self.nodes[ind_neighbour_nodes[i]].parent = node_current
        return node_current, node_nearest

    def get_neighbour_nodes(self, node_current):
        distance_between_nodes = []
        for i in range(len(self.nodes)):
            if self.isIntersect(self.nodes[i], node_current):
                distance_between_nodes.append(np.inf)
            else:
                distance_between_nodes.append(RRTStar.get_distance_between_nodes(self.nodes[i], node_current))
        ind_neighbours = np.where(np.array(distance_between_nodes) <= RADIUS_NEIGHBOUR)[0]
        return ind_neighbours

    def get_cost_between_nodes(self, node1, node2):
        cost = (node1.cost +
                self.get_distance_between_nodes(node1, node2) +
                self.get_reward_between_nodes(node1, node2) +
                self.get_cost_according_to_budget(node1, node2))
               # RRTStar.get_cost_along_path(node1.location, node2.location)
        # print("Cost: ", cost)
        return cost

    def get_reward_between_nodes(self, node1, node2):
        self.setF(node2)
        node1.EIBV = self.get_eibv_for_node(node1)
        node2.EIBV = self.get_eibv_for_node(node2)
        reward = (node1.EIBV + node2.EIBV) / 2 * RRTStar.get_distance_between_nodes(node1, node2)
        return reward

    def get_eibv_for_node(self, node):
        return self.config.GPKernel.eibv[node.F]

    def get_cost_according_to_budget(self, node1, node2):
        N = 10
        x = np.linspace(node1.location.x, node2.location.x, N)
        y = np.linspace(node1.location.y, node2.location.y, N)
        cost = []
        for i in range(N):
            cost.append(self.get_cost_from_budget_field(Location(x[i], y[i])))
        cost_total = np.trapz(cost) / N
        # print("Cost total: ", cost_total)
        # cost1 = self.get_cost_from_budget_field(node1.location)
        # cost2 = self.get_cost_from_budget_field(node2.location)
        # cost_total = (cost1 + cost2) / 2 * RRTStar.get_distance_between_nodes(node1, node2)
        return cost_total

    def get_cost_from_budget_field(self, location):
        dist = np.sqrt((location.x - self.config.goal_location.x) ** 2 +
                       (location.y - self.config.goal_location.y) ** 2)
        return dist

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
        # print("cost total: ", cost_total)
        return cost_total

    @staticmethod
    def get_cost_from_field(location):
        # return 0
        return 2**2 * np.exp(-((location.x - .5) ** 2 + (location.y - .5) ** 2) / .09)


    def isarrived(self, node):
        dist = self.get_distance_between_nodes(self.ending_node, node)
        if dist < DISTANCE_TOLERANCE:
            return True
        else:
            return False

    def set_obstacles(self):
        for i in range(self.obstacles.shape[0]):
            self.polygon_obstacles.append(Polygon(list(map(tuple, self.obstacles[i]))))

    '''
    Collision detection
    '''
    def isWithin(self, node):
        point = Point(node.location.x, node.location.y)
        within = False
        for i in range(len(self.polygon_obstacles)):
            if self.polygon_obstacles[i].contains(point):
                within = True
        return within

    def isIntersect(self, node1, node2):
        line = LineString([(node1.location.x, node1.location.y),
                           (node2.location.x, node2.location.y)])
        intersect = False
        for i in range(len(self.polygon_obstacles)):
            if self.polygon_obstacles[i].intersects(line):
                intersect = True
        return intersect
    '''
    End of collision detection
    '''

    def get_shortest_path(self):
        self.path.append([self.ending_node.location.x, self.ending_node.location.y])
        pointer_node = self.ending_node
        while pointer_node.parent is not None:

            node = pointer_node.parent
            self.path.append([node.location.x, node.location.y])
            pointer_node = node

        # self.path.append([self.starting_node.location.x, self.starting_node.location.y])
        self.path = np.array(self.path)

    def plot_tree(self):

        # plt.figure()
        if np.any(self.obstacles):
            for i in range(len(self.obstacles)):
                obstacle = np.append(self.obstacles[i], self.obstacles[i][0, :].reshape(1, -1), axis=0)
                plt.plot(obstacle[:, 0], obstacle[:, 1], 'r-.')

        for node in self.nodes:
            if node.parent is not None:
                plt.plot([node.location.x, node.parent.location.x],
                         [node.location.y, node.parent.location.y], "-g")

        path = np.array(self.path)
        plt.plot(path[:, 0], path[:, 1], "-r")
        plt.plot(self.config.starting_location.x, self.config.starting_location.y, 'k*', ms=10)
        plt.plot(self.config.ending_location.x, self.config.ending_location.y, 'g*')
        plt.grid()
        plt.title("rrt_star")
        # plt.savefig(FIGPATH + "T_{:04d}.png".format(self.counter_fig))
        # plt.show()


class GOOGLE:

    def __init__(self):
        self.trajectory = []
        pass

    def pathplanner(self):
        self.gp = GPKernel()
        self.gp.setup()
        self.gp.getEIBVField()

        starting_loc = Location(1., .0)
        goal_loc = Location(.0, .0)
        self.gp.getBudgetField(goal_loc)

        ind_min_cost = np.argmin(self.gp.eibv)
        ending_loc = Location(self.gp.grid_vector[ind_min_cost, 0], self.gp.grid_vector[ind_min_cost, 1])
        # ending_loc = Location(.0, 1.)

        # plotf_vector(self.gp.grid_vector, self.gp.mu_truth, "Truth")
        # plt.show()
        #
        # plotf_vector(self.gp.grid_vector, self.gp.mu_prior_vector, "Prior")
        # plt.show()

        budget = BUDGET
        distance_travelled = 0
        current_loc = [starting_loc.x, starting_loc.y]
        self.trajectory.append([starting_loc.x, starting_loc.y])

        print("Total budget: ", budget)

        for i in range(NUM_STEPS):
            print("Step: ", i)

            # == path planning ==
            rrtconfig = RRTConfig(starting_location=starting_loc, ending_location=ending_loc,
                                  goal_sample_rate=GOAL_SAMPLE_RATE, step=STEP, GPKernel=self.gp,
                                  budget=budget, goal_location=goal_loc)
            self.rrt = RRTStar(rrtconfig)
            self.rrt.expand_trees()
            self.rrt.get_shortest_path()
            path = self.rrt.path


            # == plotting ==
            fig = plt.figure(figsize=(20, 5))
            gs = GridSpec(nrows=1, ncols=4)
            ax = fig.add_subplot(gs[0])
            cmap = get_cmap("RdBu", 10)
            plotf_vector(self.gp.grid_vector, self.gp.mu_truth, "Ground Truth", cmap=cmap)

            ax = fig.add_subplot(gs[1])
            plotf_vector(self.gp.grid_vector, self.gp.mu_cond, "Conditional Mean", cmap=cmap)
            plotf_trajectory(self.trajectory)

            ax = fig.add_subplot(gs[2])
            plotf_vector(self.gp.grid_vector, np.sqrt(np.diag(self.gp.Sigma_cond)), "Prediction Error", cmap=cmap)
            plotf_trajectory(self.trajectory)

            ax = fig.add_subplot(gs[3])
            self.rrt.plot_tree()
            plotf_vector(self.gp.grid_vector, .6 * self.gp.eibv + .45 * self.gp.budget_field, "EIBV cost valley + Budget Radar", alpha=.1, cmap=cmap)
            plotf_budget_radar([goal_loc.x, goal_loc.y], budget)
            # plt.show()
            plt.savefig(FIGPATH + "P_{:03d}.png".format(i))
            plt.close("all")
            # == end plotting ==


            next_loc = path[-2, :]
            self.trajectory.append(next_loc)
            distance_travelled = np.sqrt((current_loc[0] - next_loc[0]) ** 2 +
                                         (current_loc[1] - next_loc[1]) ** 2)
            current_loc = next_loc

            budget = budget - distance_travelled
            print("Budget left: ", budget)

            ind_F = self.gp.getIndF(next_loc[0], next_loc[1])
            F = np.zeros([1, self.gp.grid_vector.shape[0]])
            F[0, ind_F] = True
            self.gp.mu_cond, self.gp.Sigma_cond = self.gp.GPupd(self.gp.mu_cond, self.gp.Sigma_cond, F,
                                                                self.gp.R, F @ self.gp.mu_truth)
            self.gp.getEIBVField()
            self.gp.getBudgetField(goal_loc)

            starting_loc = Location(next_loc[0], next_loc[1])
            node_start = TreeNode(starting_loc)
            node_end = TreeNode(ending_loc)
            if RRTStar.get_distance_between_nodes(node_start, node_end) < DISTANCE_TOLERANCE:
                print("Arrived")

            ind_min_cost = np.argmin(.6 * self.gp.eibv + .45 * self.gp.budget_field)
            ending_loc = Location(self.gp.grid_vector[ind_min_cost, 0], self.gp.grid_vector[ind_min_cost, 1])

            #     starting_loc = Location(.0, 1.)
            #     ending_loc = Location(1., 1.)


        pass




if __name__ == "__main__":
    # starting_loc = Location(.0, .0)
    # ending_loc = Location(.0, 1.)
    # gp = GPKernel()
    # rrtconfig = RRTConfig(starting_location=starting_loc, ending_location=ending_loc, goal_sample_rate=GOAL_SAMPLE_RATE,
    #                       step=STEP, GPKernel=gp)
    # rrt = RRTStar(rrtconfig)
    # rrt.set_obstacles()
    # rrt.expand_trees()
    # rrt.get_shortest_path()
    # rrt.plot_tree()
    g = GOOGLE()
    g.pathplanner()
    pass


#%%
cmap = get_cmap("RdBu", 10)

plt.scatter(g.gp.grid_vector[:, 0], g.gp.grid_vector[:, 1], c=g.gp.budget_field, cmap=cmap)
plt.colorbar()
plt.show()


