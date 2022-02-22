import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


xlim = [0, 1]
ylim = [0, 1]

loc_start = [.5, 0.5]
loc_end = [1, 1]

step = .01

NUM = 50

loc = []
loc.append(loc_start)


def dist(loc1, loc2):
    return np.linalg.norm(np.array(loc1) - np.array(loc2))


def step_from_to(loc1, loc2):
    if dist(loc1, loc2) < step:
        return loc2
    else:
        theta = np.arctan2(loc2[1] - loc1[1], loc2[0] - loc1[0])
        return [loc1[0] + step * np.cos(theta), loc1[1] + step * np.sin(theta)]

fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    print("Step: ", i)
    loc_rng = np.random.rand(2).tolist()

    loc_nearest = loc[0]
    for l in loc:
        if dist(l, loc_rng) < dist(loc_nearest, loc_rng):
            loc_nearest = l

    print("nearest: ", loc_nearest)

    loc_new = step_from_to(loc_nearest, loc_rng)
    loc.append(loc_new)


    line.set_data([loc_nearest[0], loc_new[0]],
                  [loc_nearest[1], loc_new[1]])
    return line
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)
anim.save('/Users/yaoling/OneDrive - NTNU/Self-improvements/LearnedAlgorithms/pathplanning/fig/test.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


#%%

# !/usr/bin/env python

# rrtstar.py
# This program generates a
# asymptotically optimal rapidly exploring random tree (RRT* proposed by Sertac Keraman, MIT) in a rectangular region.
#
# Originally written by Steve LaValle, UIUC for simple RRT in
# May 2011
# Modified by Md Mahbubur Rahman, FIU for RRT* in
# January 2016

import sys, random, math, pygame
from pygame.locals import *
from math import sqrt, cos, sin, atan2

# constants
XDIM = 640
YDIM = 480
WINSIZE = [XDIM, YDIM]
EPSILON = 7.0
NUMNODES = 5000
RADIUS = 15


def dist(p1, p2):
    return sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


def step_from_to(p1, p2):
    if dist(p1, p2) < EPSILON:
        return p2
    else:
        theta = atan2(p2[1] - p1[1], p2[0] - p1[0])
        return p1[0] + EPSILON * cos(theta), p1[1] + EPSILON * sin(theta)


def chooseParent(nn, newnode, nodes):
    for p in nodes:
        if dist([p.x, p.y], [newnode.x, newnode.y]) < RADIUS and p.cost + dist([p.x, p.y],
                                                                               [newnode.x, newnode.y]) < nn.cost + dist(
                [nn.x, nn.y], [newnode.x, newnode.y]):
            nn = p
    newnode.cost = nn.cost + dist([nn.x, nn.y], [newnode.x, newnode.y])
    newnode.parent = nn
    return newnode, nn


def reWire(nodes, newnode, pygame, screen):
    white = 255, 240, 200
    black = 20, 20, 40
    for i in range(len(nodes)):
        p = nodes[i]
        if p != newnode.parent and dist([p.x, p.y], [newnode.x, newnode.y]) < RADIUS and newnode.cost + dist([p.x, p.y],
                                                                                                             [newnode.x,
                                                                                                              newnode.y]) < p.cost:
            pygame.draw.line(screen, white, [p.x, p.y], [p.parent.x, p.parent.y])
            p.parent = newnode
            p.cost = newnode.cost + dist([p.x, p.y], [newnode.x, newnode.y])
            nodes[i] = p
            pygame.draw.line(screen, black, [p.x, p.y], [newnode.x, newnode.y])
    return nodes


def drawSolutionPath(start, goal, nodes, pygame, screen):
    pink = 200, 20, 240
    nn = nodes[0]
    for p in nodes:
        if dist([p.x, p.y], [goal.x, goal.y]) < dist([nn.x, nn.y], [goal.x, goal.y]):
            nn = p
    while nn != start:
        pygame.draw.line(screen, pink, [nn.x, nn.y], [nn.parent.x, nn.parent.y], 5)
        nn = nn.parent


class Node:
    x = 0
    y = 0
    cost = 0
    parent = None

    def __init__(self, xcoord, ycoord):
        self.x = xcoord
        self.y = ycoord


def main():
    # initialize and prepare screen
    pygame.init()
    screen = pygame.display.set_mode(WINSIZE)
    pygame.display.set_caption('RRTstar')
    white = 255, 240, 200
    black = 20, 20, 40
    screen.fill(white)

    nodes = []

    # nodes.append(Node(XDIM/2.0,YDIM/2.0)) # Start in the center
    nodes.append(Node(0.0, 0.0))  # Start in the corner
    start = nodes[0]
    goal = Node(630.0, 470.0)
    for i in range(NUMNODES):
        rand = Node(random.random() * XDIM, random.random() * YDIM)
        nn = nodes[0]
        for p in nodes:
            if dist([p.x, p.y], [rand.x, rand.y]) < dist([nn.x, nn.y], [rand.x, rand.y]):
                nn = p
        interpolatedNode = step_from_to([nn.x, nn.y], [rand.x, rand.y])

        newnode = Node(interpolatedNode[0], interpolatedNode[1])
        [newnode, nn] = chooseParent(nn, newnode, nodes);

        nodes.append(newnode)
        pygame.draw.line(screen, black, [nn.x, nn.y], [newnode.x, newnode.y])
        nodes = reWire(nodes, newnode, pygame, screen)
        pygame.display.update()
        # print i, "    ", nodes

        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                sys.exit("Leaving because you requested it.")
    drawSolutionPath(start, goal, nodes, pygame, screen)
    pygame.display.update()


# if python says run, then we should run
if __name__ == '__main__':
    main()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False











