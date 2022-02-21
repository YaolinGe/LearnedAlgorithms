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









