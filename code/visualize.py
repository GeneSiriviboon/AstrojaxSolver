import matplotlib.pyplot as plt
from copy import copy
from matplotlib import animation
from util import *


def plot2D(canvas, state):
    if canvas is not None:
        (point0, point1, point2, line) = canvas
    else: 
        ax = plt.axes(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect='equal')
        point0, = ax.plot([], [], 'x',lw=2, label = '1')
        point1, = ax.plot([], [], 'o',lw=2, label = '1')
        point2, = ax.plot([], [], 'o',lw=2, label = '2')
        line, = ax.plot([], [], 'black',lw=2, label = '2')
    
    endpoint = state[12:15]
    point0.set_data(endpoint[0], endpoint[2])
    point1.set_data(state[0], state[2])
    point2.set_data(state[3], state[5])
    line.set_data([endpoint[0], state[0], state[3]],[endpoint[2], state[2], state[5]])
    

"""
animate the trajectory
 - inputs
    trajectory - array of the state of the system trajectory[i] => state at timestep i
    time - array of time at each timestep
"""
def animate2D(trajectory, time, A, w, save = None):

    fig = plt.figure()
    ax = plt.axes(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect = 'equal')

    point0, = ax.plot([], [], 'x',lw=2, label = '1')
    point1, = ax.plot([], [], 'o',lw=2, label = '1')
    point2, = ax.plot([], [], 'o',lw=2, label = '2')
    line, = ax.plot([], [], 'black',lw=2, label = '2')
    T_text = ax.text(0.05, 1.01, ' ', transform=ax.transAxes, fontsize = 16, color = 'k')
    endpoint = A * np.sin(w * time)[:, np.newaxis] 
    # initialization function: plot the background of each frame
    def init():
        point0.set_data([],[])
        point1.set_data([],[])
        point2.set_data([],[])
        line.set_data([],[])
        T_text.set_text('')
        return point0, point1,point2,line,T_text

    # animation function.  This is called sequentially
    def animate(i):
        t = time[i]
        plot2D((point0, point1, point2, line), np.hstack([trajectory[i], endpoint[i]]))
        l = length(trajectory[i], A, w, t)
        E = Energy(trajectory[i])
        T_text.set_text('L = {:1.3f} E = {:2.3f} t = {:2.3f}'.format(l, E, t))
        return point0, point1, point2, line, T_text

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=len(trajectory), interval=20, blit=False)

    dt = time[1] - time[0]
    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    if save is not None:
        print('saving...')
        anim.save(save, fps=1/dt/3, extra_args=['-vcodec', 'libx264'], dpi = 300)
        print('saved')
    plt.show()


if __name__ == '__main__':
    z = np.random.random([100, 10])
    z[:, 1] = 0
    z[:, 3] = 0
    z[:, 4] = 0.5
    # z = np.array([1/4, 0, 0, 0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])[np.newaxis, :]
    # print(z.shape)
    endpoint = np.zeros([100, 3])
    endpoint[:, 0] = 0.3
    states = polar2Cartesian(z[:,:5], z[:, 5:10], endpoint, endpoint, endpoint)
    # print(z.shape)
    
    
    fig = plt.figure()
    for i in range(100):
        plot2D(None, states[i])
    plt.show()