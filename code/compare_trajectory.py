from astrojax import *
from astrojax_Newton import *


def compare(initial_state, t):
    astrojax = Astrojax()
    trajectory1 = astrojax.trajectory(initial_state, t)
    trajectory2 = solveAstrojax(initial_state, t)

    plt.xlabel('time', fontsize=15)
    plt.ylabel(r'$z_2$', fontsize=15)
    plt.title('Comparison of Trajectory Between \n Newton-Raphson Method and Autograd Method')
    plt.plot(t, trajectory1[:, 5], 'b', label = 'Newton-Raphson')
    plt.plot(t, trajectory2[:, 5], 'r', label = 'Autograd')

    plt.legend()

def compare_err(initial_state1, initial_state2, t, err):
    astrojax = Astrojax()
    trajectory1 = astrojax.trajectory(initial_state1, t)
    trajectory2 = astrojax.trajectory(initial_state2, t)

    plt.xlabel('time', fontsize=15)
    plt.ylabel(r'$z_2$', fontsize=15)
    plt.title(r'Comparison of Trajectory with $\epsilon$ =  {:.2E}'.format(err))
    plt.plot(t, trajectory1[:, 5], 'b', label = 'original')
    plt.plot(t, trajectory2[:, 5], 'r', label = 'perturbed')

    plt.legend()


if __name__ == '__main__':

    err = 1e-6

    state = np.array([0.5, 0.0, 0.0, 1.0, 0.0, 0.0,  
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    state2 = np.array([0.5 - err, 0.0, 0.0, 1.0, 0.0, 0.0,  
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    t = np.linspace(0, 60, num = 6000)

    # compare(state, t)
    # plt.savefig('./compare_traj.png', dpi = 300)
    # plt.show()
    # compare_err(state, state2, t, err)
    # plt.savefig('./compare_err6.png', dpi = 300)
    compare_err(state, state2, t, 1e-7)
    plt.savefig('./compare_err7.png', dpi = 300)
    plt.show()


