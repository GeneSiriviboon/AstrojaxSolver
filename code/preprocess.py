from astrojax import *

def generate_inputs_labels(num_data):

    astrojax = Astrojax()

    A = np.random.random([num_data, 3])  
    w = np.random.random([num_data]) * 3 
    t = np.random.random([num_data]) / w

    initial = generate_data(num_data, A, w, t)

    Dstate = astrojax.dstate(t_end = t[-1], A = A, w = w, p = False)
    
    Ts = []
    Acc = []
    for i in range(num_data):
        print_progress((i+1)/num_data,length = 30)
        Ts.append(astrojax.solveAcc(initial[i, :12], t[i], A[i], w[i]))

    return initial, np.array(Ts).reshape([num_data, 7]), A, w ,t


if __name__ == '__main__':
    train_inputs, train_labels, train_A, train_w , train_t = generate_inputs_labels(50000)
    np.savez('train.npz', inputs = train_inputs, labels = train_labels, A = train_A, w = train_w, t = train_t)
    test_inputs, test_labels, test_A, test_w , test_t = generate_inputs_labels(10000)
    np.savez('test.npz', inputs = test_inputs, labels = test_labels, A = test_A, w = test_w, t = test_t)
    




