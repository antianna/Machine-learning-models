import numpy as np
import matplotlib.pyplot as plt


def dist(u, v):
    # this only works because we are in 2 dimensional space (D = 2)
    diff = u - v
    return diff.dot(diff)

def cost(X, R, M):
    cost = 0
    # k is number of clusters
    for k in range(len(M)):
        # n is number of points
        for n in range(len(X)):
            # multiply the responsibility of each point to corresponding cluster
            # and multiply by distance to corresponding cluster
            cost += R[n,k]*dist(M[k], X[n])
    return cost

def plot_k_means(X, K, max_iter=20, beta=1.0):
    """
    :param K: number of clusters
    """
    # get shape of N
    N, D = X.shape

    # initialised means
    M = np.zeros((K,D))

    # responsibility matrix
    R = np.zeros((N,K))

    # initialise k clusters at random points in X
    for k in range(K):
        M[k] = X[np.random.choice(N)]

    # grid
    grid_width = 5
    grid_height = max_iter/grid_width
    random_colors = np.random.random((K,3))
    plt.figure(figsize=(20,15))

    # calculating cost function
    costs = []
    for i in range(max_iter):
        print(i)
        # plotting progression of clusters
        print(R)
        colors = R.dot(random_colors)
        plt.subplot(grid_width, grid_height, i+1)
        plt.scatter(X[:,0], X[:,1], c=colors)
        plt.title("i:{}, k:{}".format(i, K))

        for k in range(K):
            for n in range(N):
                R[n, k] = np.exp( -beta*dist(M[k], X[n])) / np.sum( np.exp(-beta*dist(M[j], X[n])) for j in range(K))

        # re-calculate the means
        for k in range(K):
            M[k] = R[:, k].dot(X) / R[:, k].sum()
        # break if cost doesnt change much from one iteration to next
        #costs[i] = cost(X, R, M)
        costs.append(cost(X, R, M))
        print('cost',i)

        if i >0:
            if np.abs(costs[i] - costs[i-1]) < 0.1:
                break
    plt.show()

    # plotting the cost
    plt.plot(costs)
    plt.title('costs, K:{}'.format(K))
    plt.show()



def main():
    # dimensions
    D = 2
    # a number
    s = 4

    # set initial means
    m1 = np.array([0,0])
    m2 = np.array([s,s])
    m3 = np.array([0,s])

    # number of samples
    N = 900
    # samples
    X = np.zeros((N,D))
    # assinging first, middle, and last 300 points to separate clusters with means m1, m2, m3
    X[:300, :] = np.random.randn(300, D) + m1
    X[300:600, :] = np.random.randn(300, D) + m2
    X[600:, :] = np.random.randn(300, D) + m3

    # plot random data
    plt.scatter(X[:, 0], X[:, 1])
    plt.title('The data')
    plt.show()

    # number of clusters
    K = 3
    plot_k_means(X, K)

    K = 5
    plot_k_means(X, K, max_iter=30)

    K = 5
    plot_k_means(X, K, max_iter=30, beta=0.3)



if __name__ == "__main__":
    main()