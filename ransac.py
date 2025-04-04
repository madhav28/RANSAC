import numpy as np
from matplotlib import pyplot as plt
from common import *

def homography_transform(X, H):
    # TODO
    # Perform homography transformation on a set of points X
    # using homography matrix H
    # Input - a set of 2D points in an array with size (N,2)
    #         a 3*3 homography matrix 
    # Output - a set of 2D points in an array with size (N,2)

    X = np.hstack((X, np.ones((X.shape[0], 1))))
    Y_hat = X @ H.T
    Y = Y_hat / Y_hat[:, 2:3]
    return Y[:, :2]


def fit_homography(XY):
    # TODO
    # Given two set of points X, Y in one array,
    # fit a homography matrix from X to Y
    # Input - an array with size(N,4), each row contains two
    #         points in the form[x^T_i,y^T_i]1×4
    # Output - a 3*3 homography matrix
    X = XY[:, :2]
    Y = XY[:, 2:4]

    A = []
    for i in range(X.shape[0]):
        x, y = X[i]
        x_prime, y_prime = Y[i]
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
    A = np.array(A)

    U, S, Vh = np.linalg.svd(A)
    H = Vh[-1].reshape(3, 3)
    
    return H


def p1():
    # 1. load points X from p1/transform.npy
    XY = np.load('./data/transform.npy')
    X = XY[:, 0:2]
    Y = XY[:, 2:4] 
    X = np.hstack((X, np.ones((X.shape[0], 1)))) 

    # 2. fit a transformation y=Sx+t
    M = np.linalg.inv(X.T @ X) @ X.T @ Y
    print("S =")
    print(M[:2, :].T)
    print("t =")
    print(M[2:3, :])

    # 3. transform the points
    Y_hat = X @ M

    # 4. plot the original points and transformed points
    plt.scatter(X[:, 0], X[:, 1], label='x', color='blue')
    plt.scatter(Y[:, 0], Y[:, 1], label='y', color='green')
    plt.scatter(Y_hat[:, 0], Y_hat[:, 1], label='y_hat', color='red')
    plt.legend()
    plt.savefig("./results/transform.png")
    plt.close()

    case = 8
    for i in range(case):
        XY = np.load('./data/points_case_'+str(i)+'.npy')
        # 1. generate your Homography matrix H using X and Y
        #
        #    specifically: fill function fit_homography() 
        #    such that H = fit_homography(XY)
        H = fit_homography(XY)
        # 2. Report H in your report
        print(f"case: {i}")
        print("H = ")
        print(H)
        # 3. Transform the points using H
        #
        #    specifically: fill function homography_transform
        #    such that Y_H = homography_transform(X, H)
        Y_H = homography_transform(XY[:,:2], H)
        # 4. Visualize points as three images in one figure
        # the following codes plot figure for you
        plt.scatter(XY[:,1],XY[:,0],c="red") #X
        plt.scatter(XY[:,3],XY[:,2],c="green") #Y
        plt.scatter(Y_H[:,1],Y_H[:,0],c="blue") #Y_hat
        plt.savefig('./results/case_'+str(i))
        plt.close()


if __name__ == "__main__":
    p1()