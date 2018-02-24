# CS231A Homework 0, Problem 2
import numpy as np
import matplotlib.pyplot as plt


def main():
    # ===== Problem 2a =====
    # Define Matrix M and Vectors a,b,c in Python with NumPy

    M, a, b, c = None, None, None, None

    # BEGIN YOUR CODE HERE
    M = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 2]).reshape(4, 3);
    a = np.array([1, 1, 0]).reshape(3, 1);
    b = np.array([-1, 2, 5]).reshape(3, 1);
    c = np.array([0, 2, 3, 2]).reshape(4, 1);

    # END YOUR CODE HERE

    # ===== Problem 2b =====
    # Find the dot product of vectors a and b, save the value to aDotb

    aDotb = None

    # BEGIN YOUR CODE HERE
    aDotb = a.T.dot(b);
    print("Result for 2b:");
    print(aDotb);

    # END YOUR CODE HERE

    # ===== Problem 2c =====
    # Find the element-wise product of a and b

    # BEGIN YOUR CODE HERE
    result2c = a * b;
    print("Result for 2c:");
    print(result2c);

    # END YOUR CODE HERE

    # ===== Problem 2d =====
    # Find (a^T b)Ma

    # BEGIN YOUR CODE HERE
    result2d = a.T.dot(b) * M.dot(a);
    print("Result for 2d:");
    print(result2d);

    # END YOUR CODE HERE

    # ===== Problem 2e =====
    # Without using a loop, multiply each row of M element-wise by a.
    # Hint: The function repmat() may come in handy.

    newM = None

    # BEGIN YOUR CODE HERE
    newM = M * np.tile(a.T, (4, 1));
    print("Result for 2e:");
    print(newM);

    # END YOUR CODE HERE

    # ===== Problem 2f =====
    # Without using a loop, sort all of the values 
    # of M in increasing order and plot them.
    # Note we want you to use newM from e.

    # BEGIN YOUR CODE HERE
    result2f = np.sort(newM, axis = None);
    print("Result for 2f:");
    print(result2f);
    plt.plot(result2f);
    plt.savefig("ps0_2f.png")
    plt.show();

    # END YOUR CODE HERE


if __name__ == '__main__':
    main()
