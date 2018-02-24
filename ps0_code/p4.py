# CS231A Homework 0, Problem 4
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from imageio import imwrite 
from skimage.color import rgb2gray
from scipy import misc

def getRankNApprox(U, S, V, n):
    Sn = np.zeros(S.shape);
    Sn[:n, :n] = S[:n, :n];
    return U.dot(Sn).dot(V);

def main():
    # ===== Problem 4a =====
    # Read in image1 as a grayscale image. Take the singular value
    # decomposition of the image.

    img1 = None

    # BEGIN YOUR CODE HERE
    img1 = np.array(imread("image1.jpg"));
    img1 = rgb2gray(img1);
    U, s, V = np.linalg.svd(img1, full_matrices=True);
    S = np.diag(s);
    print U.shape, s.shape, V.shape

    # END YOUR CODE HERE

    # ===== Problem 4b =====
    # Save and display the best rank 1 approximation 
    # of the (grayscale) image1.

    rank1approx = None

    # BEGIN YOUR CODE HERE
    rank1approx = getRankNApprox(U, S, V, 1);
    imwrite("ps0_4b.png", rank1approx);
    plt.imshow(rank1approx, cmap = "gray");
    plt.show();

    # END YOUR CODE HERE

    # ===== Problem 4c =====
    # Save and display the best rank 20 approximation
    # of the (grayscale) image1.

    rank20approx = None

    # BEGIN YOUR CODE HERE
    rank20approx = getRankNApprox(U, S, V, 20);
    imwrite("ps0_4c.png", rank20approx);
    plt.imshow(rank20approx, cmap = "gray");
    plt.show();

    # END YOUR CODE HERE


if __name__ == '__main__':
    main()
