import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.io as sio
from epipolar_utils import *
from solve_homogeneous import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''
def lls_eight_point_alg(points1, points2):
    N = points1.shape[0];
    W = np.ones((N, 9));
    for i in range(N):
        p1 = np.array(points1[i]).reshape((3, 1));
        p2 = np.array(points2[i]).reshape((3, 1));
        mi = p2.dot(p1.T).reshape((1, 9));
        W[i, :] = mi;

    f = solveHomogeneous(W);
    F_hat = np.array(f).reshape((3, 3));
    U, s, VT = np.linalg.svd(F_hat);
    s2 = np.zeros(s.shape);
    s2[:2] = s[:2];
    S2 = np.diag(s2);
    F = U.dot(S2).dot(VT);
    return F;

'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    np1, T1 = normPoints(points1);
    np2, T2 = normPoints(points2);
    Fq = lls_eight_point_alg(np1, np2);
    return T2.T.dot(Fq).dot(T1);

def normPoints(points):
    center = np.mean(points, axis=0);
    dxs = points[:, 0] - center[0];
    dys = points[:, 1] - center[1];
    mean_square = np.mean(dxs * dxs + dys * dys);
    scale = np.sqrt(2 / mean_square);

    T = np.array([[scale, 0, -center[0] * scale],
                  [0, scale, -center[1] * scale], 
                  [0, 0, 1]]);
    newpoints = T.dot(points.T).T;
    return newpoints, T;


'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''

def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    l1 = F.T.dot(points2.T).T;
    plt.subplot(121);
    plotLines(im1, l1, points1);
    l2 = F.dot(points1.T).T;
    plt.subplot(122);
    plotLines(im2, l2, points2);
    plt.show();

def plotLines(im, l, p):
    N = l.shape[0];
    x = np.array([0, 512]);
    for i in range(N):
        y = (-l[i, 2] - l[i, 0] * x) / l[i, 1];
        plt.plot(p[i, 0], p[i, 1], 'go'); 
        plt.plot(x, y, 'r-');
    plt.imshow(im, cmap = 'gray');
    return;


'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    N = points1.shape[0];
    l2 = F.dot(points1.T).T;
    sumdist = 0;
    for i in range(N):
        sumdist += np.abs(l2[i, :].dot(points2[i, :].T)) /\
            np.sqrt(l2[i, 0] ** 2 + l2[i, 1] ** 2);
    return sumdist / N;

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print '-'*80
        print "Set:", im_set
        print '-'*80

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)

        print "Fundamental Matrix from LLS  8-point algorithm:\n", F_lls
        print "Distance to lines in image 1 for LLS:", \
                compute_distance_to_epipolar_lines(points1, points2, F_lls)
        print "Distance to lines in image 2 for LLS:", \
                compute_distance_to_epipolar_lines(points2, points1, F_lls.T)

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i])) 
                for i in xrange(points1.shape[0])]
        print "p'^T F p =", np.abs(pFp).max()
        print "Fundamental Matrix from normalized 8-point algorithm:\n", \
                F_normalized
        print "Distance to lines in image 1 for normalized:", \
                compute_distance_to_epipolar_lines(points1, points2, F_normalized)
        print "Distance to lines in image 2 for normalized:", \
                compute_distance_to_epipolar_lines(points2, points1, F_normalized.T)

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()
        break;
