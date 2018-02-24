import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from fundamental_matrix_estimation import *
from solve_homogeneous import * 

'''
COMPUTE_EPIPOLE computes the epipole in homogenous coordinates
given matching points in two images and the fundamental matrix
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the Fundamental matrix such that (points1)^T * F * points2 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(points1, points2, F):
    l1 = F.T.dot(points2.T).T;
    e1 = solveHomogeneous(l1);
    e1 /= e1[-1];
    return e1;
    
'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    # Compute H2
    e = np.array(e2).reshape((3, 1));
    height, width = im2.shape;
    T = np.array([[1, 0, -width / 2],
                  [0, 1, -height / 2],
                  [0, 0, 1]]);
    Te = T.dot(e).flatten();
    c = np.sqrt(Te[0] ** 2 + Te[1] ** 2);
    if (Te[0] < 0):
        c = -c;
    R = np.array([[Te[0] / c, Te[1] / c, 0],
                  [-Te[1] / c, Te[0] / c, 0],
                  [0, 0, 1]]);
    f = R.dot(Te)[0];
    G = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [-1 / f, 0, 1]]);
    H2 = inv(T).dot(G).dot(R).dot(T);

    # Compute H1
    ex = np.array([[0, -e[2], e[1]],
                   [e[2], 0, -e[0]],
                   [-e[1], e[0], 0]]);
    vT = np.array([[1, 1, 1]]);
    M = ex.dot(F) + e.dot(vT);
    p1h = H2.dot(M).dot(points1.T).T;
    p1h /= p1h[:, [2]].dot(vT);
    p2h = H2.dot(points2.T).T;
    p2h /= p2h[:, [2]].dot(vT);
    b = p2h[:, 0];
    a, residuals, rank, s = np.linalg.lstsq(p1h, b, rcond=None);
    HA = np.array([[a[0], a[1], a[2]],
                   [0, 1, 0],
                   [0, 0, 1]]);
    H1 = HA.dot(H2).dot(M);
    return H1, H2;

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    print "e1", e1
    print "e2", e2

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print "H1:\n", H1
    print
    print "H2:\n", H2

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
