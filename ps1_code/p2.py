# CS231A Homework 1, Problem 2
import numpy as np

'''
DATA FORMAT

In this problem, we provide and load the data for you. Recall that in the original
problem statement, there exists a grid of black squares on a white background. We
know how these black squares are setup, and thus can determine the locations of
specific points on the grid (namely the corners). We also have images taken of the
grid at a front image (where Z = 0) and a back image (where Z = 150). The data we
load for you consists of three parts: real_XY, front_image, and back_image. For a
corner (0,0), we may see it at the (137, 44) pixel in the front image and the
(148, 22) pixel in the back image. Thus, one row of real_XY will contain the numpy
array [0, 0], corresponding to the real XY location (0, 0). The matching row in
front_image will contain [137, 44] and the matching row in back_image will contain
[148, 22]
'''

'''
COMPUTE_CAMERA_MATRIX
Arguments:
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    camera_matrix - The calibrated camera matrix (3x4 matrix)
'''
def compute_camera_matrix(real_XY, front_image, back_image):
    N = real_XY.shape[0];
    # Put two real_XY together, one for front position and one for back position.
    points = np.ones((2 * N, 4));
    points[:N, 0:2] = real_XY;
    points[:N, 2] *= 0;
    points[N:, 0:2] = real_XY;
    points[N:, 2] *= 150;

    # Construct matrix P in Pm = 0 equation.
    M = points.shape[0];
    matP = np.zeros((2 * M, 9));
    for row in range(2 * M):
        i = row / 2;
        if row & 1 == 0:
            matP[row, 0:4] = points[i];
        else:
            matP[row, 4:8] = points[i];
        if points[i, 2] == 0:
            # Front image
            matP[row, 8] = -front_image[i, row & 1];
        else:
            # Back image
            matP[row, 8] = -back_image[i - N, row & 1];
    
    # Solve m in Pm = 0 using SVD.
    U, s, VT = np.linalg.svd(matP);
    V = VT.T;
    m = V[:, -1];
    m /= m[-1];

    # Construct actual camera matrix.
    cameraMat = np.array(m[:-1]).reshape(2, 4);
    cameraMat = np.vstack((cameraMat, [0, 0, 0, 1]));
    return cameraMat;

'''
RMS_ERROR
Arguments:
     camera_matrix - The camera matrix of the calibrated camera
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    rms_error - The root mean square error of reprojecting the points back
                into the images
'''
def rms_error(camera_matrix, real_XY, front_image, back_image):
    N, M = real_XY.shape;
    colOnes = np.ones((N, 1));
    front_XYZ1 = np.hstack((real_XY, colOnes * 0, colOnes));
    back_XYZ1 = np.hstack((real_XY, colOnes * 150, colOnes));

    # cal_front_image and cal_back_image are the calculated image
    # points using estimated camera matrix.
    cal_front_image = camera_matrix.dot(front_XYZ1.T).T[:,:2];
    cal_back_image = camera_matrix.dot(back_XYZ1.T).T[:,:2];
    diff_front = front_image - cal_front_image;
    diff_back = back_image - cal_back_image;
    diff_total = np.vstack((diff_front, diff_back));
    err = np.sqrt((diff_total * diff_total).sum() / (N * 2));
    return err;

if __name__ == '__main__':
    # Loading the example coordinates setup
    real_XY = np.load('real_XY.npy')
    front_image = np.load('front_image.npy')
    back_image = np.load('back_image.npy')

    camera_matrix = compute_camera_matrix(real_XY, front_image, back_image)
    print "Camera Matrix:\n", camera_matrix
    print
    print "RMS Error: ", rms_error(camera_matrix, real_XY, front_image, back_image)
