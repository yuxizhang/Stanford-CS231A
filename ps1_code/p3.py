# CS231A Homework 1, Problem 3
import numpy as np
from utils import mat2euler
import math

'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - a list of all the points where each row is (x, y). Generally,
            it will contain four points: two for each parallel line.
            You can use any convention you'd like, but our solution uses the
            first two rows as points on the same line and the last
            two rows as points on the same line.
Returns:
    vanishing_point - the pixel location of the vanishing point
'''
def compute_vanishing_point(points):
    points = points.astype(float);
    # k = (y2 - y1) / (x2 - x1)
    k1 = (points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0]);
    k2 = (points[3, 1] - points[2, 1]) / (points[3, 0] - points[2, 0]);

    # y - y1 = k(x - x1)
    x = (k1 * points[0, 0] - points[0, 1] - k2 * points[2, 0] + points[2, 1]) / (k1 - k2);
    y = k1 * (x - points[0, 0]) + points[0, 1];
    print "Vanishing point: ", x, y;
    return np.array((x, y));

'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''
def compute_K_from_vanishing_points(vanishing_points):
    v1 = vanishing_points[0];
    v2 = vanishing_points[1];
    v3 = vanishing_points[2];
    
    # v1.T * W * v2 = 0
    # Figure out matV * w = 0, where w.T = [w1, w4, w5, 1].
    v12 = [v1[0] * v2[0] + v1[1] * v2[1], v1[0] + v2[0], v1[1] + v2[1], 1];
    v13 = [v1[0] * v3[0] + v1[1] * v3[1], v1[0] + v3[0], v1[1] + v3[1], 1];
    v23 = [v2[0] * v3[0] + v2[1] * v3[1], v2[0] + v3[0], v2[1] + v3[1], 1];
    matV = np.vstack((v12, v13, v23));

    # Solve matV * w = 0 using SVD.
    U, s, VT = np.linalg.svd(matV);
    V = VT.T;
    w = V[:, -1];
    w /= w[-1];
    W = np.array([[w[0], 0, w[1]],
         [0, w[0], w[2]],
         [w[1], w[2], w[3]]]);

    # Solve W^-1 = K * K.T using Cholesky.
    K = np.linalg.pinv(np.linalg.cholesky(W)).T;
    return K;

'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''
def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    v1 = np.hstack((vanishing_pair1[0], 1));
    v2 = np.hstack((vanishing_pair1[1], 1));
    v3 = np.hstack((vanishing_pair2[0], 1));
    v4 = np.hstack((vanishing_pair2[1], 1));

    l1 = np.cross(v1, v2);
    l2 = np.cross(v3, v4);

    # w^-1 = K * K.T
    winv = K.dot(K.T);
    cos = l1.T.dot(winv).dot(l2) / (np.sqrt(l1.T.dot(winv).dot(l1)) * np.sqrt(l2.T.dot(winv).dot(l2)));
    return np.arccos(cos) / np.pi * 180;

'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''
def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    V = np.hstack((vanishing_points1, np.ones((3, 1)))).T;
    Vb = np.hstack((vanishing_points2, np.ones((3, 1)))).T;
    # Compute d = K^-1 * v
    D = np.linalg.inv(K).dot(V);
    Db = np.linalg.inv(K).dot(Vb);
    # Divided by 2-norm
    for i in range(3):
        D[:, i] /= np.linalg.norm(D[:, i]);
        Db[:, i] /= np.linalg.norm(Db[:, i]);
    # Db = R * D
    # R = Db * D^-1
    R = Db.dot(np.linalg.inv(D));
    return R;

if __name__ == '__main__':
    # Part A: Compute vanishing points
    v1 = compute_vanishing_point(np.array([[674,1826],[2456,1060],[1094,1340],[1774,1086]]))
    v2 = compute_vanishing_point(np.array([[674,1826],[126,1056],[2456,1060],[1940,866]]))
    v3 = compute_vanishing_point(np.array([[1094,1340],[1080,598],[1774,1086],[1840,478]]))

    v1b = compute_vanishing_point(np.array([[314,1912],[2060,1040],[750,1378],[1438,1094]]))
    v2b = compute_vanishing_point(np.array([[314,1912],[36,1578],[2060,1040],[1598,882]]))
    v3b = compute_vanishing_point(np.array([[750,1378],[714,614],[1438,1094],[1474,494]]))

    # Part B: Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    print "Intrinsic Matrix:\n",compute_K_from_vanishing_points(vanishing_points)

    K_actual = np.array([[2448.0, 0, 1253.0],[0, 2438.0, 986.0],[0,0,1.0]])
    print
    print "Actual Matrix:\n", K_actual

    # Part D: Estimate the angle between the box and floor
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094,1340],[1774,1086],[1080,598],[1840,478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2], K_actual)
    print
    print "Angle between floor and box:", angle

    # Part E: Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print
    print "Rotation between two cameras:\n", rotation_matrix
    z,y,x = mat2euler(rotation_matrix)
    print
    print "Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi)
    print "Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi)
    print "Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi)
