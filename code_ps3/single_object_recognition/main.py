import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from utils import *
import math
from collections import defaultdict


'''
MATCH_KEYPOINTS: Given two sets of descriptors corresponding to SIFT keypoints, 
find pairs of matching keypoints.

Note: Read Lowe's Keypoint matching, finding the closest keypoint is not
sufficient to find a match. thresh is the theshold for a valid match.

Arguments:
    descriptors1 - Descriptors corresponding to the first image. Each row
        corresponds to a descriptor. This is a ndarray of size (M_1, 128).

    descriptors2 - Descriptors corresponding to the second image. Each row
        corresponds to a descriptor. This is a ndarray of size (M_2, 128).

    threshold - The threshold which to accept from Lowe's Keypoint Matching
        algorithm

Returns:
    matches - An int ndarray of size (N, 2) of indices that for keypoints in 
        descriptors1 match which keypoints in descriptors2. For example, [7 5]
        would mean that the keypoint at index 7 of descriptors1 matches the
        keypoint at index 5 of descriptors2. Not every keypoint will necessarily
        have a match, so N is not the same as the number of rows in descriptors1
        or descriptors2. 
'''
def match_keypoints(descriptors1, descriptors2, threshold = 0.7):
    matches = np.empty((0, 2), int)
    for i in range(descriptors1.shape[0]):
        best_match_idx, best_match_dis = -1, float("inf")
        second_match_dis = float("inf")
        for j in range(descriptors2.shape[0]):
            dis = np.linalg.norm(descriptors1[i] - descriptors2[j], 2)
            if (dis < best_match_dis):
                second_match_dis = best_match_dis
                best_match_idx, best_match_dis = j, dis
            elif (dis < second_match_dis):
                second_match_dis = dis

        if (best_match_idx != -1):
            if (best_match_dis < second_match_dis * threshold):
                matches = np.vstack((matches, np.array([i, best_match_idx])))

    return matches


'''
REFINE_MATCH: Filter out spurious matches between two images by using RANSAC
to find a projection matrix. 

Arguments:
    keypoints1 - Keypoints in the first image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_1, 4).

    keypoints2 - Keypoints in the second image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_2, 4).

    matches - An int ndarray of size (N, 2) of indices that indicate what
        keypoints from the first image (keypoints1)  match with the second 
        image (keypoints2). For example, [7 5] would mean that the keypoint at
        index 7 of keypoints1 matches the keypoint at index 5 of keypoints2). 
        Not every keypoint will necessarily have a  match, so N is not the same
        as the number of rows in keypoints1 or keypoints2. 

    reprojection_threshold - If the reprojection error is below this threshold,
        then we will count it as an inlier during the RANSAC process.

    num_iterations - The number of iterations we will run RANSAC for.

Returns:
    inliers - A vector of integer indices that correspond to the inliers of the
        final model found by RANSAC.

    model - The projection matrix H found by RANSAC that has the most number of
        inliers.
'''
def refine_match(keypoints1, keypoints2, matches, reprojection_threshold = 10,
        num_iterations = 1000):
    print "matches", matches
    N = matches.shape[0]
    kp1 = keypoints1[:, 0 : 2]
    kp2 = keypoints2[:, 0 : 2]
    match_points1 = kp1[matches[:, 0]]
    match_points2 = kp2[matches[:, 1]]
    subset_size = 4;

    max_inliers = 0
    final_inliers = np.array([])
    final_H = np.array([])
    for it in range(num_iterations):
        subset = matches[random.sample(range(N), 4)]
        points1 = kp1[subset[:, 0]]
        points1 = eucl2homo(points1).T

        points2 = kp2[subset[:, 1]]
        points2 = eucl2homo(points2).T
        H = points2.dot(np.linalg.pinv(points1))

        points2_projected = H.dot(eucl2homo(match_points1).T)
        points2_projected = homo2eucl(points2_projected.T)
        err = np.linalg.norm(match_points2 - points2_projected, axis=1)
        
        total_inliers = sum(err < reprojection_threshold)
        inliers = np.arange(N)[err < reprojection_threshold]

        if (total_inliers > max_inliers):
            max_inliers = total_inliers
            final_inliers = inliers
            final_H = H

    return final_inliers, final_H

def eucl2homo(points):
    N = points.shape[0]
    return np.hstack((points, np.ones((N, 1))))

def homo2eucl(points):
    return (points.T[:-1] / points.T[-1]).T


'''
GET_OBJECT_REGION: Get the parameters for each of the predicted object
bounding box in the image

Arguments:
    keypoints1 - Keypoints in the first image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_1, 4).

    keypoints2 - Keypoints in the second image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_2, 4).

    matches - An int ndarray of size (N, 2) of indices that indicate what
        keypoints from the first image (keypoints1)  match with the second 
        image (keypoints2). For example, [7 5] would mean that the keypoint at
        index 7 of keypoints1 matches the keypoint at index 5 of keypoints2). 
        Not every keypoint will necessarily have a  match, so N is not the same
        as the number of rows in keypoints1 or keypoints2.

    obj_bbox - An ndarray of size (4,) that contains [xmin, ymin, xmax, ymax]
        of the bounding box. Note that the point (xmin, ymin) is one corner of
        the box and (xmax, ymax) is the opposite corner of the box.

    thresh - The threshold we use in Hough voting to state that we have found
        a valid object region.

Returns:
    cx - A list of the x location of the center of the bounding boxes

    cy - A list of the y location of the center of the bounding boxes

    w - A list of the width of the bounding boxes

    h - A list of the height of the bounding boxes

    orient - A list f the orientation of the bounding box. Note that the 
        theta provided by the SIFT keypoint is inverted. You will need to
        re-invert it.
'''
def get_object_region(keypoints1, keypoints2, matches, obj_bbox, thresh = 4, 
        nbins = 4):
    cx, cy, w, h, orient = [], [], [], [], []

    for match in matches:
        key_index1 = match[0]
        key_index2 = match[1]
        key_point1 = keypoints1[key_index1]
        key_point2 = keypoints2[key_index2]
        u1, v1, s1, theta1 = parse_para(key_point1)
        u2, v2, s2, theta2 = parse_para(key_point1)

        xmin, ymin, xmax, ymax = obj_bbox
        xc1 = (xmin + xmax) / 2.0
        yc1 = (ymin + ymax) / 2.0
        w1 = (xmax - xmin) * 1.0
        h1 = (ymax - ymin) * 1.0
        o2 = theta2 - theta1
        xc2 = (s2/s1)*np.cos(o2)*(xc1-u1) - (s2/s1)*np.sin(o2)*(yc1-v1) + u2
        yc2 = (s2/s1)*np.sin(o2)*(xc1-u1) + (s2/s1)*np.cos(o2)*(yc1-v1) + v2
        w2 = (s2/s1) * w1
        h2 = (s2/s1) * h1
        cx.append(xc2)
        cy.append(yc2)
        w.append(w2)
        h.append(h2)
        orient.append(o2)

    cx_min = min(cx)
    cx_max = max(cx)
    cy_min = min(cy)
    cy_max = max(cy)
    w_min = min(w)
    w_max = max(w)
    h_min = min(h)
    h_max = max(h)
    orient_min = min(orient)
    orient_max = max(orient)
    cx_bin_size = (cx_max - cx_min) / float(nbins)
    cy_bin_size = (cy_max - cy_min) / float(nbins)
    w_bin_size = (w_max - w_min) / float(nbins)
    h_bin_size = (h_max - h_min) / float(nbins)
    orient_bin_size = (orient_max - orient_min) / float(nbins)

    bins = defaultdict(list)
    for n in range(matches.shape[0]):
        cx_point = cx[n]
        cy_point = cy[n]
        w_point = w[n]
        orient_point = orient[n]

        for i in range(nbins):
            for j in range(nbins):
                for k in range(nbins):
                    for l in range(nbins):
                        if (cx_min+i*cx_bin_size <= cx_point <= cx_min+(i+1)*cx_bin_size):
                            if (cy_min+j*cy_bin_size <= cy_point <= cy_min+(j+1)*cy_bin_size):
                                if (w_min+k*w_bin_size <= w_point <= w_min+(k+1)*w_bin_size):
                                    if (orient_min+l*orient_bin_size <= orient_point <= orient_min+(l+1)*orient_bin_size):
                                        bins[(i,j,k,l)].append(n)

    cx0, cy0, w0, h0, orient0 = [], [], [], [], []
    for bin_index in bins:
        indices = bins[bin_index]
        votes = len(indices)

        if votes >= thresh:
            cx0.append(np.sum(np.array(cx)[indices]) / votes)
            cy0.append(np.sum(np.array(cy)[indices]) / votes)
            w0.append(np.sum(np.array(w)[indices]) / votes)
            h0.append(np.sum(np.array(h)[indices]) / votes)
            orient0.append(np.sum(np.array(orient)[indices]) / votes)

    return cx0, cy0, w0, h0, orient0

def parse_para(key_point):
    u1 = key_point[0]
    v1 = key_point[1]
    s1 = key_point[2]
    theta1 = key_point[3]


'''
MATCH_OBJECT: The pipeline for matching an object in one image with another

Arguments:
    im1 - The first image read in as a ndarray of size (H, W, C).

    descriptors1 - Descriptors corresponding to the first image. Each row
        corresponds to a descriptor. This is a ndarray of size (M_1, 128).

    keypoints1 - Keypoints in the first image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_1, 4).

    im2 - The second image read in as a ndarray of size (H, W, C).

    descriptors2 - Descriptors corresponding to the second image. Each row
        corresponds to a descriptor. This is a ndarray of size (M_2, 128).

    keypoints2 - Keypoints in the second image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_2, 4).

    obj_bbox - An ndarray of size (4,) that contains [xmin, ymin, xmax, ymax]
        of the bounding box. Note that the point (xmin, ymin) is one corner of
        the box and (xmax, ymax) is the opposite corner of the box.

Returns:
    descriptors - The descriptors corresponding to the keypoints inside the
        bounding box.

    keypoints - The pixel locations of the keypoints that reside in the 
        bounding box
'''
def match_object(im1, descriptors1, keypoints1, im2, descriptors2, keypoints2,
        obj_bbox):
    # Part A
    descriptors1, keypoints1, = select_keypoints_in_bbox(descriptors1,
        keypoints1, obj_bbox)
    matches = match_keypoints(descriptors1, descriptors2)
    #  plot_matches(im1, im2, keypoints1, keypoints2, matches)
    
    # Part B
    inliers, model = refine_match(keypoints1, keypoints2, matches)
    print "part b"
    print matches[inliers, :]
    #  plot_matches(im1, im2, keypoints1, keypoints2, matches[inliers,:])

    # Part C
    cx, cy, w, h, orient = get_object_region(keypoints1, keypoints2,
        matches[inliers,:], obj_bbox)

    plot_bbox(cx, cy, w, h, orient, im2)

if __name__ == '__main__':
    # Load the data
    data = sio.loadmat('SIFT_data.mat')
    images = data['stopim'][0]
    obj_bbox = data['obj_bbox'][0]
    keypoints = data['keypt'][0]
    descriptors = data['sift_desc'][0]
    
    np.random.seed(0)

    for i in [2, 1, 3, 4]:
        match_object(images[0], descriptors[0], keypoints[0], images[i],
            descriptors[i], keypoints[i], obj_bbox)
