# CS231A Homework 0, Problem 3
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from imageio import imread
from imageio import imwrite 
from skimage.color import rgb2gray
from scipy import misc

def normalizeImg(img):
    imgMin = img.min();
    imgMax = img.max();
    return (img - imgMin) / (imgMax - imgMin);

def main():
    # ===== Problem 3a =====
    # Read in the images, image1.jpg and image2.jpg, as color images.

    img1, img2 = None, None

    # BEGIN YOUR CODE HERE
    img1 = np.array(imread("image1.jpg"));
    img2 = np.array(imread("image2.jpg"));

    # END YOUR CODE HERE

    # ===== Problem 3b =====
    # Convert the images to double precision and rescale them
    # to stretch from minimum value 0 to maximum value 1.

    # BEGIN YOUR CODE HERE
    img1f = img1.astype(float);
    img2f = img2.astype(float);
    img1f = normalizeImg(img1f);
    img2f = normalizeImg(img2f);

    # END YOUR CODE HERE

    # ===== Problem 3c =====
    # Add the images together and re-normalize them 
    # to have minimum value 0 and maximum value 1. 
    # Display this image.

    # BEGIN YOUR CODE HERE
    imgSum = img1f + img2f;
    imgSum = normalizeImg(imgSum);
    imwrite("ps0_3c.png", imgSum);
    plt.imshow(imgSum);
    plt.show();

    # END YOUR CODE HERE

    # ===== Problem 3d =====
    # Create a new image such that the left half of 
    # the image is the left half of image1 and the 
    # right half of the image is the right half of image2.

    newImage1 = None

    # BEGIN YOUR CODE HERE
    height, width, color = img1f.shape;
    leftHalf = img1f[:, : width / 2];
    rightHalf = img2f[:, width / 2 :];
    newImage1 = np.hstack((leftHalf, rightHalf));
    imwrite("ps0_3d.png", newImage1);
    plt.imshow(newImage1);
    plt.show();

    # END YOUR CODE HERE

    # ===== Problem 3e =====
    # Using a for loop, create a new image such that every odd 
    # numbered row is the corresponding row from image1 and the 
    # every even row is the corresponding row from image2. 
    # Hint: Remember that indices start at 0 and not 1 in Python.

    newImage2 = None

    # BEGIN YOUR CODE HERE
    newImage2 = np.zeros(img1f.shape).astype(float);
    for row in range(height):
        for col in range(width):
            newImage2[row, col] = img1f[row, col] if (row & 1) else img2f[row, col];
    imwrite("ps0_3e.png", newImage2);
    plt.imshow(newImage2);
    plt.show();

    # END YOUR CODE HERE

    # ===== Problem 3f =====
    # Accomplish the same task as part e without using a for-loop.
    # The functions reshape and repmat may be helpful here.

    newImage3 = None

    # BEGIN YOUR CODE HERE
    onePixel = np.ones(color);
    zeroPixel = np.zeros(color);

    evenUnit = np.vstack((onePixel, zeroPixel));
    evenMat = repmat(evenUnit, height / 2, width);
    evenMat = evenMat.reshape(height, width, color);

    oddMat = np.ones((height, width, color)) - evenMat;
    newImage3 = oddMat * img1f + evenMat * img2f;
    imwrite("ps0_3f.png", newImage3);
    plt.imshow(newImage3);
    plt.show();

    # END YOUR CODE HERE

    # ===== Problem 3g =====
    # Convert the result from part f to a grayscale image. 
    # Display the grayscale image with a title.

    # BEGIN YOUR CODE HERE
    grayImg = rgb2gray(newImage3);
    plt.imshow(grayImg, cmap='gray');
    plt.title("Grayscale image.");
    plt.savefig("ps0_3g.png");
    plt.clf();

    # END YOUR CODE HERE


if __name__ == '__main__':
    main()
