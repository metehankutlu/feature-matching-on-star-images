import cv2
import numpy as np

def read_imgs(img1path, img2path):
    """Function to read images.

    Args:
        img1path: The path of the first image.
        img2path: The path of the second image.

    Returns:
        img1, img2: Images read from the disk.
    """
    img1 = cv2.imread(img1path)
    img2 = cv2.imread(img2path)

    if img1.size > img2.size:
        img1, img2 = img2, img1

    return img1, img2

def get_keypoints(img1, img2):
    """Function to detect keypoints with Fast Feature Detector.

    Args:
        img1: The first image.
        img2: The second image.

    Returns:
        kp1, kp2: The keypoints detected from the images respectively.
    """
    fast = cv2.FastFeatureDetector_create()
    fast.setThreshold(10)
    fast.setNonmaxSuppression(0)

    kp1 = fast.detect(img1)
    kp2 = fast.detect(img2)

    return kp1, kp2

def get_descriptors(img1, kp1, img2, kp2):
    """Function to get descriptors using Brisk.

    Args:
        img1: The first image.
        kp1: The keypoints of the first image.
        img2: The second image.
        kp2: The keypoints of the second image.

    Returns:
        kp1, des1, kp2, des2: The keypoints and desriptors of the first 
        and second images respectively.
    """
    brisk = cv2.BRISK_create()

    kp1, des1 = brisk.compute(img1, kp1)
    kp2, des2 = brisk.compute(img2, kp2)

    return kp1, des1, kp2, des2

def get_matches(des1, des2):
    """Function to get matches between images using Brute Force feature 
    matching algorithm.

    Args:
        des1: The descriptors of the first image.
        des2: The descriptors of the second image.

    Returns:
        The matches between img1 and img2.
    """
    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append([m])
    
    return good

def get_coordinates(img1, kp1, img2, kp2, matches):
    """Function to get coordinates of the rectangle surrounding the matches.

    Args:
        img1: The first image.
        kp1: The keypoints of the first image.
        img2: The second image.
        kp2: The keypoints of the second image.
        matches: The matches between img1 and img2.

    Returns:
        Coordinates of the rectangle surrounding the matches.
    """
    src_points = np.float32([kp1[m[0].queryIdx].pt for m in matches])
    src_points = np.reshape(src_points, (-1, 1, 2))
    dst_points = np.float32([kp2[m[0].trainIdx].pt for m in matches])
    dst_points = np.reshape(dst_points, (-1, 1, 2))
    m, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    h,w,_ = img1.shape
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]])
    pts = np.reshape(pts, (-1,1,2))
    dst = cv2.perspectiveTransform(pts,m)

    return dst