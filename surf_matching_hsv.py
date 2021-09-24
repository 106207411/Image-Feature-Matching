"""The SURFmatching_HSV.py is used to mathch the two images by SURF.
   The steps of process are as follows:
   * Blur the images by median filter and use SURF to find the 20 best matches points.
   * Use Least Square method to find the transformation matrix with these 20 matches points.
   * Rectify and centralize the source image by 2D-affine transformation.
   * Draw the destination image and matching box correspond to both images.
   
   Note:
   You have to insatll the following packages with specific version before running this code:
   * pip install opencv-python==3.4.2.16
   * pip install opencv-contrib-python==3.4.2.16
   * pip install numpy
   * pip install scipy
   * pip install argparse
   @author Andy Lee
"""       

from scipy.signal import medfilt
from PIL import Image
from PIL.ExifTags import TAGS
from PIL.ExifTags import GPSTAGS
from math import tan, sin, cos, radians, sqrt
import os
import cv2
import argparse
import numpy as np
import time


def main(args):
    start = time.time()
    print("Processing the images: surf matching...\n")
    print(f"Step 1: finding image2 similar to {args.image1} in the given folder...")
    path_to_img1, path_to_img2 = findTwoImages(args)
    print("Step 2: preprocessing the images...")
    # resize img to 20%
    resize = 0.2
    img1 = cv2.resize(cv2.imread(path_to_img1), None, fx = resize, fy = resize, interpolation = cv2.INTER_CUBIC) 
    img2 = cv2.resize(cv2.imread(path_to_img2), None, fx = resize, fy = resize, interpolation = cv2.INTER_CUBIC) 
    img_blur1, img_blur2 = preprocess_img(img1, img2)
    print("Step 3: doing surf-matching...")
    kp1, kp2, best_matches = go_surf_matching(img_blur1, img_blur2)
    # Find all the 20 best matches points
    query_pts = np.float64([ kp1[m.queryIdx].pt for m in best_matches ]) 
    train_pts = np.float64([ kp2[m.trainIdx].pt for m in best_matches ]) 
    # Rectify and centralize the image by 2D affine transformation
    print("Step 4: saving affine transformation.jpg...")
    dst_img, dst_corner_pts = getAffineTransform(src_img=img1, pts1=query_pts, pts2=train_pts)
    cv2.imwrite("affine transformation.jpg", dst_img)
    print("Step 5: saving matches.jpg...")
    # Show both images with first 20 matches
    new_img = draw_matches(img1, kp1, img2, kp2, best_matches, dst_corner_pts)
    cv2.imwrite('matches.jpg', new_img)
    print(f"Step 6: saving {path_to_img2} to txt...\n")
    f = open("path_to_img2.txt", 'w')
    f.write(f"{path_to_img2}")
    f.close()
    stop = time.time()
    print(f'Execution Time: {(stop - start):.2f} seconds') 
    print("Process successful!")
    


def findTwoImages(args):
    """Find the filename of img1 and img2.
    img1: the given image from the first arg in cmd.
    img2: image that is most similar to img1 in the given folder from the second arg in cmd.
          (according to the UAV coordinates E N H)
    Return two images.
    """
    img1 = args.image1
    path = args.image2Folder
    e_image1, n_image1, h_image1 = get_twd97_coordinates(img1)
    images = os.listdir(path)

    dist_to_img1 = float("inf")
    img2 = ""
    for image in images:
        e_image2, n_image2, h_image2 = get_twd97_coordinates(path+image)
        # using euclidean distance to measure the similarity
        d = sqrt((e_image1-e_image2)**2 + (n_image1-n_image2)**2 + (h_image1-h_image2)**2)
        if d < dist_to_img1:
            dist_to_img1 = d
            img2 = image
    print(f"        found image2: {path+img2}")
    
    return img1, path+img2


def get_twd97_coordinates(filename):
    """Calculate the coordinate e n of given image
    Return E N coordinate
    """
    def get_geotagging(exif):
        if not exif:
            raise ValueError("No EXIF metadata found")

        geotagging = {}
        for (idx, tag) in TAGS.items():
            if tag == 'GPSInfo':
                if idx not in exif:
                    raise ValueError("No EXIF geotagging found")

                for (key, val) in GPSTAGS.items():
                    if key in exif[idx]:
                        geotagging[val] = exif[idx][key]

        return geotagging

    def wgs84_to_twd97(lat, lon):
        """Convert lat lon to twd97"""
        a = 6378137.0 # WGS84 Semi-major axis a(m)
        b = 6356752.314245 # WGS84 Semi-minor axis b(m)
        long0 = radians(121) # central meridian of zone
        k0 = 0.9999 # scale along central meridian of zone
        dx = 250000 # x offset(m)
        e = (1-b**2/(a**2))**0.5
        e2 = e**2/(1-e**2)
        n = (a-b)/(a+b)
        nu = a/(1-(e**2)*(sin(lat)**2))**0.5
        p = lon-long0
        A = a*(1 - n + (5/4.0)*(n**2 - n**3) + (81/64.0)*(n**4 - n**5))
        B = (3*a*n/2.0)*(1 - n + (7/8.0)*(n**2 - n**3) + (55/64.0)*(n**4 - n**5))
        C = (15*a*(n**2)/16.0)*(1 - n + (3/4.0)*(n**2 - n**3))
        D = (35*a*(n**3)/48.0)*(1 - n + (11/16.0)*(n**2 - n**3))
        E = (315*a*(n**4)/51.0)*(1 - n)
        S = A*lat - B*sin(2*lat) + C*sin(4*lat) - D*sin(6*lat) + E*sin(8*lat)
        K1 = S*k0
        K2 = k0*nu*sin(2*lat)/4.0
        K3 = (k0*nu*sin(lat)*(cos(lat)**3)/24.0) * (5 - tan(lat)**2 + 9*e2*(cos(lat)**2) + 4*(e2**2)*(cos(lat)**4))
        y = K1 + K2*(p**2) + K3*(p**4)
        K4 = k0*nu*cos(lat)
        K5 = (k0*nu*(cos(lat)**3)/6.0) * (1 - tan(lat)**2 + e2*(cos(lat)**2))
        x = K4*p + K5*(p**3) + dx

        return x, y

    def get_decimal_from_dms(dms, ref):
        degrees = dms[0]
        minutes = dms[1] / 60.0
        seconds = dms[2] / 3600.0

        if ref in ['S', 'W']:
            degrees = -degrees
            minutes = -minutes
            seconds = -seconds

        return degrees + minutes + seconds
    
    image = Image.open(filename)
    image.verify()
    exif = image._getexif()
    geotags = get_geotagging(exif)

    lat = get_decimal_from_dms(geotags['GPSLatitude'], geotags['GPSLatitudeRef'])
    lon = get_decimal_from_dms(geotags['GPSLongitude'], geotags['GPSLongitudeRef'])
    e, n = wgs84_to_twd97(radians(lat), radians(lon))
    _h = geotags['GPSAltitude']
    h = _h.numerator / _h.denominator

    return (e, n, h)


def preprocess_img(img1, img2): 
    """Return two blurred images"""
    img_hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img_hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    img_hsv1 = img_hsv1[:,:,2]  # Retain channel v only
    img_hsv2 = img_hsv2[:,:,2] 
    filter_size = (9, 9)
    # Blur the image by median filter
    img_blur1 = medfilt(volume = img_hsv1, kernel_size = filter_size).astype("uint8")
    img_blur2 = medfilt(volume = img_hsv2, kernel_size = filter_size).astype("uint8")

    return img_blur1, img_blur2


def go_surf_matching(img_blur1, img_blur2, n_best = 20):
    '''Find the best 20 matches by SURF
    Return keypoints of two images and the 20 best matches
    '''
    surf_dict = {}
    d_min = float("inf")
    for threshold in [400]:
        # Initiate SURF detector
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=threshold)

        # Find keypoints and descriptors with SURF
        kp1, des1 = surf.detectAndCompute(img_blur1,None)
        kp2, des2 = surf.detectAndCompute(img_blur2,None)

        # Create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors
        matches = bf.match(des1,des2)
        
        # Sort them in the order of their distance
        matches = sorted(matches, key = lambda x:x.distance)
        d = sum([m.distance for m in matches[:n_best]])
        if d < d_min:
            surf_dict["kp1"] = kp1
            surf_dict["kp2"] = kp2
            surf_dict["matches"] = matches[:n_best]

    return surf_dict["kp1"], surf_dict["kp2"], surf_dict["matches"]


def draw_matches(img1, kp1, img2, kp2, matches, dst_corner_pts, color=None): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.

        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.

        kp2: A list of cv2.KeyPoint objects for img2.

        matches: A list of DMatch objects whose trainIdx attribute refers to 
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.

        dst_corner_pts: Corner points transformed from the source image to 
        destination image, for drawing the matching box.

        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 10
    thickness = 2
    if color:
        c = color
    for idx, m in enumerate(matches):
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color: 
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
            c = tuple((int(c[0]), int(c[1]), int(c[2])))
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(np.float32(kp1[m.queryIdx].pt)).astype(int))
        end2 = tuple(np.round(np.float32(kp2[m.trainIdx].pt)).astype(int) + np.array([img1.shape[1], 0]))
        text = f"{idx+1}"
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.putText(new_img, text, end1, cv2.FONT_HERSHEY_PLAIN, 1.5, c, thickness, cv2.LINE_AA)
        cv2.circle(new_img, end2, r, c, thickness)
        cv2.putText(new_img, text, end2, cv2.FONT_HERSHEY_PLAIN, 1.5, c, thickness, cv2.LINE_AA)
        # Draw the red matching box
        pts = dst_corner_pts.astype(int) + np.array([img1.shape[1], 0])
        cv2.polylines(new_img, [pts], isClosed=True, color=(0, 0, 255), thickness=thickness+1)
    
    return new_img


def getAffineTransform(src_img, pts1:np.ndarray, pts2:np.ndarray):
    """Return the affine transformed image and the 
    4 corner points (transformed) used for drawing matching box
    """
    # secondary_system = A * primary_system + b
    primary = pts1
    secondary = pts2
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    X = pad(primary)
    Y = pad(secondary)
    
    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A 
    A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    A[np.abs(A) < 1e-10] = 0  # set really small values to zero
    trans = A.T[:2]   

    # Transfrom the 4 corners of the input image
    h, w = src_img.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = cv2.transform(np.array([src_pts]), trans)[0]
    min_x, max_x = np.min(dst_pts[:, 0]), np.max(dst_pts[:, 0])
    min_y, max_y = np.min(dst_pts[:, 1]), np.max(dst_pts[:, 1])

    # Width and height of destination image
    dst_w = int(max_x - min_x + 1) 
    dst_h = int(max_y - min_y + 1) 

    # Add translation to the transformation matrix to avoid cropping the image
    shifted_transf = trans + [[0, 0, -min_x], [0, 0, -min_y]]
    # Transform
    dst_img = cv2.warpAffine(src_img, shifted_transf, (dst_w, dst_h))

    return dst_img, dst_pts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image1", help="path to image1")
    parser.add_argument("image2Folder", help="path to folder of image2")
    args = parser.parse_args()
    main(args)
