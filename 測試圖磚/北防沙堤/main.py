from scipy.signal import medfilt
from PIL import Image
from PIL.ExifTags import TAGS
from PIL.ExifTags import GPSTAGS
import sys
import cv2
import argparse
import numpy as np
import time
import pandas as pd
import math
import os
from pyproj import Transformer
from datetime import datetime


def main(args):   
    start = time.time()
    print("Processing the images: surf matching...\n")
    print("Step 1: finding big tile and save as bigtile.png and uav_bigtile.png...")
    draw_uav_and_tile(args)
    print("Step 2: finding UAV img2...")
    path_to_img1, path_to_img2 = args.UAV_img1, get_uav_img2(args.UAV_img1)

    print("Step 3: preprocessing two images...")
    # Both UAV images resize 20%
    img1 = cv2.resize(cv2.imread(path_to_img1), None, fx = 0.2, fy = 0.2, interpolation = cv2.INTER_CUBIC)
    img2 = cv2.resize(cv2.imread(path_to_img2), None, fx = 0.2, fy = 0.2, interpolation = cv2.INTER_CUBIC) 
    img_blur1, img_blur2 = preprocess_img(img1, img2)

    print("Step 4: doing surf-matching...")
    # Find all the 20 best matches points
    kp1, kp2, best_matches = go_surf_matching(img_blur1, img_blur2, n_best=20)
    query_pts = np.float64([ kp1[m.queryIdx].pt for m in best_matches ])
    train_pts = np.float64([ kp2[m.trainIdx].pt for m in best_matches ])

    print("Step 5: Rotate UAV img to north and save as rotate_north.png...\n")
    # 2D affine transformation with RANSAC
    idx6 = get_indexes_with_ransac(pts1=query_pts, pts2=train_pts)
    save_rotate_img(path_to_img1, path_to_img2, query_pts[idx6], train_pts[idx6])
    stop = time.time()
    print(f'Execution Time: {(stop - start):.2f} seconds') 
    print("Process successful!")


def get_bigtile_img(args):
    """Find the level20 tile image which is closest to UAV img1.
    Then concat its 8 neighbors to a big tile.
    Return the big tile.
    """

    def concat_tiles(tiles9_sorted):
        tiles9_sorted["ImagePath"] = tiles9_sorted[['Level', 'FolderNum', "Image"]].astype(str).agg('/'.join, axis=1).values
        tiles_list = tiles9_sorted['ImagePath'].tolist()
        tiles_list = np.reshape(tiles_list,(3,3))
        dirName = "./tiles/"
        
        return cv2.vconcat([cv2.hconcat(list(map(lambda x: cv2.imread(f"{dirName}{x}"),tiles_list_h))) for tiles_list_h in tiles_list])

    # Read the tiles coordinates
    tiles = pd.read_csv("tiles20XY.csv", delim_whitespace=True)
    
    # Compute TWD97 coordinates of UAV img1
    if len(args.coord) == 0:
        e_img1, n_img1 = get_twd97_coordinates(args.UAV_img1)[:2]
    elif len(args.coord) == 2:
        e_img1, n_img1 = float(args.coord[0]), float(args.coord[1])
    else:
        print("Error in UAV img coordinates, please check again.")
        sys.exit(1)

    # Find the tile image which is closest to UAV img1
    tiles["X"] = e_img1
    tiles["Y"] = n_img1
    tiles["Dist"] = ((tiles["TilesX"]-tiles["X"])**2 + (tiles["TilesY"]-tiles["Y"])**2)**0.5
    center_tile = tiles[tiles.Dist == tiles.Dist.min()]

    # GSD is 0.14929 m/pixel in level20 according to 
    # https://developer.tomtom.com/maps-api/maps-api-documentation/zoom-levels-and-tile-grid?fbclid=IwAR2FBZOvzSh_o9_wpREeFJw5Uuh9ju2V8XXOhF3MaOloEjyBXXcAL8zQT8c
    gsd = 0.14929
    pixel = 256
    interval = gsd * pixel
    tiles9 = tiles[(tiles["TilesX"] >= center_tile["TilesX"].values[0] - interval) & (tiles["TilesX"] <= center_tile["TilesX"].values[0] + interval) & \
                    (tiles["TilesY"] >= center_tile["TilesY"].values[0] - interval) & (tiles["TilesY"] <= center_tile["TilesY"].values[0] + interval)]
    tiles9_sorted = tiles9.sort_values(["TilesY", "TilesX"], ascending=[False, True])
    big_tile = concat_tiles(tiles9_sorted)
    
    return big_tile


def get_uav_img2(img1):
    """Find the filename of uav img2.
    img1: uav image from the first arg in cmd.
    img2: uav image which timestamp is later (or earlier if none) than uav img1.
    """
    path_to_img2 = "./20210902/"
    time1 = get_timestamp(img1)
    dtime_pos, dtime_neg = float("inf"), -float("inf")
    later, earlier = "", ""
    
    for img in os.listdir(path_to_img2):
        time2 = get_timestamp(path_to_img2+img)
        dtime = time2 - time1

        # timestamp is later than img1
        if dtime < dtime_pos and dtime > 0:
            dtime_pos = dtime
            later = img
        # timestamp is earlier than img1
        if dtime > dtime_neg and dtime < 0:
            dtime_neg = dtime
            earlier = img
    img2 = path_to_img2 + later if later != "" else path_to_img2 + earlier
        
    # print(f"UAV img1: {img1}")
    # print(f"UAV img2: {img2}")
    
    return img2


def draw_uav_and_tile(args): 
    """Find the big tile from UAV img1.
    Draw two images and place them side by side in a new image.
    Save the new image and big tile."""
    big_tile = get_bigtile_img(args)
    cv2.imwrite("./bigtile.png", big_tile)
    img1 = cv2.resize(cv2.imread(args.UAV_img1), None, fx = 0.2, fy = 0.2, interpolation = cv2.INTER_CUBIC)
    img2 = big_tile

    # We're drawing them side by side. Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2

    cv2.imwrite('uav_bigtile.png', new_img)



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
    # Initiate SURF detector
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=300)

    # Find keypoints and descriptors with SURF
    kp1, des1 = surf.detectAndCompute(img_blur1,None)
    kp2, des2 = surf.detectAndCompute(img_blur2,None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1,des2)
    
    # Sort them in the order of their distance
    matches = sorted(matches, key = lambda x:x.distance)

    return kp1, kp2, matches[:n_best]


def get_timestamp(filename):
    """Return the datetime(timestamp) of UAV image"""
    image = Image.open(filename)
    image.verify()
    exif = image._getexif()
    time_str = exif.get(36867)
    time = list(map(int, ":".join(time_str.split(" ")).split(":")))
    
    return datetime(*time).timestamp()
    

def get_twd97_coordinates(filename):
    """Calculate the coordinate e n h of given UAV image
    Return E N H coordinate
    """
    def get_geotagging(exif):
        if not exif:
            raise ValueError("No EXIF metadata found. Please enter coordinate of UAV image.")

        geotagging = {}
        for (idx, tag) in TAGS.items():
            if tag == 'GPSInfo':
                if idx not in exif:
                    raise ValueError("No EXIF geotagging found. Please enter coordinate of UAV image.")

                for (key, val) in GPSTAGS.items():
                    if key in exif[idx]:
                        geotagging[val] = exif[idx][key]

        return geotagging

    def wgs84_to_twd97(lat_degree, lon_degree):
        """Convert lat lon to twd97"""
        # from WGS84 to TWD97
        transformer = Transformer.from_crs("epsg:4326", "epsg:3826")
        x, y = transformer.transform(lat_degree, lon_degree)
        x, y = round(x, 3), round(y, 3)
        
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
    
    try:
        image = Image.open(filename)
        image.verify()
        exif = image._getexif()
        geotags = get_geotagging(exif)
    except Exception as e:
        print("No EXIF metadata found. Please enter coordinate of UAV image.")
        sys.exit(1)
        
    lat = get_decimal_from_dms(geotags['GPSLatitude'], geotags['GPSLatitudeRef'])
    lon = get_decimal_from_dms(geotags['GPSLongitude'], geotags['GPSLongitudeRef'])
    e, n = wgs84_to_twd97(lat, lon)
    _h = geotags['GPSAltitude']
    h = _h.numerator / _h.denominator

    return (e, n, h)

def get_indexes_with_ransac(pts1:np.ndarray, pts2:np.ndarray):
    """Using matching points to calculate affine transformation matrix.
    Return indexes of selected points with RANSAC"""
    # Get affine transformation using RANSAC
    idx6 = np.where(cv2.estimateAffine2D(pts1, pts2)[1].reshape(20) == 1)

    return idx6

def save_rotate_img(img1, img2, pts1, pts2):
    """Use matching points selected by RANSAC of uav1 and uav2,
    and coordinates of uav1 and uav2 to get rotation angle.
    Args:
        img1: Filename of UAV img1.
        img2: Filename of UAV img2.
        pts1: Matching points selected by RANSAC of uav1
        pts2: Matching points selected by RANSAC of uav2
    Save the rotated img1."""
    # Coordinates of uav1 and uav2
    coord1, coord2 = get_twd97_coordinates(img1), get_twd97_coordinates(img2)
    dx, dy = (coord2[0] - coord1[0]), (coord2[1] - coord1[1])
    theta = math.atan2(dy, dx)

    # Mean of matching points coordinates of uav1 and uav2
    mpc1_bar, mpc2_bar = np.mean(pts1, axis=0), np.mean(pts2, axis=0)
    c1_bar, r1_bar = mpc1_bar[0], mpc1_bar[1]
    c2_bar, r2_bar = mpc2_bar[0], mpc2_bar[1]
    dc_bar, dr_bar = (c2_bar - c1_bar), (r2_bar - r1_bar)
    theta_prime = math.atan2(dr_bar, dc_bar)
    dtheta = theta - theta_prime

    # Rotation matrix
    rotation_mat = np.array([[math.cos(dtheta), math.sin(dtheta), 0],
                             [-math.sin(dtheta), math.cos(dtheta), 0]])
    
    # Transform the 4 corners of img1
    src_img = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2BGRA)
    h, w = src_img.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_corner_pts = cv2.transform(np.array([src_pts]), rotation_mat)[0]
    min_x, max_x = np.min(dst_corner_pts[:, 0]), np.max(dst_corner_pts[:, 0])
    min_y, max_y = np.min(dst_corner_pts[:, 1]), np.max(dst_corner_pts[:, 1])

    # Width and height of destination image
    dst_w = int(max_x - min_x + 1) 
    dst_h = int(max_y - min_y + 1) 

    # Add translation to the transformation matrix to avoid cropping the image
    shifted_transf = rotation_mat + [[0, 0, -min_x], [0, 0, -min_y]]

    # Rotate image1 to north
    north_img = cv2.warpAffine(src_img, shifted_transf, (dst_w, dst_h), borderMode=cv2.BORDER_TRANSPARENT)
    cv2.imwrite(f"rotate_north.png", north_img)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("UAV_img1", help="path to UAV image")
    parser.add_argument("--coord", help="enter TWD97 X Y coordinate UAV image", nargs="+", default=[])
    args = parser.parse_args()
    main(args)