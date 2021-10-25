main.py
============================

Use SURF to do surf matching with UAV image and orthophoto. Then rotate the UAV image to north.

### Usage
1. Enter UAV image name (required).

`$ python main.py ./test_uav_img/P1482162.jpg`

2. Enter TWD97 coordinate x,y of UAV image (optional) if image doesn’t contain any geoinfo.

`python main.py ./test_uav_img/P1482162.jpg --coord 162615.934, 2550191.182`

3. Enter the output directory (default=“./outputDir”)

`python main.py ./test_uav_img/P1482162.jpg --outputDir ./outputFolder`

### Output
    .
    ├── outputDir						         # default output folder
        ├── bigtile_<UavImgName>.png             # big tile 
        ├── uav_bigtile_<UavImgName>.png         # big tile correspond to UAV img1
        └── rotate_north_<UavImgName>.png        # Rotated UAV image1

### Folder structure
    .
    ├── main.py
    ├── tiles20XY.csv               # coordinates of tiles         
    ├── test_uav_img                # folder of test uav images
    │   ├── P1482162.JPG            # UAV image
    │   └── ...                     # etc.
    ├── 20210902                    # folder of total UAV images
    │   ├── P1451904.JPG            # UAV image2
    │   └── ...                     # etc.
    └── tiles                       # folder of tiles
        └── 20                      # level 20
            └── ...   