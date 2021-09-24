surf_matching_hsv.py
============================

### 程式用途
>找出最佳UAV影像並利用SURF演算法進行影像匹配

### 第三方函式庫

>opencv-python==3.4.2.16
opencv-contrib-python==3.4.2.16
numpy
scipy
argparse

### 操作說明
>輸入UAV影像1、UAV影像2資料夾的位置 /path/to/img 
### 輸入範例(CMD)
```shell
$ python .\surf_matching_hsv.py .\P0770319.JPG .\UAV_image2\
```
    .
    ├── surf_matching_hsv.py       
    ├── P0770319.JPG                # image1
    ├── UAV_image2                  # folder of candidate images
    │   ├── P0770320.JPG            # image2
    │   └── ...                     # etc.
    └── README.md
### 輸出
    .
    ├── ...
    ├── affine transformation.JPG               
    ├── matches.JPG                  
    └── path_to_img2.txt             
>1. affine transformation.JPG為經過匹配及仿射轉換的影像。
>2. matches.JPG為影像匹配結果，左圖為影像1，右圖為影像2。圖中包含20組最佳匹配點位置(圓圈表示)，並以二維仿射轉換計算，在圖上標示轉換後套疊位置，用紅色線標示該套疊圖框。
>3. path_to_img2.txt為影像2的路徑