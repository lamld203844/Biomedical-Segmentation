# About dataset (Pubic Symphysis-Fetal Head Segmentation)[https://ps-fh-aop-2023.grand-challenge.org/]

20190916T104520, 20190916T105526... and other 78 folders are all screenshots extracted from ultrasound videos.
It contains multiple subfolders, and the name of each folder corresponds to the name of the original video.

File structure, for example:
- 20190909T155747 (including all screenshots and corresponding annotations extracted from the video named "20190909T155747")
- image (Screenshot extracted from the video, naming rule: "video name" + "_frame sequence number".png)
- mask (Save the mask of the corresponding image in the image folder, naming rules: "video name" + "_frame number" + "_mask".png)
- mask_enhance (the folder after image processing, red represents SP, green represents Head)
- framelabel.csv (The frame number and corresponding frame label extracted from the video. frame_id is the frame number, frame_label is the frame label)


Note:
1. The images in the image folder have undergone basic preprocessing. After cropping and overwriting operations, the original interface toolbar and text information in the image are removed. The processed size is 1295*1026. In order to prevent information loss, the downsampling process is not performed. At the same time, it has been converted into a grayscale image, which can be directly read in grayscale format when using it.
2. The image in the mask folder may look completely black due to the low label value (pixel value 7-SP, 8-Head).
3. Frame_label frame label 3-None, 4-OnlySP, 5-OnlyHead, 6-SP+Head in framelabel.csv.

**78Video, (6254 image, 6247 mask, 6224 maske enhance)(frame), 51patients**
