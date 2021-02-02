# Stereo_Camera_Depth_Inference
 The Depth Inference with Stereo Camera and Disparity Map

For camera calibration:

    The data should be put into the '.\Images\CameraCaliberation\OriginalImages\'.

    Then running the CameraCaliberator.py file.

    Finally, the result of the camera calibration will be saved into '.\CameraCaliberation\CameraL' and '.\CameraCaliberation\CameraR'.

For disparity map and depth map:

    If running with our own dataset.

    The data should be put into the '.\Images\DepthMap\OriginalImages'.

    Then running the DMGenerator.py file.

    Finally, the result of the disparity map and depth map will be saved into '.\Images\DepthMap'.

If running with Middelbury dataset.

The data should be put into the '.\Images\DepthMap\CameraL' and '.\Images\DepthMap\CameraR'.

Then running the DMGenerator.py file.

Finally, the result of the disparity map and depth map will be saved into '.\Images\DepthMap'.

Also, the parameters of the disparity map and depth map for middlebury dataset are in the comments of the file DepthMapGenerator.py.

And the line 112 of the DMGenerator.py should be replaced by depthMaps = DMG.DepthMap(imgsL, KL, KR, 191.003).

For combination of the yolo.

The yolo model should be put into the root directory of the whole project.

After running the DMGenerator.py file with our own dataset, running the YoLov4.py file, and the result will be saved into'.\Images\YoLoPrediction'


The linking of the datasets is:

https://pan.baidu.com/s/1fOfxIWumA1DAWe5SpHwDpQ

Extract Code: opcv

The linking of the trained yolo model is:

https://pan.baidu.com/s/1uC75i23s7I_VyLcffFZ_sg

Extract Code: opcv