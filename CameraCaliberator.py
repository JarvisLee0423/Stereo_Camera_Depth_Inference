'''
    Copyright:  JarvisLee
    Date:       2020/19/12
    File Name:  CameraCaliberator.py
'''

# Importing the necessary library.
import cv2 as cv
import os
from Utils.ImageShow import imshow as imshow
from Utils.ImageCrop import imcrop as imcrop
from CameraCaliberation.CameraCaliberation import CameraCaliberation as CC

# Caliberating the camera.
if __name__ == "__main__":
    # Croppting the images.
    imcrop('./Images/CameraCaliberation/')
    # Creating the paths.
    if not os.path.exists('./CameraCaliberation/CameraR/'):
        os.mkdir('./CameraCaliberation/CameraR/')
    if not os.path.exists('./CameraCaliberation/CameraL/'):
        os.mkdir('./CameraCaliberation/CameraL/')
    # Getting the images for caliberating the right camera.
    imgsR = [cv.imread('./Images/CameraCaliberation/CameraR/' + i) for i in os.listdir('./Images/CameraCaliberation/CameraR')]
    # Getting the corner information.
    imgsR, pts3d, pts2d, size = CC.GetCorner(imgsR, (7, 6))
    # Showing the images.
    imshow([cv.cvtColor(img, cv.COLOR_BGR2RGB) for img in imgsR], nrows = 4, ncols = 3, figsize = (30, 30))
    # Getting the right camera parameters. 
    ret, K, dist, R, T = CC.Caliberate(pts3d, pts2d, size)
    # Saving the parameters.
    file = open('./CameraCaliberation/CameraR/IntrinsicMatrix.txt', 'w')
    file.write(str(K))
    file.close()
    file = open('./CameraCaliberation/CameraR/RotationMatrix.txt', 'w')
    file.write(str(R))
    file.close()
    file = open('./CameraCaliberation/CameraR/TranslationMatrix.txt', 'w')
    file.write(str(T))
    file.close()
    file = open('./CameraCaliberation/CameraR/DistortionCoefficient.txt', 'w')
    file.write(str(dist))
    file.close()
    # Getting the images for caliberating the left camera.
    imgsL = [cv.imread('./Images/CameraCaliberation/CameraL/' + i) for i in os.listdir('./Images/CameraCaliberation/CameraL')]
    # Getting the corner information.
    imgsL, pts3d, pts2d, size = CC.GetCorner(imgsL)
    # Showing the images.
    imshow([cv.cvtColor(img, cv.COLOR_BGR2RGB) for img in imgsL], nrows = 4, ncols = 3, figsize = (30, 30))
    # Getting the left camera parameters. 
    ret, K, dist, R, T = CC.Caliberate(pts3d, pts2d, size)
    # Saving the parameters.
    file = open('./CameraCaliberation/CameraL/IntrinsicMatrix.txt', 'w')
    file.write(str(K))
    file.close()
    file = open('./CameraCaliberation/CameraL/RotationMatrix.txt', 'w')
    file.write(str(R))
    file.close()
    file = open('./CameraCaliberation/CameraL/TranslationMatrix.txt', 'w')
    file.write(str(T))
    file.close()
    file = open('./CameraCaliberation/CameraL/DistortionCoefficient.txt', 'w')
    file.write(str(dist))
    file.close()