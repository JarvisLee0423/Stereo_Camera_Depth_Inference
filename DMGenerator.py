'''
    Copyright:  JarvisLee
    Date:       2020/12/19
    File Name:  DMGenerator.py
'''

# Importing the necessary library.
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from Utils.ImageCrop import imcrop as imcrop
from Utils.ImageShow import imshow as imshow
from DepthMap.DepthMapGenerator import DepthMapGenerator as DMG
from CameraCaliberation.CameraCaliberation import CameraCaliberation as CC

# Setting the main function to generate the depth map.
if __name__ == "__main__":
    # Getting the roots.
    root = './Images/DepthMap/'
    # Creating the directory for disparity map.
    if not os.path.exists(root + '/DispsL/'):
        os.mkdir(root + '/DispsL/')
    if not os.path.exists(root + '/DispsR/'):
        os.mkdir(root + '/DispsR/')
    # Creating the directory for depth map.
    if not os.path.exists(root + '/DepthL/'):
        os.mkdir(root + '/DepthL/')
    # Getting the intrinsic matrix.
    file = open('./CameraCaliberation/CameraR/IntrinsicMatrix.txt', 'r')
    temp = []
    for line in file.readlines():
        temp.append(','.join(line.split('\n')[0].split(' ')))
    KR = np.array(eval(''.join(temp)))
    #print(KR, KR.shape, type(KR))
    file = open('./CameraCaliberation/CameraL/IntrinsicMatrix.txt', 'r')
    temp = []
    for line in file.readlines():
        temp.append(','.join(line.split('\n')[0].split(' ')))
    KL = np.array(eval(''.join(temp)))
    #print(KL, KL.shape, type(KL))
    # Getting the distortion coefficient.
    file = open('./CameraCaliberation/CameraR/DistortionCoefficient.txt', 'r')
    temp = []
    line = file.readline()
    temp.append(','.join([line.split('\n')[0].split(' ')[i] for i in range(4)]))
    line = file.readline()
    temp.append(line.split('\n')[0].split(' ')[-1])
    distR = np.array(eval(','.join(temp)))
    #print(distR, distR.shape, type(distR))
    file = open('./CameraCaliberation/CameraL/DistortionCoefficient.txt', 'r')
    temp = []
    line = file.readline()
    temp.append(','.join(line.split('\n')[0].split(' ')))
    line = file.readline()
    temp.append(line.split('\n')[0].split(' ')[-1])
    distL = np.array(eval(','.join(temp)))[0]
    #print(distL, distL.shape, type(distL))
    file.close()
    # Getting all the right images.
    if os.path.exists(root + '/CameraR/'):
        # Getting the titles.
        titles = os.listdir(root + '/CameraR/')
        #print(titles)
        # Getting all the right images.
        imgsR = [cv.imread(root + '/CameraR/' + i) for i in titles]
    else:
        # Cropping the images.
        imcrop(root)
        # Getting the titles.
        titles = os.listdir(root + '/CameraR/')
        #print(titles)
        # Getting all the right images.
        imgsR = [cv.imread(root + '/CameraR/' + i) for i in titles]
    # Getting all the left images.
    if os.path.exists(root + '/CameraL/'):
        # Getting the titles.
        titles = os.listdir(root + '/CameraL/')
        #print(titles)
        # Getting all the left images.
        imgsL = [cv.imread(root + '/CameraL/' + i) for i in titles]
    else:
        # Cropping the images.
        imcrop(root)
        # Getting the titles.
        titles = os.listdir(root + '/CameraL/')
        #print(titles)
        # Getting all the left images.
        imgsL = [cv.imread(root + '/CameraL/' + i) for i in titles]
    # Rectifying the images.
    imgsR, imgsL = CC.ImageRectify(imgsR, imgsL, KR, KL, distR, distL)
    # for i in range(len(imgsR)):
    #     imshow([cv.cvtColor(imgsR[i], cv.COLOR_BGR2RGB), cv.cvtColor(imgsL[i], cv.COLOR_BGR2RGB)], ncols = 2)
    # Preprocessing the images.
    imgsR, imgsL = DMG.Preprocessor(imgsR, imgsL)
    # for i in range(len(imgsR)):
    #     imshow([imgsR[i], imgsL[i]], ncols = 2)
    # Getting the disparity map.
    imgsR, imgsL = DMG.SGBM(imgsR, imgsL, 0, int((imgsR[0].shape[1] / 8) + 15) & -16 , 9)
    # Filling the images.
    for i in range(len(imgsR)):
        imgsL[i] = DMG.HoleFiller(imgsL[i])
        imgsR[i] = DMG.HoleFiller(imgsR[i])
    # Showing and storing the disparity map.
    for i in range(len(imgsR)):
        #imshow([imgsR[i], imgsL[i]], ncols = 2)
        cv.imwrite(root + f'/DispsL/ImageL0{i + 1}.jpg', imgsL[i])
        cv.imwrite(root + f'/DispsR/ImageR0{i + 1}.jpg', imgsR[i])
    # Getting the left disparity map.
    titles = os.listdir(root + '/DispsL/')
    # Creating the depth map.
    depthMaps = DMG.DepthMap(imgsL, KL, KR, 68)
    # Showing and storing the depth map.
    for i in range(len(depthMaps)):
        # print(depthMaps[i].shape)
        # print(depthMaps[i])
        #depthMaps[i] = cv.normalize(depthMaps[i], depthMaps[i], alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_16UC1)
        # print(depthMaps[i])
        #plt.imshow(depthMaps[i], cmap = 'gray')
        plt.show()
        plt.imsave(root + f'/DepthL/ImageL0{i + 1}.jpg', depthMaps[i])