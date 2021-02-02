'''
    Copyright:  JarvisLee
    Data:       2020/12/19
    File Name:  CameraCaliberation.py
'''

# Importing the necessary library.
import cv2 as cv
import numpy as np
import os

# Creating the class for camera caliberation.
class CameraCaliberation():
    '''
        This class is used to set the tools for caliberating the camera.
    '''
    # Setting the method to get the corner information.
    @staticmethod
    def GetCorner(imgs, bdsize = (7, 6), visible = True, maxIter = 30, minEps = 0.001):
        '''
            This function is used to prepare the information for camera caliberation. \n
            Params Description:
                - 'imgs' is the list of all the data of the images.
                - 'bdsize' is the size of the caliberating board.
                - 'visible' is the boolean value to test whether showing the images.
                - 'maxIter' is the number of the iteration for getting corner information.
                - 'minEps' is the hyperparamter of the getting corner algorithm.
        '''
        # Setting the list to store the corner information.
        cornerImg = []
        pts2d = []
        pts3d = []
        # Getting the corner information.
        for i in range(len(imgs)):
            # Changing the image into gray mode.
            grayImg = cv.cvtColor(imgs[i], cv.COLOR_BGR2GRAY)
            # Getting the corner information.
            ret, corner = cv.findChessboardCorners(grayImg, bdsize)
            # Termination criteria for sub pixel corners refinement.
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, maxIter, minEps)
            # Drawing the corner information.
            if ret:
                cv.cornerSubPix(grayImg, corner, bdsize, (-1, -1), criteria)
                if visible:
                    cv.drawChessboardCorners(imgs[i], bdsize, corner, ret)
            # Getting the corner image.
            cornerImg.append(imgs[i])
            # Getting the 2D information.
            pts2d.append(corner.reshape(-1, 2).astype(np.float32))
            # Getting the 3D information.
            tmp3d = np.zeros((bdsize[0] * bdsize[1], 3))
            tmp3d[:, :2] = np.mgrid[0:bdsize[0], 0:bdsize[1]].T.reshape(-1, 2)
            pts3d.append(tmp3d.astype(np.float32) * 10)
        # Returning the data.
        return imgs, pts3d, pts2d, cornerImg[0].shape[:-1][::-1]
    # Setting the method to caliberate the camera.
    @staticmethod
    def Caliberate(objectPoints, imagePoints, size, cameraMatrix = None, distortion = None):
        '''
            This function is used to caliberate the camera. \n
            Params Description:
                - 'object' is the object points.
                - 'image' is the image points.
                - 'size' is the image size.
                - 'cameraMatrix' is the camera matrix.
                - 'distortion' is the distortion coefficients.
        '''
        # Caliberating the camera.
        ret, K, dist, R, T = cv.calibrateCamera(objectPoints = objectPoints, imagePoints = imagePoints, imageSize = size, cameraMatrix = cameraMatrix, distCoeffs = distortion)
        # Computing the total error.
        totalError = 0
        for i in range(len(objectPoints)):
            pst2dReproj, _ = cv.projectPoints(objectPoints[i], R[i], T[i], np.array(K), dist)
            error = cv.norm(imagePoints[i], pst2dReproj.reshape(-1, 2), cv.NORM_L2) / len(pst2dReproj)
            totalError += error
        # Printing the error.
        print(f'The reprojection error is {totalError / len(objectPoints)}')
        # Returning the caliberating information.
        return ret, K, dist, R, T
    # Setting the method to undistort the image.
    @staticmethod
    def ImageRectify(imgsR, imgsL, KR, KL, distR, distL):
        '''
            This function is used to rectify the images. \n
            Params Description:
                - 'imgsR' is the list for all the data of right image.
                - 'imgsL' is the list for all the data of left image.
                - 'KR' is the intrinsic matrix of the right camera.
                - 'KL' is the intrinsic matrix of the left camera.
                - 'distR' is the distortion coefficient of the right camera.
                - 'distL' is the distortion coefficient of the left camera.
        '''
        # Rectifying the images.
        rectifyedImgsR = [cv.undistort(imgs, KR, distR) for imgs in imgsR]
        rectifyedImgsL = [cv.undistort(imgs, KL, distL) for imgs in imgsL]
        # Returning the rectifyed image.
        return rectifyedImgsR, rectifyedImgsL