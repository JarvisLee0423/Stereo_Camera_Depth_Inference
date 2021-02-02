'''
    Copyright:  JarvisLee
    Date:       2020/12/19
    File Name:  DepthMapGenerator.py
'''

# Importing the necessary library.
import cv2 as cv
import numpy as np
from Utils.ImageCrop import imcrop as imcrop
from Utils.ImageShow import imshow as imshow

# Setting the class for generating the disparity map.
class DepthMapGenerator():
    '''
        This class is used to generating the depth map.
    '''
    # Setting the function to fill the hole of the disparity image.
    @staticmethod
    def HoleFiller(img):
        '''
            This function is used to fill the image.
            Params Description:
                - 'img' is the data of the image.
        '''
        # Getting the image size.
        w = img.shape[1]
        h = img.shape[0]
        # Initializing the integral map.
        integral = np.zeros((h, w), np.float)
        ptsIntegral = np.zeros((h, w), np.float)
        # Computing the gradients.
        for i in range(0, h):
            for j in range(0, w):
                if img[i][j] > 1e-3:
                    integral[i][j] = img[i][j]
                    ptsIntegral[i][j] = 1
        # Computing the calculus interval.
        for i in range(0, h):
            for j in range(1, w):
                integral[i][j] = integral[i][j] + integral[i][j - 1]
                ptsIntegral[i][j] = ptsIntegral[i][j] + ptsIntegral[i][j - 1]
        for i in range(1, h):
            for j in range(0, w):
                integral[i][j] = integral[i][j] + integral[i][j - w]
                ptsIntegral[i][j] = ptsIntegral[i][j] + ptsIntegral[i][j - w]
        # Computing the real depth image.
        dWnd = 2
        while dWnd > 1:
            wnd = int(dWnd)
            dWnd = dWnd / 2
            for i in range(0, h):
                for j in range(0, w):
                    left = j - wnd - 1
                    right = j + wnd
                    top = i - wnd - 1
                    bottom = i + wnd
                    left = max(0, left)
                    right = min(right, w - 1)
                    top = max(0, top)
                    bottom = min(bottom, h - 1)
                    dx = right - left
                    dy = bottom - top
                    ptsCnt = ptsIntegral[top + dy][left + dx] + ptsIntegral[top][left] - (ptsIntegral[top + dy][left] + ptsIntegral[top][left + dx])
                    sumGray = integral[top + dy][left + dx] + integral[top][left] - (integral[top + dy][left] + integral[top][left + dx])
                    if ptsCnt <= 0:
                        continue
                    img[i][j] = float(sumGray / ptsCnt)
            s = int(wnd / 2 * 2 + 1)
            if s > 201:
                s = 201
            img = cv.GaussianBlur(img, (s, s), s, s)
        # Returning the disparity image.
        return img
    # Setting the function to do the image preprocessing.
    @staticmethod
    def Preprocessor(imgsR, imgsL):
        '''
            This function is used to preprocess the image and generating the cost. \n
            Params Description:
                - 'imgsR' is the list of all right images.
                - 'imgsL' is the list of all left images.
        '''
        # Setting the histograms list.
        histsR = []
        histsL = []
        for i in range(len(imgsR)):
            # Converting the images to be the gray image.
            imgR = cv.cvtColor(imgsR[i], cv.COLOR_BGR2GRAY)
            imgL = cv.cvtColor(imgsL[i], cv.COLOR_BGR2GRAY)
            # Creating the histogram.
            histsR.append(cv.equalizeHist(imgR))
            histsL.append(cv.equalizeHist(imgL))
            #histsR.append(imgR)
            #histsL.append(imgL)
        # Returning the histograms.
        return histsR, histsL
    # Setting the function to computing the disparity map.
    @staticmethod
    def SGBM(imgsR, imgsL, minDisparity, numDisparities, blockSize):
        '''
            This function is used to getting the disparity map. \n
            Params Description:
                - 'imgsR' is the image from right camera.
                - 'imgsL' is the image from left camera.
                - 'minDisparity' is the accepted minmum disparity.
                - 'numDisparities' is the number of the different disparities should be computed.
                - 'blockSize' is the window size of the BM agorithm.
        '''
        # Getting the parameters of the SGBM.
        params = {
            'minDisparity' : minDisparity,
            'numDisparities' : numDisparities,
            'blockSize' : blockSize,
            'P1' : 8 * blockSize ** 2,
            'P2' : 32 * blockSize ** 2,
            'disp12MaxDiff' : 1,
            'preFilterCap' : 63,
            'uniquenessRatio' : 10,
            'speckleWindowSize' : 100,
            'speckleRange' : 32,
            'mode' : cv.StereoSGBM_MODE_SGBM
        }
        # Creating the SGBM object.
        sgbm = cv.StereoSGBM_create(**params)
        # Getting two list to store the disparity images.
        dispsR = []
        dispsL = []
        # Computing the disparity maps.
        for i in range(len(imgsR)):
            dispR = sgbm.compute(imgsR[i], imgsL[i])
            dispL = sgbm.compute(imgsL[i], imgsR[i])
            dispR = np.divide(dispR.astype(np.float), 16.)
            dispL = np.divide(dispL.astype(np.float), 16.)
            dispR = cv.normalize(dispR, dispR, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_16UC1)
            dispL = cv.normalize(dispL, dispL, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_16UC1)
            dispsR.append(dispR)
            dispsL.append(dispL)
        # Returning the disparity maps.
        return dispsR, dispsL
    # Setting the function to computing the depth map.
    @staticmethod
    def DepthMap(imgs, KL, KR, baseline):
        '''
            This function is used to computing the depth map.
            Params Description:
                - 'imgs' is the list of the image data.
                - 'K' is the intrinsic matrix of the input images.
                - 'baseline' is the baseline of the stereo camera.
        '''
        # Getting the focal length.
        fx = KL[0][0]#3997.684
        fy = KL[1][1]#3997.684
        cx = KL[0][2]#1307.839
        cy = KR[0][2]#1176.728
        # Initializing the depth map list.
        depthMaps = []
        # Computing the depth maps.
        for i in range(len(imgs)):
            # Getting the image size.
            h = imgs[i].shape[0]
            w = imgs[i].shape[1]
            # Initializing the depth map.
            depthMap = np.zeros((h, w))
            # Creating the depth map.
            for k in range(0, h):
                for j in range(0, w):
                    if imgs[i][k][j] == 0:
                        continue
                    depthMap[k][j] = (fx + fy) * baseline / (imgs[i][k][j] + abs(cx - cy))
            #np.set_printoptions(threshold = np.inf)
            #print(depthMap)
            # Storing the depth map.
            depthMaps.append(depthMap)
        # Returning the depth map.
        return depthMaps