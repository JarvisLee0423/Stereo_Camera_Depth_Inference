'''
    Copyright:  JarvisLee
    Date:       2020/12/19
    File Name:  ImageCrop.py
'''

# Importing the necessary library.
import cv2 as cv
import os
from Utils.ImageShow import imshow as imshow

# The function which is used to crop the image into two part.
def imcrop(root):
    '''
        This function is used to crop the images. \n
        Params Description:
            - 'root' is the root directory of the images.
    '''
    # Checking the root.
    assert os.path.exists(root), 'Please input the correct root!!!'
    # Setting the paths.
    original = root + '/OriginalImages/'
    cameraR = root + '/CameraR/'
    cameraL = root + '/CameraL/'
    # Checking the save directory.
    if os.path.exists(cameraL):
        # Removing all the exists images.
        for i in os.listdir(cameraL):
            os.remove(cameraL + i)
    else:
        # Creating the save directory.
        os.mkdir(cameraL)
    if os.path.exists(cameraR):
        # Removing all the exists images.
        for i in os.listdir(cameraR):
            os.remove(cameraR + i)
    else:
        # Creating the save directory.
        os.mkdir(cameraR)
    # Getting all the files inside the root.
    titles = os.listdir(original)
    # Getting all the images.
    for i, file in enumerate(titles):
        # Reading the image.
        img = cv.cvtColor(cv.imread(original + file), cv.COLOR_BGR2RGB)
        # Showing the image.
        #imshow([img], [file])
        # Cropping the image.
        imgR = img[:, 0:1520]
        imgL = img[:, 1520:3040]
        # Showing the image.
        #imshow([imgR, imgL], ['ImageR', 'ImageL'], 1, 2)
        # Saving the image.
        cv.imwrite(cameraR + f'ImageR0{i + 1}.jpg', cv.cvtColor(imgR, cv.COLOR_RGB2BGR))
        cv.imwrite(cameraL + f'ImageL0{i + 1}.jpg', cv.cvtColor(imgL, cv.COLOR_RGB2BGR))