'''
    Copyright:  JarvisLee
    Date:       2020/12/18
    File Name:  ImageShow.py
'''

# Importing the necessary library.
import matplotlib.pyplot as plt

# The function which is used to display the image.
def imshow(imgs : list, titles : list = None, nrows = 1, ncols = 1, figsize = None, cmap = 'gray'):
    '''
        This function is used to display the images. \n
        Params Description:
            - 'imgs' is the list for all the data of the images.
            - 'titles' is the titles of all the images. (Default is None)
            - 'nrows' is the number of the images would be shown as one row. (Default is 1)
            - 'ncols' is the number of the images would be shown as one column. (Default is 1)
            - 'figsize' is the size of each images. (Default is None)
            - 'cmap' is the color mapping of each image. (Default is gray)
    '''
    # Checking the input params.
    assert imgs is not None and len(imgs) > 0, 'There must be some images!!!'
    assert len(imgs) == len(titles) if titles is not None else 1, 'Please input all the titles!!!'
    assert len(imgs) <= nrows * ncols, 'Not enough places to show all the images!!!'
    # Showing the image.
    if figsize is not None:
        plt.figure(figsize = figsize)
    for i in range(len(imgs)):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(imgs[i], cmap = cmap)
        if titles:
            plt.title(titles[i])
    plt.show()