from skimage import color, filters, measure, morphology, transform

from scipy import ndimage

from math import ceil, floor

from functools import reduce

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os

def bbox_reducer(a,b):
    """Reduces two bounding boxes tuples to a bounding box
    encompassing both. The bounding box format expected is
    
    (min_row, min_col, max_row, max_col)
    
    Used with the reduce function to merge bounding boxes
    
    """
    min_row = min(a[0],b[0])
    min_col = min(a[1],b[1])
    max_row = max(a[2],b[2])
    max_col = max(a[3],b[3])
    return (min_row, min_col, max_row, max_col)

def plot_intermediate_steps(arr):
    """
    Plot a sequence of intermediate steps
    """
    n_img = len(arr)
    plt.figure(figsize=(16,24))
    
    for idx, item in enumerate(arr):
        image, cmap, box = item
        ax = plt.subplot(1,n_img,idx+1)
        plt.imshow(image,cmap)
        if box:
            ax.add_patch(box)
    plt.show()
    
def center_and_resize(image,new_size = (64,64),plot=False,square=True, id = None):
    #Criando as máscaras que representam os objetos
    image_bw = color.rgb2gray(image)
    image_countour = filters.sobel(image_bw)
    image_filled = ndimage.binary_fill_holes(image_countour)
    
    image_mask = morphology.convex_hull_image(image_filled)
    
    #Identificando os objetos na imagem
    labels, n_objects = ndimage.label(image_mask)
    regions = measure.regionprops(labels)
    slices = ndimage.find_objects(labels)

    #Getting Bounding box that encompasses all regions
    bbox_list = [r.bbox for r in regions]
    min_row, min_col, max_row, max_col = reduce(bbox_reducer,bbox_list)
    
    #If the bounding box is not squared, make it so
    if square:
        len_row = max_row - min_row
        len_col = max_col - min_col

        if len_row > len_col:
            min_col -= ceil((len_row - len_col)/2)
            max_col += floor((len_row - len_col)/2)
        else:
            min_row -= ceil((len_col - len_row)/2)
            max_row += floor((len_col - len_row)/2)      
    

    #We may have some out of bound stuff hapenning here
    if (max_row - min_row) > image.shape[0]:
        raise ValueError("ID = {id} - Bounding box height is greater than image height".format(id=id))
    if (max_col - min_col) > image.shape[1]:
        raise ValueError("ID = {id} - Bounding box width is greater than image width".format(id=id))
        
    #If Bounding box exceeds image limits, we shift it inside
    if min_row < 0:
        max_row += abs(min_row)
        min_row = 0
    if min_col < 0:
        max_col += abs(min_col)
        min_col = 0
    if max_row >= image.shape[1]:
        min_row -= max_row - image.shape[1] + 1
        max_row = image.shape[0]-1
    if max_col >= image.shape[1]:
        min_col -= max_col - image.shape[1] + 1
        max_col = image.shape[1]-1    
    
    image_slice = (
        slice(min_row,max_row,None) ,
        slice(min_col,max_col,None)               
    )
    
    image_bounded = image[image_slice]
    
    image_resize = transform.resize(image_bounded,new_size)
    
    if plot:
        image_box = patches.Rectangle(
            (min_col,min_row),
            max_row - min_row,  
            max_col - min_col,  
            fc = "none",
            ec = "red"
        )
        img_arr = [
            (image,None,None),
            (image_bw,"gray",None),
            (image_countour,"gray",None),
            (image_filled,"gray",None),
            (image_mask,"gray",None),
            (image,None,image_box),
            (image_resize,None,None)
        ]
        plot_intermediate_steps(img_arr)
    
    return image_resize    