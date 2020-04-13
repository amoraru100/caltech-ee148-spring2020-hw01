import os
import numpy as np
import json
from PIL import Image

def normalize(v):
    '''
    This function returns the normalized vector v
    '''

    norm = np.linalg.norm(v)
    if norm ==0:
        return v
    return v/norm

def find_peaks(a, reach=10):
    ''' 
    This function returns the indicies of all non-zero values in array a that are
    the maximum of a square of length 2 x reach centered at it's location. The
    default square size is 21x21
    '''

    rows, columns = np.shape(a)

    # create a new array that is just a with a border of all 0s so we don't have to deal with edge cases
    new_a = np.zeros([rows+(2*reach),columns+(2*reach)])
    new_a[reach:-reach,reach:-reach] = a

    peaks = []
    # loop through each element in a
    for r in range(reach,rows+reach):
        for c in range(reach,columns+reach):
            mid = new_a[r][c]
            # first check if the value is nonzero
            if mid!= 0:
                # check if the value is the max value of the square
                if mid == np.amax(new_a[r-reach:r+reach+1,c-reach:c+reach+1]):
                    peaks.append([r-reach,c-reach])
    
    return peaks         

def get_circle_kernel(size, color):
    '''
    This function defines a square kernel matrix  of size (2*size+1) X (2*size+1)
    for a circle with the color given by the [R,G,B] in the top 1/3 of the rectangle
    with the value of color with a black background
    '''
    
    # first check if size is a valid number
    if size < 1:
        print('size cannot be < 1')
        return
    
    # define the size of the square kernel matix
    k = np.zeros((2*size+1,2*size+1,3))

    for r in range(2*size+1):
        for c in range(2*size+1):
            if np.sqrt((r-size)**2+(c-size)**2) < size:
                k[r][c] = color
    
    return k   

def shrink_kernel(k):
    '''
    This function shrinks the size of the kernel by 2 rows and 2 columns
    by averaging each pixel RBG value with it's neighboring pixels
    '''

    new_k = np.zeros((np.shape(k)[0]-2,np.shape(k)[1]-2,3))
    for r in range(np.shape(new_k)[0]):
        for c in range(np.shape(new_k)[1]):
            red_square = k[r:r+3,c:c+3,0]
            green_square = k[r:r+3,c:c+3,1]
            blue_square = k[r:r+3,c:c+3,2]
            new_k[r][c] = [np.average(red_square),np.average(green_square),np.average(blue_square)]
    
    return new_k
            
def detect_red_light(I,k,threshold = 0.95):
    
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''    
    
    '''
    BEGIN YOUR CODE
    '''

    bounding_boxes = []

    # define the height and width of the kernel
    k_height = np.shape(k)[0]
    k_width = np.shape(k)[1]


    # normalize the kernel
    k_red = normalize(k[...,0]).flatten()
    k_green = normalize(k[...,1]).flatten()
    k_blue = normalize(k[...,2]).flatten()

    # create a matrix to store the scores
    score = np.zeros([np.shape(I)[0]-k_height,np.shape(I)[1]-k_width])

    # convolve the kernel with the image
    # loop through the image
    for r in range(np.shape(I)[0]-k_height):
        for c in range(np.shape(I)[1]-k_width):

            # select a region of the image with the same size as the kernel
            d = I[r:r+k_height,c:c+k_width,:]

            # normalize the selected region of the image
            d_red = normalize(d[...,0]).flatten()
            d_green = normalize(d[...,1]).flatten()
            d_blue = normalize(d[...,2]).flatten()

            red_score = np.inner(d_red,k_red)
            green_score = np.inner(d_green,k_green)
            blue_score = np.inner(d_blue,k_blue)

            avg_score = (red_score+blue_score+green_score)/3
                
            if avg_score > threshold:
                score[r][c] = avg_score

    peaks = find_peaks(score)

    for peak in peaks:
        bounding_boxes.append([peak[0],peak[1],peak[0]+k_height,peak[1]+k_width])

    '''
    END YOUR CODE
    '''

    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

# set the current path
path = os.getcwd()

# set the path to the downloaded data: 
data_path = path + '\\data\\RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = path + '\\data\\hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

# define the kernel
# read image using PIL:
I = Image.open(os.path.join(data_path,'RL-011.jpg'))
I = np.array(I)
k = I[68:132,349:376,:]

# read image using PIL:
I = Image.open(os.path.join(data_path,'RL-032.jpg'))
I = np.array(I)
k2 = I[209:229,306:314,:]

preds = {}
for i in range(len(file_names)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.array(I)
    
    preds[file_names[i]] =detect_red_light(I,k2)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
