# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:06:48 2020

@author: woocj
"""
import numpy as np
import os
import cv2 
mypath = os.path.dirname(os.path.abspath(__file__))+'\\'


def Gaussian_Mask(size, sigma=1):
    """
    Apply an user-specific size Gaussian mask filter to the image by convolution operation.
    The smaller the mask size, the lesser the blur.
    
    input: size = integer # if size=1, mask size = 1x1, size=3, mask size = 3x3 so on.
    output: 
    """
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    # Using G(x,y) = e^((-x^2-y^2)/2σ^2)/(2πσ^2) to create the mask
    normal = 1 / (2.0 * np.pi * sigma**2)
    mask =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return mask

def Gaussian_Blur(image):
    """
    since edge detection is highly sensitive to noise, gaussian blur is used to get rid of the noise.
    
    type of noise:
        1. gaussian noise/white noise
        2. impulse noise
    type of filtering:
        1. linear smoothing = gaussian smoothing/temperal averaging/spatial averaging
        2. non-linear smoothing = median filtering
    
    input: image
    output: blurred image
    """
    image = np.array(image)
    after_blur = np.array(image)
    sum = 0

    for i in range(3, image.shape[0] - 3):
        for j in range(3, image.shape[1] - 3):
            sum = applyGaussianFilterAtPoint(image, i, j)
            after_blur[i][j] = sum
    return after_blur



def applyGaussianFilterAtPoint(imageData, row, column):
    """
    applying gaussian filter at each and every point
    """
    sum = 0
    for i in range(row - 3, row + 4):
        for j in range(column - 3, column + 4):
            sum += gaussian_filter[i - row + 3][j - column + 3] * imageData[i][j]

    return sum

def log_2nd_derivative(sigma = 1):
    """
    log 2nd derivative edge detection
    
    G(x,y) = (1/(2πσ^2)*(x^2+y^2-σ^2)/σ^4) *e^(-(x^2+y^2)/2σ^2) to create the mask
    """
    size = 7
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    mask = (((x**2 + y**2 - (2.0*sigma**2)) / sigma**4)*normal) * np.exp(-(x**2+y**2) / (2.0*sigma**2)) 
    
    return mask

def convolve(image, mask):
    """
    convolving image with gaussian filter
    """
    xmax = image.shape[0]
    ymax = image.shape[1]
    Kmax = mask.shape[0]
    Koffset = Kmax//2
    convo = np.zeros([xmax, ymax], dtype=np.int32)
    # Border pixels were not convolved, which will be removed from the final image.
    for i in range (Koffset, xmax-Koffset):
        for j in range (Koffset, ymax-Koffset):
            sum = 0
            for a in range (0, Kmax):
                for b in range (0, Kmax):
                    sum += mask[a][b]*image[i+a-Koffset][j+b-Koffset]
            convo[i][j] = sum
    return convo

 
    
def edge_filter(img):
    """
    Detects the edge intensity and direction by calculating the gradient of the image using edge detection operator
    We create 2 filters to highlight the intensity changes.
    
    Magnitude, G = (Ix**2+Iy**2)^0.5
    
    input: blurred image
    output: b&w gradient intensity image and it's angle
    """
    edge_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], np.float32)
    edge_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], np.float32)
    # Get Derivatives in x direction and y direction
    Ix = convolve(img, edge_x)
    Iy = convolve(img, edge_y)
    # Compute Gradient and Angle based on x and y derivatives.
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    angle = np.arctan2(Iy, Ix)
    return (G, angle)

def non_max_suppression(image_matrix, angle_matrix):
    """
    This function is to thin out the thick edges by going through all the points on gradient intensity 
    matrix and find the pixel with the maximum value in the edge directions.
    steps:
        1. create matrix 
        2. identify edge direction based on angle values
        3. check if pixel in the same direction has a higher intensity or not
        4. return the image processed.
        
    input: b&w image and angle
    output:
    """
    x, y = image_matrix.shape
    Z = np.zeros((x, y), dtype = np.int32)
    # Convert radian into degree
    angle_matrix_360 = angle_matrix * 180/np.pi
    
    # Convert angle into 0-180 for comparison
    angle_matrix_360[angle_matrix_360 < 0 ] += 180
    for i in range (1, x-1):
        for j in range (1, y-1):
            p = 255
            q = 255
            
            # See the neighbors of the edges to see if it satisfies the non-maximum
            if (0 <= angle_matrix_360[i,j] < 22.5) or (157.5 <= angle_matrix_360[i, j ] < 180):
                p = image_matrix[i, j+1]
                q = image_matrix[i, j-1]
            elif(22.5 <= angle_matrix_360[i,j] < 67.5):
                p = image_matrix[i+1, j-1]
                q = image_matrix[i-1, j+1]
            elif(67.5 <= angle_matrix_360[i,j] < 112.5):
                p = image_matrix[i+1, j]
                q = image_matrix[i-1, j]
            elif(112.5 <= angle_matrix_360[i,j] < 157.5):
                p = image_matrix[i-1, j-1]
                q = image_matrix[i+1, j+1]
            if (image_matrix[i,j] >= p) and (image_matrix[i, j] >= q):
                Z[i,j] = image_matrix[i,j]
            else:
                Z[i,j] = 0
    return Z
    

"""
processing our .raw files and setting the output

input: .raw files in same directory
output: processed files will be in 'Outputs' folder
"""
for subdir, dirs, files in os.walk(mypath):
        for file in files:
            if file.endswith('.raw'):
                mypath = os.path.join(subdir, file)
                name = str(file.split('.',1)[0])
                #printing the name of the file that is being processed 
                print(name)  
                
                #changing the raw file to binary array
                file = open(mypath, 'r+b')  
                data = file.read()
                raw_array = []
                i = 1
                image_row = 0
                matrix = []
                """ 
                byte 1&2 = row information
                byte 3&4 = column information
                byte 5 = header
                """
                for byte in data:
                    if i == 1 or i == 3:
                        previous_data = byte
                    if i == 2:
                        row = 256*byte+previous_data
                    if i == 4:
                        col = 256*byte+previous_data
                    
                    if i > 5:   
                        if image_row < row:
                            raw_array.append(byte)
                            image_row += 1
                            if image_row == row:
                                matrix.append(raw_array)
                        else:
                            raw_array = [byte]
                            image_row = 1
                    i += 1
                    
                np_array = np.array(matrix)
                

                #choosing the gassian_mask to do edge detection using the mask size 7
                gaussian_filter = Gaussian_Mask(size = 7)     
                
                #applying gaussian smoothing
                gaussianData = Gaussian_Blur(np_array)
                photo , angle = edge_filter(gaussianData)
                
                #using non max suppression to get edge thinned across the edge contour
                final_processed_img = non_max_suppression(photo,angle)
                
                output = 'canny_edge_output/'+name+'_1st.jpg'
                cv2.imwrite(output, final_processed_img)
                
                #using log 2nd derivative to do tha edge detection
                gaussian_filter = log_2nd_derivative()
                
                gaussianData = Gaussian_Blur(np_array)
                photo , angle = edge_filter(gaussianData)
                
                final_processed_img = non_max_suppression(photo,angle)
                output = 'log_edge_output/'+name+'_2nd.jpg'
                cv2.imwrite(output, final_processed_img)
