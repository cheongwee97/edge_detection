import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

fname = "Lenna.png"
#Read img as grayscale
img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2GRAY)



def conv(img, kernel):
    kH,kW = kernel.shape
    (imH,imW) = img.shape
    new_img = np.zeros(img.shape)
    pad = int((kH-1)/2)
    
    for y in range(imH-kH):
        for x in range(imW-kW):
            window = img[y:y+kH,x:x+kW]
            new_img[y+pad,x+pad] = (kernel * window).sum()
    
    return new_img

def sobel_edge(img):
    #Smoothing out img with gaussian filter
    for i in range(5):
        img = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)

    sobel_x = np.array([[-1,0,1],
                         [-2,0,2],
                         [-1,0,1]])
                
    sobel_y = np.array([[-1,-2,-1],
                         [0,0,0],
                         [1,2,1]])

    Gx = conv(img, sobel_x)
    Gy = conv(img, sobel_y)

    edge = np.sqrt((Gx)**2 + (Gy)**2)

    return edge.astype(np.uint8)

def canny_edge(img):
    imH,imW = img.shape #Image Height and Width
    kH,kW = 3,3 #Kernel Height and Width
    directions_img = np.zeros((img.shape[0], img.shape[1], 3))
    nms_img = np.zeros(img.shape)
    pad = int((kH-1)/2)

    #Smoothing out img with gaussian filter
    for i in range(5):
        img = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)

    sobel_x = np.array([[-1,0,1],
                         [-2,0,2],
                         [-1,0,1]])
                
    sobel_y = np.array([[-1,-2,-1],
                         [0,0,0],
                         [1,2,1]])

    Gx = conv(img, sobel_x)
    Gy = conv(img, sobel_y)
    ratio = Gy/Gx
    theta = np.arctan(ratio)
    theta = theta.flatten()
    theta = [90 if math.isnan(x) else math.degrees(x) for x in theta]
    theta = np.reshape(theta, (imH,imW))

    edge = np.sqrt((Gx)**2 + (Gy)**2)

        

    
    for i in range(imH-1):
        for j in range(imW-1):
            if(theta[i,j] != 0):
                quad = theta[i,j]//22.5
                if quad in [0,8]:
                    neighborA = edge[i,j-1] #Left
                    neighborB = edge[i,j+1] #Right
                elif quad in [1,2]:
                    neighborA = edge[i-1,j-1] #LowerLeft
                    neighborB = edge[i+1,j+1] #UpperRight
                elif quad in [5,6]:
                    neighborA = edge[i-1,j+1] #LowerRight
                    neighborB = edge[i+1,j-1] #UpperLeft
                elif quad in [3,4]:
                    neighborA = edge[i-1,j] #Down
                    neighborB = edge[i+1,j] #Up
                if(edge[i,j] >= 255 * 0.2 and edge[i,j] > neighborA and edge[i,j] > neighborB):
                    nms_img[i,j] = edge[i,j]
                    directions_img[i,j,:] = [theta[i,j],255,255]

    directions_img = cv2.cvtColor(directions_img.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return edge.astype(np.uint8),directions_img,nms_img.astype(np.uint8)





sobel_img,gradient, nms_img = canny_edge(img)



plt.figure()

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title("Original Image")

plt.subplot(222)
plt.title("Sobel Filter Edge Detection")
plt.imshow(sobel_img, cmap="gray")

plt.subplot(223)
plt.title("Direction of gradient")
plt.imshow(gradient)

plt.subplot(224)
plt.title("After performing Non-Max-Suppression")
plt.imshow(nms_img,cmap="gray")

plt.show()


