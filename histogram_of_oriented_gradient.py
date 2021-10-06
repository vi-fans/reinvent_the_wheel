import cv2
import sys
import math
import numpy as np

def compute_hog(img,output_file_name):
    #break each quadrant into eight bins, total of 32
    quadrant_resolution=(math.pi/2)/8

    #compute gradient image
    gradient_img_x=cv2.Sobel(img,cv2.CV_64F,1,0)
    gradient_img_y=cv2.Sobel(img,cv2.CV_64F,0,1)

    #boundary conditions
    gradient_img_y[gradient_img_x==0]=0
    gradient_img_x[gradient_img_x==0]=1

    w,h=np.shape(gradient_img_x)
    oriented_img=np.zeros((w,h))

    #compute gradient orientation for each pixel, should be able to vectorise for faster computation
    for row in range(0,h):
        for col in range(0,w):
            oriented_img[row,col]=np.arctan(gradient_img_y[row,col]/gradient_img_x[row,col])
            oriented_img[row,col]=np.floor(np.abs(oriented_img[row,col]/quadrant_resolution))
            if gradient_img_x[row,col]<0 and gradient_img_y[row,col]>0:
                oriented_img[row,col]=oriented_img[row,col]+8
            elif gradient_img_x[row,col]<0 and gradient_img_y[row,col]<0:
                oriented_img[row,col]=oriented_img[row,col]+16
            elif gradient_img_x[row,col]>0 and gradient_img_y[row,col]<0:
                oriented_img[row,col]=oriented_img[row,col]+24

    #scaling back for visualisation
    min_oriented_img=np.min(oriented_img)
    max_oriented_img=np.max(oriented_img)
    visualise_oriented_img=(oriented_img-min_oriented_img)/(max_oriented_img-min_oriented_img)*255

    #write into an image
    cv2.imwrite(output_file_name,visualise_oriented_img)

    #turn into a histogram
    hog_vector=np.histogram(oriented_img,bins=32,range=(0,32))
    return hog_vector

if __name__=='__main__':
    img=cv2.imread(sys.argv[1],0)
    hog_vector=compute_hog(img,sys.argv[2])
    print(hog_vector)

