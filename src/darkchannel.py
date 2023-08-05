import cv2
import math
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim 

from sklearn.metrics import mean_squared_error
from skimage import measure

import time
start_time = time.time()

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(img,dark):
    [h,w,_] = np.shape(img)
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = np.reshape(dark,(imsz,1))
    imvec = np.reshape(img,(imsz,3))

    indi = darkvec.argsort()
    indi = indi[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indi[ind]]

    A = atmsum / numpx
    print('A',A)
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95
    im3 = np.empty(im.shape,im.dtype)

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz)
    return transmission


def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype)
    t = cv2.max(t,tx)

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

def mse(x, y):
    return np.linalg.norm(x - y)


if __name__ == '__main__':

    img = 'test/1-inputs.png'

    src = cv2.imread(img)
    src= cv2.resize(src,(256,256))
    I = src.astype('float64')/255


    dark = DarkChannel(I,15)
    A = AtmLight(I,dark)
    te = TransmissionEstimate(I,A,15)
    J1 = Recover(I, te, A, 0.1)


    img_gt = 'test/GTpng'
    gt = cv2.imread(img_gt)
    gt= cv2.resize(gt,(256,256))

    # analysis
    mse_noise = mse(J1, gt)
    print('mse:',mse_noise)
    
    RMSE_noise =np.sqrt(mse_noise)
    print('RMSE:',RMSE_noise)
    
    
    # PSNR_noise = psnr(np.uint8(J1), gt)
    psnr = cv2.PSNR(np.uint8(J1), gt)
    print('PSNR:',psnr)
    
    ssim_noise = ssim(cv2.cvtColor(np.uint8(J1), cv2.COLOR_BGR2GRAY), cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY))
    print('SSIM:',ssim_noise)
    
    
# show all images
    cv2.imshow('I',src) #source image
    cv2.imshow("GT",gt) # Ground Truth image
    cv2.imshow("dark",dark) #dark channel output
    cv2.imshow("transmission estimation",te) #output of transmission Estimation
    cv2.imshow("transmission refine",te) #output of refinement 
    cv2.imshow('Recover Image',J1) #output dehaze image


    #cv2.imwrite("input.png", src)
    #cv2.imwrite("GT.png", gt)
    #cv2.imwrite("darkchannel.png",dark*255)
    #cv2.imwrite("transmission.png",te*255)
    #arr= np.uint8(J1.astype('float64')*255)
    #cv2.imwrite("output_dc.png",arr)
    
    
    end_time  = time.time()
    print("time = %s seconds:" % (end_time - start_time))
    cv2.waitKey()