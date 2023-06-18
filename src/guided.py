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

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60# radius of filter
    eps = 0.0001# regularization parameter
    t = Guidedfilter(gray,et,r,eps)
    return t

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype)
    t = cv2.max(t,tx)

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

def mse(x, y):
    return np.linalg.norm(x - y)


if __name__ == '__main__':

    img = 'dataset/test/input/1-inputs.png'

    src = cv2.imread(img)
    src= cv2.resize(src,(256,256))
    I = src.astype('float64')/255

    dark = DarkChannel(I,15)
    A = AtmLight(I,dark)
    te = TransmissionEstimate(I,A,15)
    t = TransmissionRefine(src,te)
    J = Recover(I,t,A,0.1)

    img_gt = 'dataset/test/target/1-targets.png'
    gt = cv2.imread(img_gt)
    gt= cv2.resize(gt,(256,256))

    
    mse_noise = mse(J, gt)
    print('mse:',mse_noise)
    
    RMSE_noise =np.sqrt(mse_noise)
    print('RMSE:',RMSE_noise)


    psnr = cv2.PSNR(np.uint8(J), gt)
    print('PSNR:',psnr)
    
    ssim_noise = ssim(cv2.cvtColor(np.uint8(J), cv2.COLOR_BGR2GRAY), cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY))
    print('SSIM:',ssim_noise)
    

    cv2.imshow('I',src)
    cv2.imshow("GT",gt)
    cv2.imshow("dark",dark)
    cv2.imshow("transmission estimation",t)
    cv2.imshow("transmission refine",t) 
    cv2.imshow('Recover Image',J)

    #cv2.imwrite("input.png", src)
    #cv2.imwrite("dark.png",dark*255)
    #cv2.imwrite("transmission.png",te*255)
    #cv2.imwrite("GT.png",gt)
    #cv2.imwrite("output.png",J*255)
     
    end_time  = time.time()
    print("time = %s seconds:" % (end_time - start_time))
    cv2.waitKey()