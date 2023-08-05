import streamlit as st
from PIL import Image
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim 
from sklearn.metrics import mean_squared_error
from skimage import measure
import time
start_time = time.time()

st.set_page_config(
    page_title="Image Haze Removal App",
    layout="wide"
)

st.header("Haze Remover")

############################################# Guided ##############################################
def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def AtmLight(img, dark):
    [h, w, _] = np.shape(img)
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = np.reshape(dark, (imsz, 1))
    imvec = np.reshape(img, (imsz, 3))

    indi = darkvec.argsort()
    indi = indi[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indi[ind]]

    A = atmsum / numpx
    print('A', A)
    return A

def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission

def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q

def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60  # radius of filter
    eps = 0.0001  # regularization parameter
    t = Guidedfilter(gray, et, r, eps)
    return t

def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res

def mse(x, y):
    return np.linalg.norm(x - y)

def Guided(image1, image2):
    img = np.array(image1)
    img = cv2.resize(img, (256, 256))
    I = img.astype('float64') / 255

    dark = DarkChannel(I,15)
    A = AtmLight(I,dark)
    transmission = TransmissionEstimate(I,A,15)
    refined_transmission = TransmissionRefine(img,transmission)
    recovered_image = Recover(I,refined_transmission,A,0.1)
    recovered_image = np.clip(recovered_image, 0.0, 1.0)

    gt = np.array(image2)
    gt = cv2.resize(gt, (256, 256))
    mse_noise = np.around(mse(recovered_image, gt), 2)
    psnr = np.around(cv2.PSNR(np.uint8(recovered_image), gt), 5)
    ssim_noise = ssim(cv2.cvtColor(np.uint8(recovered_image), cv2.COLOR_BGR2GRAY),
                      cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY))
    ssim_noise = np.around(ssim_noise, 5)

    return recovered_image, mse_noise, psnr, ssim_noise

############################################################ Dark Channel #######################################################

def Dark_Channel(image1, image2):
    img = np.array(image1)
    img = cv2.resize(img, (256, 256))
    I = img.astype('float64') / 255

    dark = DarkChannel(I,15)
    A = AtmLight(I,dark)
    transmission = TransmissionEstimate(I,A,15)
    recovered_image = Recover(I, transmission, A, 0.1)
    recovered_image = np.clip(recovered_image, 0.0, 1.0)

    gt = np.array(image2)
    gt = cv2.resize(gt, (256, 256))
    mse_noise = np.around(mse(recovered_image, gt), 2)
    psnr = np.around(cv2.PSNR(np.uint8(recovered_image), gt), 5)
    ssim_noise = ssim(cv2.cvtColor(np.uint8(recovered_image), cv2.COLOR_BGR2GRAY),
                      cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY))
    ssim_noise = np.around(ssim_noise, 5)

    return recovered_image, mse_noise, psnr, ssim_noise


col1, col2, col3 = st.columns([0.2, 0.6, 0.2])

with col1:
    algo = st.selectbox("Select Algorithum", ['Guided', 'DarkChannel'])

with col2:
    upload_file_1 = st.file_uploader("Upload Input Image", type=['png', 'jpeg', 'jpg'], key="image_1")
    upload_file_2 = st.file_uploader("Upload Target Image", type=['png', 'jpeg', 'jpg'], key="image_2")

    if upload_file_1 is not None:
        try:
            image_1 = Image.open(upload_file_1)
        except:
            pass

        try:
            image_2 = Image.open(upload_file_2)
        except:
            st.stop()

        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.markdown('<p style="text-align: center;">Before</p>', unsafe_allow_html=True)
            st.image(image_1, width=300)

        with col2:
            st.markdown('<p style="text-align: center;">After</p>', unsafe_allow_html=True)
            if algo == 'Guided':
                recovered_image, mse_noise, psnr, ssim_noise = Guided(image_1, image_2)
                st.image(recovered_image, width=300)

            if algo == 'DarkChannel':
                recovered_image, mse_noise, psnr, ssim_noise = Dark_Channel(image_1, image_2)
                st.image(recovered_image, width=300)

with col3:
    container = st.container()

    container.markdown("""
    <style>
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px;
        border-radius: 10px;
        background-color: #2b3240;
        margin-bottom: 10px;
    }
    
    .metric-label {
        font-weight: bold;
        font-size: 18px;
    }
    .metric-value {
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

    with container:
        st.write("Quantative Analysis")
        if algo == 'Guided' and upload_file_1 and upload_file_2 is not None:
            recovered_image, mse_noise, psnr, ssim_noise = Guided(image_1, image_2)
            st.markdown(
                '<div class="metric-row"><span class="metric-label">MSE:</span><span class="metric-value">{}</span></div>'.format(mse_noise), unsafe_allow_html=True)
            st.markdown(
                '<div class="metric-row"><span class="metric-label">PSNR:</span><span class="metric-value">{}</span></div>'.format(psnr), unsafe_allow_html=True)
            st.markdown(
                '<div class="metric-row"><span class="metric-label">SSIM:</span><span class="metric-value">{}</span></div>'.format(ssim_noise), unsafe_allow_html=True)
            
        if algo == 'DarkChannel' and upload_file_1 and upload_file_2 is not None:
            recovered_image, mse_noise, psnr, ssim_noise = Dark_Channel(image_1, image_2)
            st.markdown(
                '<div class="metric-row"><span class="metric-label">MSE:</span><span class="metric-value">{}</span></div>'.format(mse_noise), unsafe_allow_html=True)
            st.markdown(
                '<div class="metric-row"><span class="metric-label">PSNR:</span><span class="metric-value">{}</span></div>'.format(psnr), unsafe_allow_html=True)
            st.markdown(
                '<div class="metric-row"><span class="metric-label">SSIM:</span><span class="metric-value">{}</span></div>'.format(ssim_noise), unsafe_allow_html=True)
