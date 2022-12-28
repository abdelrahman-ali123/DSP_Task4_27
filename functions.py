import cv2
import numpy as np
import matplotlib.pyplot as plt


#upload images and resize them
img1_path = cv2.imread("static\IMG\img2.jpg", 0)
first_img= cv2.resize(img1_path, (1440, 1080))

img2_path = cv2.imread("static\IMG\img7.jpg", 0)
second_img= cv2.resize(img2_path, (1440, 1080))

#appling fft 
first_img_fft = np.fft.fftshift(np.fft.fft2(first_img))
second_img_fft = np.fft.fftshift(np.fft.fft2(second_img))

#git amplitude and phase

# first_img_amplitude = np.sqrt(np.real(first_img_fft) ** 2 + np.imag(first_img_fft) ** 2)
# first_img_phase = np.arctan2(np.imag(first_img_fft), np.real(first_img_fft))
# second_img_amplitude = np.sqrt(np.real(second_img_fft) ** 2 + np.imag(second_img_fft) ** 2)
# second_img_phase = np.arctan2(np.imag(second_img_fft), np.real(second_img_fft))

first_img_amplitude = np.abs(first_img_fft)
first_img_phase = np.angle(first_img_fft)
second_img_amplitude = np.abs(second_img_fft)
second_img_phase = np.angle(second_img_fft)

#merge between images
first_second_imgs_combine = np.multiply(first_img_amplitude, np.exp(1j * second_img_phase))
first_second_imgs = np.real(np.fft.ifft2(first_second_imgs_combine))  

second_first_imgs_combine = np.multiply(second_img_amplitude, np.exp(1j * first_img_phase))
second_first_imgs = np.real(np.fft.ifft2(np.fft.ifftshift(second_first_imgs_combine)))

#removing specific range of frequencies
new_first_img_amplitude=first_img_amplitude.copy() #take a copy of the magnitude 
new_first_img_phase=first_img_phase.copy()#take a copy of the phase
for i in range(first_img_amplitude.shape[1]):
    for j in range(first_img_amplitude.shape[0]):
        if (i >=500 and i<=600 )and (j>=700 and j <=800 ):
            new_first_img_amplitude[i][j]=1
            new_first_img_phase[i][j]=0

#reconstruct signal
imgs_comb = np.multiply((new_first_img_amplitude), np.exp(1j * new_first_img_phase))
first_second_imgs = np.real(np.fft.ifft2(np.fft.ifftshift(imgs_comb)))
