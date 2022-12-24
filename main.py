import cv2
import numpy as np
import matplotlib.pyplot as plt

dog = cv2.imread('img2.jpg', 0)
mouse = cv2.imread('img3.jpg', 0)
plt.figure(figsize=(14, 18))
plt.subplot(121)
plt.imshow(dog, cmap='gray')
plt.subplot(122)
plt.imshow(mouse, cmap='gray')
plt.show()

dog_fft = np.fft.fftshift(np.fft.fft2(dog))
mouse_fft = np.fft.fftshift(np.fft.fft2(mouse))
plt.figure(figsize=(14, 18))
plt.subplot(121)
plt.imshow(np.log(np.abs(dog_fft)), cmap='gray')
plt.subplot(122)
plt.imshow(np.log(np.abs(mouse_fft)), cmap='gray')
plt.show()

dog_amplitude = np.sqrt(np.real(dog_fft) ** 2 + np.imag(dog_fft) ** 2)
dog_phase = np.arctan2(np.imag(dog_fft), np.real(dog_fft))
mouse_amplitude = np.sqrt(np.real(mouse_fft) ** 2 + np.imag(mouse_fft) ** 2)
mouse_phase = np.arctan2(np.imag(mouse_fft), np.real(mouse_fft))
#phase+magnitude of first img
plt.figure(figsize=(14, 18))
plt.subplot(121)
plt.imshow(np.log(dog_amplitude+1e-10), cmap='gray')
plt.subplot(122)
plt.imshow(dog_phase, cmap='gray')
#phase+magnitude of second img
plt.figure(figsize=(14, 18))
plt.subplot(121)
plt.imshow(np.log(mouse_amplitude+1e-10), cmap='gray')
plt.subplot(122)
plt.imshow(mouse_phase, cmap='gray')

#combination of phase and magnitude try1
dog_mouse_comb = np.multiply(dog_amplitude, np.exp(1j * mouse_phase))
dog_mouse = np.real(np.fft.ifft2(dog_mouse_comb))  # drop imagniary as they are around 1e-14
# combined image has values < 0 and > 1, needs to be scaled.
plt.figure(figsize=(15, 20))
plt.subplot(131)
plt.imshow(np.abs(dog_mouse), cmap='gray')
plt.subplot(132)
dog_mouse_shift = dog_mouse + dog_mouse.min()
dog_mouse_shift[dog_mouse_shift>255] = 255
plt.imshow(dog_mouse_shift)
plt.subplot(133)
dog_mouse[dog_mouse>255] = 255
dog_mouse[dog_mouse <0] = 0
plt.imshow(dog_mouse)

#combination of phase and magnitude try2
dog_mouse_comb = np.multiply(mouse_amplitude, np.exp(1j * dog_phase))
dog_mouse = np.real(np.fft.ifft2(dog_mouse_comb))  # drop imagniary as they are around 1e-14
# combined image has values < 0 and > 1, needs to be scaled.
plt.figure(figsize=(15, 20))
plt.subplot(131)
plt.imshow(np.abs(dog_mouse), cmap='gray')
plt.subplot(132)
dog_mouse_shift = dog_mouse + dog_mouse.min()
dog_mouse_shift[dog_mouse_shift>255] = 255
plt.imshow(dog_mouse_shift)
plt.subplot(133)
dog_mouse[dog_mouse>255] = 255
dog_mouse[dog_mouse <0] = 0
plt.imshow(dog_mouse)
