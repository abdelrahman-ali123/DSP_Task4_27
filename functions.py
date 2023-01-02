import cv2
import numpy as np
import matplotlib.pyplot as plt

class Image:
    def _init_(self, filename, filepath):
        self.filename = filename
        self.filepath = filepath
     
    def getresize(self):
        original_img = cv2.imread(self.filepath, 0)
        img =  cv2.resize(original_img, (1440, 1080))   
        return img

    def getfft( img ):
        img_fft = np.fft.fftshift(np.fft.fft2(img))
        return img_fft

    def getmag(self, img_fft):
        # img_mag = np.sqrt(np.real(img_fft) * 2 + np.imag(img_fft) * 2)
        img_mag = np.abs(img_fft)
        magpath = (f"static/IMG/{self.filename}_mag.jpg")
        plt.imsave(magpath , np.log(img_mag+1e-10) , cmap='gray')
        return magpath, img_mag

    def getphase(self, img_fft):
        # img_phase = np.arctan2(np.imag(img_fft), np.real(img_fft))
        img_phase = np.angle(img_fft)
        phasepath = (f"static/IMG/{self.filename}_phase.jpg")
        plt.imsave(phasepath, img_phase, cmap='gray')
        return phasepath, img_phase
    
    def freqrange(img_mag , img_phase ,y1,y2, x1,x2):
        new_img_mag = img_mag.copy() #take a copy of the magnitude
        new_img_phase = img_phase.copy() #take a copy of the phase
        for i in range(img_mag.shape[0]):
            for j in range(img_phase.shape[1]):
                if (i <=y1 or i>=y2 ) or (j<=x1 or j>=x2):
                    new_img_mag[i][j]=1
                    new_img_phase[i][j]=0
        return new_img_mag , new_img_phase
    
    def imgcombine(self , new_img_mag , new_img_phase):
        merging = np.multiply(new_img_mag, np.exp(1j * new_img_phase))
        img_combine = np.real(np.fft.ifft2(np.fft.ifftshift(merging)))
        combinepath = (f"static/IMG/{self.filename}_combine.jpg")
        plt.imsave(combinepath, img_combine, cmap='gray')
        return combinepath , img_combine
    
