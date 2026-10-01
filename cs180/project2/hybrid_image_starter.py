import matplotlib.pyplot as plt
from align_image_code import align_images
from parts import optimizedConvolution
import numpy as np
import skimage as sk
import skimage.io as skio
import scipy
import cv2
import os 

# First load images
def part2_2(fp1,fp2,s1,s2):
    # high sf
    #im1 = plt.imread('./DerekPicture.jpg') / 255.

    im1=skio.imread(f'input/{fp1}',as_gray=True)
    im1Name =  fp1.split(".")[0]
    im2Name=fp2.split(".")[0]
    # low sf
    #im2 = plt.imread('./nutmeg.jpg') / 255.
    im2=skio.imread(f'input/{fp2}',as_gray=True)
    dirName =im1Name+im2Name
    if not os.path.isdir(dirName):
        os.mkdir(dirName)
    # Next align images (this code is provided, but may be improved)
    im1_aligned,im2_aligned  = align_images(im1, im2)

    plt.imsave(f"{dirName}/{im1Name}_aligned_hf.jpg",im1_aligned,cmap='gray')
    plt.imsave(f"{dirName}/{im2Name}_aligned_lf.jpg",im2_aligned,cmap='gray')
    plt.imsave(f"{dirName}/{im1Name}_aligned_fft_hf.jpg",np.abs(np.fft.fftshift(np.fft.fft2(im1_aligned))),cmap='gray')
    plt.imsave(f"{dirName}/{im2Name}_aligned_fft_lf.jpg",np.abs(np.fft.fftshift(np.fft.fft2(im2_aligned))),cmap='gray')

    ## You will provide the code below. Sigma1 and sigma2 are arbitrary 
    ## cutoff values for the high and low frequencies
    def hybrid_image(im1,im2,sigma1,sigma2):
        highsfsigma=sigma1
        highsfimg=im1
        lowsfsigma = sigma2
        lowsfimg=im2

        ###LOW SF
        lkernel= cv2.getGaussianKernel(int(lowsfsigma)*6+1,lowsfsigma)
        lkernel = np.outer(lkernel,lkernel.T)
        lowsfimg = optimizedConvolution(lowsfimg,lkernel)
        plt.imsave(f"{dirName}/{im2Name}_aligned_fft_lf_pass.jpg",np.abs(np.fft.fftshift(np.fft.fft2(lowsfimg))),cmap='gray')
        


        ## High SF
        hkernel = cv2.getGaussianKernel(int(highsfsigma)*6+1,highsfsigma)
        hkernel = np.outer(hkernel,hkernel.T)
        highsfimg2 = optimizedConvolution(highsfimg,hkernel)
        highsfimg=highsfimg-highsfimg2

        plt.imsave(f"{dirName}/{im1Name}_aligned_fft_hf_pass.jpg",np.abs(np.fft.fftshift(np.fft.fft2(highsfimg))),cmap='gray')

        return (highsfimg+lowsfimg)

    sigma1 = s1
    sigma2 = s2
    hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
    plt.imsave(f"{dirName}/hybrid_{sigma1}_{sigma2}.jpg",hybrid,cmap='gray')
    plt.imsave(f"{dirName}/hybrid_fft",np.abs(np.fft.fftshift(np.fft.fft2(hybrid))),cmap='gray')


    # plt.imshow(hybrid)
    # plt.show()