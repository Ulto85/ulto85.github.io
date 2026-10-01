from matplotlib.widgets import Slider

import matplotlib.pyplot as plt
from align_image_code import align_images
from part1 import optimizedConvolution
import numpy as np
import skimage as sk
import skimage.io as skio
import scipy
import cv2

# First load images

# high sf
im1 = plt.imread('./DerekPicture.jpg') / 255.
im1=skio.imread('./aayan1.png',as_gray=True)
# low sf
im2 = plt.imread('./nutmeg.jpg') / 255.
im2=skio.imread('./akhil1.png',as_gray=True)
# Next align images (this code is provided, but may be improved)
im2_aligned,im1_aligned  = align_images(im2, im1)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies
def hybrid_image(im1,im2,sigma1,sigma2):
    highsfsigma=sigma1
    highsfimg=im1
    lowsfsigma = sigma2
    lowsfimg=im2

    ###LOW SF
    lkernel= cv2.getGaussianKernel(int(lowsfsigma)*2+1,lowsfsigma)
    lkernel = np.outer(lkernel,lkernel.T)
    lowsfimg = optimizedConvolution(lowsfimg,lkernel)

    ## High SF
    hkernel = cv2.getGaussianKernel(int(highsfsigma)*2+1,highsfsigma)
    hkernel = np.outer(hkernel,hkernel.T)
    highsfimg2 = optimizedConvolution(highsfimg,hkernel)
    highsfimg=highsfimg-highsfimg2

    return (highsfimg+lowsfimg)

sigma1 = 4
sigma2 = 7
hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
skio.imsave("hybrid.jpg",(hybrid*255).astype(np.uint8))


# -----------------------------
# PRECOMPUTE SIGMA RESULTS
# -----------------------------

sigma_values = range(1, 31)

high_cache = {}
low_cache = {}

print("Precomputing sigma values...")

for s in sigma_values:
    print(f"Computing sigma {s}/30")

    # Same low-frequency computation as hybrid_image
    lkernel = cv2.getGaussianKernel(int(s)*2+1, s)
    lkernel = np.outer(lkernel, lkernel.T)
    low_cache[s] = optimizedConvolution(im2_aligned, lkernel)

    # Same high-frequency computation as hybrid_image
    hkernel = cv2.getGaussianKernel(int(s)*2+1, s)
    hkernel = np.outer(hkernel, hkernel.T)
    highsfimg2 = optimizedConvolution(im1_aligned, hkernel)
    high_cache[s] = im1_aligned - highsfimg2

print("Done precomputing!")


# -----------------------------
# INTERACTIVE VIEWER
# -----------------------------

fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.20)

display = ax.imshow(hybrid, cmap="gray")
ax.axis("off")

title = ax.set_title(
    f"sigma1 = {sigma1}, sigma2 = {sigma2}"
)

# Slider axes
ax_sigma1 = plt.axes([0.20, 0.10, 0.65, 0.03])
ax_sigma2 = plt.axes([0.20, 0.05, 0.65, 0.03])

sigma1_slider = Slider(
    ax_sigma1,
    "sigma1",
    1,
    30,
    valinit=sigma1,
    valstep=1
)

sigma2_slider = Slider(
    ax_sigma2,
    "sigma2",
    1,
    30,
    valinit=sigma2,
    valstep=1
)


def update(val):
    new_sigma1 = int(sigma1_slider.val)
    new_sigma2 = int(sigma2_slider.val)

    # Equivalent to hybrid_image, but uses precomputed convolutions
    new_hybrid = high_cache[new_sigma1] + low_cache[new_sigma2]

    display.set_data(new_hybrid)

    title.set_text(
        f"sigma1 = {new_sigma1}, sigma2 = {new_sigma2}"
    )

    fig.canvas.draw_idle()


sigma1_slider.on_changed(update)
sigma2_slider.on_changed(update)

plt.show()