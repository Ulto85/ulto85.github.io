import numpy as np
import skimage as sk
import skimage.io as skio
from tqdm import tqdm

def l2Loss(im1,im2,dx,dy):
    if dx<0:
        im1=im1[:,:dx]
        im2=im2[:,-dx:]
    if dx>0:
        im1 = im1[:,dx:]
        im2 = im2[:,:-dx]
    if dy<0:
        im1 = im1[:dy]
        im2 = im2[-dy:]
    if dy>0:
        im1 = im1[dy:]
        im2 = im2[:-dy]
    return np.sqrt(np.sum((im1-im2)**2))
    
def NCCLoss(im1,im2,dx,dy):
    if dx<0:
            im1=im1[:,:dx]
            im2=im2[:,-dx:]
    if dx>0:
        im1 = im1[:,dx:]
        im2 = im2[:,:-dx]
    if dy<0:
        im1 = im1[:dy]
        im2 = im2[-dy:]
    if dy>0:
        im1 = im1[dy:]
        im2 = im2[:-dy]

    im1 = im1.flatten()
    im2=im2.flatten()
    

    meanNormalizedIm1 = (im1-np.mean(im1))/np.sqrt(np.sum((im1-np.mean(im1))**2))
    meanNormalizedIm2 = (im2-np.mean(im2))/np.sqrt(np.sum((im2-np.mean(im2))**2))
    return -1*np.dot(meanNormalizedIm1,meanNormalizedIm2) #high correlation is good so multiply by negative 1 to maximize

def shift(im,dx,dy):
    im = np.roll(im,dx,axis=1)
    im = np.roll(im,dy,axis=0)
    return im

def normalize_channel(x):
    f=x.flatten()
    return (x-f.mean())/np.linalg.norm(x)

def standardize(a):
    return (a-a.mean())/a.std()