# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
from tqdm import tqdm

# name of the input file
imname = 'CS180_fa2026_proj1_data/cathedral.jpg'

# read in the image
im = skio.imread(imname)

# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)
    
# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(np.int32)

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

width = len(b)



        




# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)



def shift(im,dx,dy):
    im = np.roll(im,dx,axis=1)
    im = np.roll(im,dy,axis=0)
    return im

def l2Loss(im1,im2,dx,dy):
    im2 = shift(im2,dx,dy)
    m=len(im1)
    n=len(im1[0])
    im1=im1[int(0.05*m):int(0.95*m),int(0.05*n):int(0.95*n)]
    im2=im2[int(0.05*m):int(0.95*m),int(0.05*n):int(0.95*n)]
    
    return np.sqrt(np.sum((im1-im2)**2))
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
    im2 = shift(im2,dx,dy)
    m=len(im1)
    n=len(im1[0])
    im1=im1[int(0.05*m):int(0.95*m),int(0.05*n):int(0.95*n)]
    im2=im2[int(0.05*m):int(0.95*m),int(0.05*n):int(0.95*n)]
    
    im1 = im1.flatten()
    im2=im2.flatten()
    

    meanNormalizedIm1 = (im1-np.mean(im1))/np.sqrt(np.sum((im1-np.mean(im1))**2))
    meanNormalizedIm2 = (im2-np.mean(im2))/np.sqrt(np.sum((im2-np.mean(im2))**2))
    return -1*np.dot(meanNormalizedIm1,meanNormalizedIm2) #high correlation is good so multiply by negative 1 to maximize

# def l2Loss(im1,im2,dx,dy): 
#     m = len(im1) 
#     n= len(im1[0]) 
#     s = 0 
#     for i in range(m): 
#         for j in range(n): 
#             if i+dy<0 or i+dy>=m or j+dx<0 or j+dx>=n: 
#                 continue
#             else: 
#                 s+= (im1[i][j]-im2[i+dy][j+dx])**2 
#    return np.sqrt(s)
def align(a,b,loss):
    m = len(a)
    n= len(a[0])
    losses = float('inf')
    bestDx = None
    bestDy = None
    a1 = a[int(0.05*m):int(0.95*m),int(0.05*n):int(0.95*n)]
    b1=b[int(0.05*m):int(0.95*m),int(0.05*n):int(0.95*n)]
    # a1=a
    # b1=b
    
    for dy in tqdm(range(-20,20)):
        for dx in range(-20,20):
            l = loss(a1,b1,dx,dy)
            if l<losses:
                losses=l
                bestDx = dx
                bestDy=dy
    return shift(b,bestDx,bestDy)

# save the image


ag = align(b,g,l2Loss)
ar = align(b,r,l2Loss)
# create a color image
im_out = np.dstack([ar, ag, b])
fname = 'testresults/out_fname.jpg'
skio.imsave(fname, im_out.astype(np.uint8))

# display the image
skio.imshow(im_out)
skio.show()


