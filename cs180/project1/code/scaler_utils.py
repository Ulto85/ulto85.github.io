from utils import *
import numpy as np
import skimage as sk
import skimage.io as skio
from tqdm import tqdm


def singleScale(fp,fname,loss,f=lambda x:x, show=False):
    im = skio.imread(fp)
    im = sk.img_as_float(im)
    height = np.floor(im.shape[0] / 3.0).astype(np.int32)
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    width = len(b)
    def align(a,b,loss):
        m = len(a)
        n= len(a[0])
        losses = float('inf')
        bestDx = None
        bestDy = None
        a1 = a[int(0.05*m):int(0.95*m),int(0.05*n):int(0.95*n)]
        b1=b[int(0.05*m):int(0.95*m),int(0.05*n):int(0.95*n)] 
        for dy in tqdm(range(-20,20)):
            for dx in range(-20,20):
                l = loss(a1,b1,dx,dy)
                if l<losses:
                    losses=l
                    bestDx = dx
                    bestDy=dy
        return shift(b,bestDx,bestDy),bestDx,bestDy
    ag,dgx,dgy = align(b,g,loss)
    ar,drx,dry = align(b,r,loss)
    im_out = np.dstack([ar, ag, b])
    
    skio.imsave(fname, (im_out*255).astype(np.uint8))

    
    if show:
        skio.imshow(im_out)
        skio.show()
    return ((dgx,dgy),(drx,dry))

def multiScale(fp,fname,loss,f=lambda x:x , show=False):
    im = skio.imread(fp)
    im = sk.img_as_float(im)
    height = np.floor(im.shape[0] / 3.0).astype(np.int32)
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    width = len(b)
    def singleAlign(a,b,loss,s1=-20,s2=20,z1=-20,z2=20):
        m = len(a)
        n= len(a[0])
        losses = float('inf')
        bestDx = None
        bestDy = None
        # a1 = a[int(0.05*m):int(0.95*m),int(0.05*n):int(0.95*n)]
        # b1=b[int(0.05*m):int(0.95*m),int(0.05*n):int(0.95*n)]
        a1=a
        b1=b
        for dy in tqdm(range(s1,s2)):
            for dx in range(z1,z2):
                l = loss(a1,b1,dx,dy)
                if l<losses:
                    losses=l
                    bestDx = dx
                    bestDy=dy
        return bestDx,bestDy
    def multiAlign(a,b,loss):
        m,n = len(a),len(a[0])
        if m<=200 or n<=200:
        
            return singleAlign(a,b,loss)
        dx,dy = multiAlign(sk.transform.rescale(a,0.5),sk.transform.rescale(b,0.5),loss)
        dx = dx*2
        dy=dy*2

        print(f"{dx},{dy}")
        searchX = 2#int(0.05*dx)
        searchY = 2#int(0.05*dy)
        newX,newY = singleAlign(a,b,loss,dy+-searchY,dy+searchY+1,dx+-searchX,dx+searchX+1)
        return newX, newY

    m,n=len(b),len(b[0])
    c1=0.1
    bc = b[int(c1*m):int((1-c1)*m),int(c1*n):int((1-c1)*n)]
    gc = g[int(c1*m):int((1-c1)*m),int(c1*n):int((1-c1)*n)]
    rc = r[int(c1*m):int((1-c1)*m),int(c1*n):int((1-c1)*n)]

    # ag = shift(g,*(multiAlign(bc,gc,l2Loss)))
    # ar = shift(r,*(multiAlign(bc,rc,l2Loss)))
    dgx,dgy=multiAlign(f(bc),f(gc),loss)
    ag = shift(g,dgx,dgy)
    drx,dry = multiAlign(f(bc),f(rc),loss)

    ar = shift(r,drx,dry)
    im_out = np.dstack([ar, ag, b])

    skio.imsave(fname, (im_out*255).astype(np.uint8))

    if show:
        skio.imshow(im_out)
        skio.show()
    return ((dgx,dgy),(drx,dry))
