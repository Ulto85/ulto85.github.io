import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import distance_transform_edt
def computeH(im1_pts,im2_pts):
    A = []
    b= []
    for i in range(len(im1_pts)):
        (x,y)=im1_pts[i]
        (u,v)=im2_pts[i]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
        b.extend([u, v])
    h= np.linalg.lstsq(A,b)[0]
    h=np.append(h,1)
    return h.reshape((3,3))
def invWarp(im,H,f,bbox=None):
    height, width = im.shape[:2]
    corners = np.array([
        [0, 0, 1],
        [width - 1, 0, 1],
        [width - 1, height - 1, 1],
        [0, height - 1, 1]
    ])
    corners = np.transpose(corners)

    newCorners = H@corners
    newCorners/=newCorners[2]


    #bounding box
    if bbox is None:
        xmin = int(np.floor(newCorners[0].min()))
        xmax = int(np.ceil(newCorners[0].max()))
        ymin = int(np.floor(newCorners[1].min()))
        ymax = int(np.ceil(newCorners[1].max()))
    else:
        xmin,xmax,ymin,ymax = bbox

    warpedImage = np.zeros((ymax - ymin + 1, xmax - xmin + 1) + im.shape[2:])
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=bool)

    H_inv = np.linalg.inv(H)

    for r in range(ymax-ymin+1):
        for c in range(xmax-xmin+1):
            x_out = c+xmin
            y_out = r + ymin
            p = H_inv @ np.array([x_out, y_out, 1.0])
            x_src = p[0]/p[2]
            y_src = p[1]/p[2]
            f(im,mask,warpedImage,x_src,y_src,r,c)
    return warpedImage,mask

def nearestNeighbor(im,mask,warpedImage,x_src,y_src,r,c):
    x = round(x_src)
    y = round(y_src)
    height,width = im.shape[:2]
    if 0 <= x < width and 0 <= y < height:
                warpedImage[r, c] = im[y, x]
                mask[r, c] = True

def bilinearinterp(im,mask,warpedImage,x_src,y_src,r,c):
    height,width = im.shape[:2]
    if not (0 <= x_src <= width - 1 and
                    0 <= y_src <= height - 1):
                return
    x0 = int(np.floor(x_src))
    y0 = int(np.floor(y_src))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)

    dx = x_src - x0
    dy = y_src - y0


    warpedImage[r, c] = (
        (1-dx) * (1- dy) * im[y0, x0]
        + dx * (1 - dy) * im[y0, x1]
        + (1 - dx) * dy * im[y1, x0]
        + dx * dy * im[y1, x1]
    )

    mask[r,c] = True

def warpImageNearestNeighbor(im,H):
      w,m= invWarp(im,H,nearestNeighbor)
      return w
def warpImageBilinear(im,H):
      w,m = invWarp(im,H,bilinearinterp)
      return w

     
def part3():
    
    im1 = plt.imread("rect1_copy.jpeg") / 255.0

    pts1 = np.array([
        [604.25, 84.5],
        [442.75, 31.25],
        [543, 276.5],
        [393.5, 228.75]
    ])

    target1 = np.array([
        [125, 0],
        [0, 0],
        [125, 150],
        [0, 150]
    ])


    im2 = plt.imread("rect2_copy.jpeg") / 255.0

    pts2 = np.array([
        [505.75, 391],
        [560.25, 388.75],
        [505.75, 451.75],
        [559.75, 453.25]
    ])

    target2 = np.array([
        [0, 0],
        [50, 0],
        [0, 50],
        [50, 50]
    ])

  
    H1 = computeH(pts1, target1)
    H2 = computeH(pts2, target2)


    nn1 = warpImageNearestNeighbor(im1, H1)
    bil1 = warpImageBilinear(im1, H1)

    nn2 = warpImageNearestNeighbor(im2, H2)
    bil2 = warpImageBilinear(im2, H2)

    plt.imsave("rect1_nn.jpg", np.clip(nn1, 0, 1))
    plt.imsave("rect1_bilinear.jpg", np.clip(bil1, 0, 1))
    plt.imsave("rect2_nn.jpg", np.clip(nn2, 0, 1))
    plt.imsave("rect2_bilinear.jpg", np.clip(bil2, 0, 1))



def mosaic(im1,im2,im1_pts,im2_pts,sigma=4):
    H = computeH(im1_pts,im2_pts)
    height1,width1 = im1.shape[:2]
    height2,width2 = im2.shape[:2]


    corners = np.array([[0, width1-1, width1-1, 0],
                        [0, 0, height1-1, height1-1],
                        [1, 1, 1, 1]])
    newCorners = H@corners
    newCorners = newCorners/newCorners[2]


    xmin = int(np.floor(min(newCorners[0].min(), 0)))
    xmax = int(np.ceil(max(newCorners[0].max(), width2-1)))
    ymin = int(np.floor(min(newCorners[1].min(), 0)))
    ymax = int(np.ceil(max(newCorners[1].max(), height2-1)))

    warped1,mask1 = invWarp(im1,H,bilinearinterp,(xmin,xmax,ymin,ymax))


    top = -ymin
    left = -xmin
    placed2 = np.zeros_like(warped1)
    mask2 = np.zeros_like(mask1)
    placed2[top:top+height2, left:left+width2] = im2
    mask2[top:top+height2, left:left+width2] = True


    dist1 = distance_transform_edt(mask1)
    dist2 = distance_transform_edt(mask2)
    total = dist1 + dist2
    total[total==0] = 1   
    weight1 = np.expand_dims(dist1/total, axis=2)
    weight2 = 1 - weight1


    low1 = cv2.GaussianBlur(warped1,(0,0),sigma)
    low2 = cv2.GaussianBlur(placed2,(0,0),sigma)
    high1 = warped1 - low1
    high2 = placed2 - low2

    low = weight1*low1 + weight2*low2
    high = np.where(np.expand_dims(dist1 > dist2, axis=2), high1, high2)
    return np.clip(low + high, 0, 1)

SETS = {
    "set1": {"im1": "set1a_copy", "im2": "set1b_copy",
             "im1Points": [[849,135],[858,171],[474,102],[474,118],[915,497],[947,496],[780,275],[805,520],[479,557],[534,173]],
             "im2Points": [[508,178],[513,211],[128,81],[128,99],[555,504],[579,500],[445,298],[466,527],[119,591],[199,169]]},
    "set2": {"im1": "set2a_copy", "im2": "set2b_copy",
             "im1Points": [[509,446],[534,442],[513,477],[537,475],[29,497],[62,500],[27,534],[60,534],[376,448]],
             "im2Points": [[812,472],[840,464],[817,504],[846,503],[367,526],[392,529],[365,558],[393,559],[665,475]]},
    "set3": {"im1": "set3a_copy", "im2": "set3b_copy",
             "im1Points": [[600,280],[602,375],[617,279],[619,373],[497,379],[498,403],[789,271],[794,409],[568,350]],
             "im2Points": [[265,314],[267,413],[284,315],[286,412],[141,416],[143,444],[449,315],[455,443],[230,385]]},
}

def part4():
    for name in SETS:
        s = SETS[name]
        im1 = plt.imread(s["im1"]+".jpeg")/255.0
        im2 = plt.imread(s["im2"]+".jpeg")/255.0
        result = mosaic(im1,im2,s["im1Points"],s["im2Points"])
        plt.imsave(name+"_mosaic.jpg", result)



part3()
part4()
