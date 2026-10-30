import os
import time
import numpy as np
import matplotlib.pyplot as plt
from part4a import computeH, warpImageNearestNeighbor, warpImageBilinear, SETS

RECTS = {
    "rect1": {"file": "rect1_copy.jpeg",
              "points": [[604.25,84.5],[442.75,31.25],[543,276.5],[393.5,228.75]],
              "target": [[125,0],[0,0],[125,150],[0,150]],
              "order": [1,0,2,3,1],
              "zoom": [104,224,297,417]},
    "rect2": {"file": "rect2_copy.jpeg",
              "points": [[505.75,391],[560.25,388.75],[505.75,451.75],[559.75,453.25]],
              "target": [[0,0],[50,0],[0,50],[50,50]],
              "order": [0,1,3,2,0],
              "zoom": [665,785,1418,1538]},
}

def drawPoints(ax,im,points,numbered=True):
    ax.imshow(im)
    ax.axis("off")
    ax.scatter(points[:,0],points[:,1],s=60,c="red",edgecolors="white")
    if numbered:
        for i in range(len(points)):
            x,y = points[i]
            ax.annotate(str(i+1),(x,y),xytext=(6,-6),textcoords="offset points",color="white",fontsize=11,weight="bold")

def figures(out="figures"):
    os.makedirs(out, exist_ok=True)

    for name in SETS:
        s = SETS[name]
        im1 = plt.imread(s["im1"]+".jpeg")
        im2 = plt.imread(s["im2"]+".jpeg")
        pts1 = np.array(s["im1Points"])
        pts2 = np.array(s["im2Points"])

        fig,axs = plt.subplots(1,2,figsize=(14,5.4))
        drawPoints(axs[0],im1,pts1)
        drawPoints(axs[1],im2,pts2)
        fig.tight_layout(pad=0.3)
        fig.savefig(out+"/"+name+"_points.jpg",dpi=110)
        plt.close(fig)

        print(name, "H =")
        print(computeH(pts1,pts2))

    for name in RECTS:
        r = RECTS[name]
        im = plt.imread(r["file"])/255.0
        points = np.array(r["points"])

        fig,ax = plt.subplots(figsize=(7, 7*im.shape[0]/im.shape[1]))
        drawPoints(ax,im,points,numbered=False)
        box = points[r["order"]]
        ax.plot(box[:,0],box[:,1],c="red",lw=2)
        fig.tight_layout(pad=0)
        fig.savefig(out+"/"+name+"_points.jpg",dpi=110)
        plt.close(fig)

        H = computeH(points,np.array(r["target"]))
        print(name, "H =")
        print(H)

        start = time.time()
        nn = warpImageNearestNeighbor(im,H)
        print("nearest neighbor took", time.time()-start, "s")

        start = time.time()
        bil = warpImageBilinear(im,H)
        print("bilinear took", time.time()-start, "s")

        plt.imsave(out+"/"+name+"_nn.jpg",np.clip(nn,0,1))
        plt.imsave(out+"/"+name+"_bilinear.jpg",np.clip(bil,0,1))

        top,bottom,left,right = r["zoom"]
        for method,warped in [("nn",nn),("bilinear",bil)]:
            crop = np.clip(warped[top:bottom, left:right],0,1)
            crop = np.repeat(np.repeat(crop,5,axis=0),5,axis=1)
            plt.imsave(out+"/"+name+"_"+method+"_zoom.png",crop)


if __name__ == "__main__":
    figures()
