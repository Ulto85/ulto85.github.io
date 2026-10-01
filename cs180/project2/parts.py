import numpy as np
import skimage as sk
import skimage.io as skio
import scipy
import cv2
import matplotlib.pyplot as plt
from align_image_code import align_images
import os 
def naiveConvolution(img,kernel):
    #using same padding
    kh,kw = kernel.shape
    H,W = img.shape
    ans = np.zeros(img.shape)
    paddingW = (kw-1)//2
    paddingH = (kh-1)//2
    img = np.pad(img, ((paddingH, paddingH), (paddingW, paddingW)))
    

    for y in range(H):
        for x in range(W):
            tempans=0
            for j in range(kh):
                for i in range(kw):
                    tempans+=img[y+j][x+i]*kernel[j][i]
            ans[y,x]=tempans
    ans=np.array(ans)
    return ans

def optimizedConvolution(img,kernel):
    kh,kw = kernel.shape
    H,W = img.shape
    ans = np.zeros(img.shape)
    paddingW = (kw-1)//2
    paddingH = (kh-1)//2

    img = np.pad(img, ((paddingH, paddingH), (paddingW, paddingW)))
    
    for y in range(H):
        for x in range(W):
            ans[y,x]=np.sum(img[y:y+kh,x:x+kw]*kernel)
    ans=np.array(ans)
    return ans
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
   
    plt.imsave(f"{dirName}/{im1Name}_aligned_fft_hf.jpg",np.log(np.abs(np.fft.fftshift(np.fft.fft2(im1_aligned)))),cmap='gray')
    plt.imsave(f"{dirName}/{im2Name}_aligned_fft_lf.jpg",np.log(np.abs(np.fft.fftshift(np.fft.fft2(im2_aligned)))),cmap='gray')

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
        plt.imsave(f"{dirName}/{im2Name}_aligned_lf_pass.jpg",lowsfimg,cmap='gray')
        plt.imsave(f"{dirName}/{im2Name}_aligned_fft_lf_pass.jpg",np.log(np.abs(np.fft.fftshift(np.fft.fft2(lowsfimg)))),cmap='gray')
        


        ## High SF
        hkernel = cv2.getGaussianKernel(int(highsfsigma)*6+1,highsfsigma)
        hkernel = np.outer(hkernel,hkernel.T)
        highsfimg2 = optimizedConvolution(highsfimg,hkernel)
        highsfimg=highsfimg-highsfimg2
        plt.imsave(f"{dirName}/{im1Name}_aligned_hf_pass.jpg",highsfimg,cmap='gray')

        plt.imsave(f"{dirName}/{im1Name}_aligned_fft_hf_pass.jpg",np.log(np.abs(np.fft.fftshift(np.fft.fft2(highsfimg)))),cmap='gray')

        return (highsfimg+lowsfimg)

    sigma1 = s1
    sigma2 = s2
    hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
    plt.imsave(f"{dirName}/hybrid_{sigma1}_{sigma2}.jpg",hybrid,cmap='gray')
    plt.imsave(f"{dirName}/hybrid_fft.jpg",np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid)))),cmap='gray')


    # plt.imshow(hybrid)
    # plt.show()


def optimizedConvolution3D(img,kernel):
    kh,kw = kernel.shape
    H,W,C = img.shape
    ans = np.zeros(img.shape)
    paddingW = (kw-1)//2
    paddingH = (kh-1)//2

    img = np.pad(img, ((paddingH, paddingH), (paddingW, paddingW),(0,0)))
    for c in range(C):
        for y in range(H):
            for x in range(W):
                ans[y,x,c]=np.sum(img[y:y+kh,x:x+kw,c]*kernel)

    return ans

def part1_1():
    im = skio.imread("input/mecs180.jpeg",as_gray=True)
    dx = np.array([[-1,0,1]])
    dy=np.array([[-1],[0],[1]])
    box = np.array([[1/9]*3 for _ in range(3)])
    dy=optimizedConvolution(im,dy)
    dx = optimizedConvolution(im,dx)
    box = optimizedConvolution(im,box)
    # threshold = 0.05
    # img =np.abs(img)>threshold
    plt.imsave("part1.1/me_grayscale.jpg",im,cmap="gray")
    plt.imsave("part1.1/dy.jpg",dy,cmap="gray")
    plt.imsave("part1.1/dx.jpg",dx,cmap="gray")
    plt.imsave("part1.1/box.jpg",box,cmap="gray")
def part1_2():
    im = skio.imread("input/cameraman.png",as_gray=True)
    dx = np.array([[-1,0,1]])
    dy=np.array([[-1],[0],[1]])

    DX = optimizedConvolution(im,dx)
    DY= optimizedConvolution(im,dy)

    GRADIENTIMAGE = np.sqrt(DX**2+DY**2)
    magnitude = GRADIENTIMAGE
    plt.imsave("part1.2/dx.jpg",DX,cmap='gray')
    plt.imsave("part1.2/dy.jpg",DY,cmap='gray')
    plt.imsave("part1.2/magnitude.jpg",magnitude,cmap='gray')
    for t in np.arange(0.05,0.51,0.05):
        t=round(t,2)
        final = magnitude>t
        plt.imsave(f"part1.2/cameramanThresh{t}.jpg",final,cmap="gray")


def part1_3_a():
    kernel =  cv2.getGaussianKernel(5,1.5)
    kernel = np.outer(kernel,kernel.T)
    im = skio.imread("input/cameraman.png",as_gray=True)
    im=optimizedConvolution(im,kernel)
    dx = np.array([[-1,0,1]])
    dy=np.array([[-1],[0],[1]])

    DX = optimizedConvolution(im,dx)
    DY= optimizedConvolution(im,dy)
    plt.imsave("part1.3a/dx.jpg",DX,cmap='gray')
    plt.imsave("part1.3a/dy.jpg",DY,cmap='gray')

    GRADIENTIMAGE = np.sqrt(DX**2+DY**2)
    magnitude = GRADIENTIMAGE
    plt.imsave("part1.3a/magnitude.jpg",magnitude,cmap='gray')
    for t in np.arange(0.05,0.51,0.05):
        t=round(t,2)
        final = magnitude>t
        plt.imsave(f"part1.3a/cameramanThresh{t}.jpg",final,cmap='gray')
def part1_3_b():
    kernel =  cv2.getGaussianKernel(5,1)
    kernel = np.outer(kernel,kernel.T)
    im = skio.imread("input/cameraman.png",as_gray=True)
    im=optimizedConvolution(im,kernel)
    dx = np.array([[-1,0,1]])
    dx = optimizedConvolution(kernel,dx)
    dy=np.array([[-1],[0],[1]])
    dy = optimizedConvolution(kernel,dy)
    plt.imsave("part1.3b/gaussdx.jpg",dx,cmap='gray')
    plt.imsave("part1.3b/gaussdy.jpg",dy,cmap='gray')

    DX = optimizedConvolution(im,dx)
    DY= optimizedConvolution(im,dy)
    plt.imsave("part1.3b/dx.jpg",DX,cmap='gray')
    plt.imsave("part1.3b/dy.jpg",DY,cmap='gray')

    GRADIENTIMAGE = np.sqrt(DX**2+DY**2)
    magnitude = GRADIENTIMAGE
    plt.imsave("part1.3b/magnitude.jpg",magnitude,cmap='gray')
    for t in np.arange(0.05,0.51,0.05):
        t=round(t,2)
        final = magnitude>t
        plt.imsave(f"part1.3b/cameramanThresh{t}.jpg",final,cmap='gray')


def part2_1():
    ###taj
    im = skio.imread("input/taj.jpg")
    kernel =  cv2.getGaussianKernel(5,1)
    kernel = np.outer(kernel,kernel.T)
    blur = optimizedConvolution3D(im,kernel)
    skio.imsave("part2.1/taj_blur.jpg",(blur).astype(np.uint8))
    def visualize_frequency(x):
            max_abs = np.max(np.abs(x))
        
            return np.clip(0.5 + x / (2 * max_abs), 0, 1)
    highF = im -blur
    plt.imsave("part2.1/taj_HighF.jpg",visualize_frequency(highF),cmap='gray')
    for alpha in range(1,6):


        ans = np.zeros(im.shape)
        ans=im + alpha * (highF)

        ans = np.clip(ans, 0, 255)
        skio.imsave(f"part2.1/taj_sharp_{alpha}.jpg",(ans).astype(np.uint8))
    ##lecun
    im = skio.imread("input/lecun.jpg")
    kernel =  cv2.getGaussianKernel(5,1)
    kernel = np.outer(kernel,kernel.T)
    blur = optimizedConvolution3D(im,kernel)
    skio.imsave("part2.1/lecun_blur.jpg",(blur).astype(np.uint8))

    highF = im -blur
    plt.imsave("part2.1/lecun_HighF.jpg",visualize_frequency(highF),cmap='gray')
    for alpha in range(1,6):


        ans = np.zeros(im.shape)
        ans=im + alpha * (highF)
        ans = np.clip(ans, 0, 255)
        skio.imsave(f"part2.1/lecun_sharp_{alpha}.jpg",(ans).astype(np.uint8))
    ###lecun resharp
    #blur is blurred
    im=blur
    blur = optimizedConvolution3D(im,kernel)
    highF=im-blur
    for alpha in range(1,6):
    
    
            ans = np.zeros(im.shape)
            ans=im + alpha * (highF)
            ans = np.clip(ans, 0, 255)
            skio.imsave(f"part2.1/lecun_resharp_{alpha}.jpg",(ans).astype(np.uint8))



def part2_2_real():
    #part2_2("aayan1.png","akhil1.png",3,8)
    #part2_2("nutmeg.jpg","DerekPicture.jpg",8,10)
    part2_2("bear.jpg","oski.png",3,8)

def part2_3():
    def gaussianStack(N,I,s):
            if N==0:
                return []
            K = cv2.getGaussianKernel(s*6+1,s)
            K = np.outer(K,K.T)
            
            I=optimizedConvolution3D(I,K)
            return [I]+gaussianStack(N-1,I,s*2)
    def laplacianStack(gArr):
        ans = []
        for i in range(len(gArr)-1):
            ans.append(gArr[i]-gArr[i+1])
        ans.append(gArr[-1])
        return ans
    
    img1 = plt.imread("input/spline/apple.jpeg")/255.0
    img2 = plt.imread("input/spline/orange.jpeg")/255.0
    gaussImage1 =[img1]+gaussianStack(5,img1,1)
    gaussImage2 = [img2]+gaussianStack(5,img2,1)
    laplacianImage1 = laplacianStack(gaussImage1)
    laplacianImage2=laplacianStack(gaussImage2)
   
    # for i in range(len(laplacianImage1)):
    #     im1 = gaussMask[i]*laplacianImage1[i]
    #     im2 = (1-gaussMask[i])*laplacianImage2[i]
    #     blends.append(im1+im2)

    #     sums.append(np.sum([blends[-1],prev],axis=0))
    #     prev=sums[-1]
    #     apples.append(im1)
    #     oranges.append(im2)
    
    levels=range(len(laplacianImage1))
    def visualize_laplacian(x):
        max_abs = np.max(np.abs(x))
        return np.clip(0.5 + x / (2 * max_abs), 0, 1)
    for l in levels:
        # plt.imsave(f"part2.3/figure/apple_{l}.jpg",apples[l]/255.0)#.astype(np.uint8))
        # plt.imsave(f"part2.3/figure/orange_{l}.jpg",oranges[l]/255.0)#.astype(np.uint8))
        # plt.imsave(f"part2.3/figure/sums_{l}.jpg",sums[l]/255.0)#.astype(np.uint8))
    
        plt.imsave(f"part2.3/apple-gaussian/{l}.jpg",gaussImage1[l])
        plt.imsave(f"part2.3/orange-gaussian/{l}.jpg",gaussImage2[l])
        plt.imsave(f"part2.3/apple-laplacian/{l}.jpg",visualize_laplacian(laplacianImage1[l]))
        plt.imsave(f"part2.3/orange-laplacian/{l}.jpg",visualize_laplacian(laplacianImage2[l]))




def part2_4_indepth():
    def gaussianStack(N,I,K):
        if N==0:
            return []
        I=optimizedConvolution3D(I,K)
        return [I]+gaussianStack(N-1,I,K)
    def laplacianStack(gArr):
        ans = []
        for i in range(len(gArr)-1):
            ans.append(gArr[i]-gArr[i+1])
        ans.append(gArr[-1])
        return ans
    K = cv2.getGaussianKernel(13,2)
    K = np.outer(K,K.T)
    img1 = plt.imread("input/spline/apple.jpeg")/255.0
    img2 = plt.imread("input/spline/orange.jpeg")/255.0
    gaussImage1 =[img1]+gaussianStack(5,img1,K)
    gaussImage2 = [img2]+gaussianStack(5,img2,K)
    laplacianImage1 = laplacianStack(gaussImage1)
    laplacianImage2=laplacianStack(gaussImage2)
    mask = np.zeros(img1.shape)
    mask[:, :img1.shape[1] // 2, :] = 1
    gaussMask=[mask]+gaussianStack(5,mask,K)
    def visualize_frequency(x):
            max_abs = np.max(np.abs(x))
        
            return np.clip(0.5 + x / (2 * max_abs), 0, 1)
    for i in range(len(gaussMask)):
        skio.imsave(f"part2.4-indepth/mask{i}.jpg",(gaussMask[i]*255).astype(np.uint8))
        #plt.imsave(f"part2.4-indepth/mask{i}.jpg",visualize_frequency(gaussMask[i]),cmap='gray')
    apples =[]
    oranges=[]
    prev=np.zeros(img1.shape)
    blends=[]
    sums=[]
    for i in range(len(laplacianImage1)):
        im1 = gaussMask[i]*laplacianImage1[i]
        im2 = (1-gaussMask[i])*laplacianImage2[i]
        blends.append(im1+im2)

        sums.append(np.sum([blends[-1],prev],axis=0))
        prev=sums[-1]
        apples.append(im1)
        oranges.append(im2)
    
    levels=range(len(laplacianImage1))
    
    levels=[0,2,4]
    for l in levels:
        # plt.imsave(f"part2.3/figure/apple_{l}.jpg",apples[l]/255.0)#.astype(np.uint8))
        # plt.imsave(f"part2.3/figure/orange_{l}.jpg",oranges[l]/255.0)#.astype(np.uint8))
        # plt.imsave(f"part2.3/figure/sums_{l}.jpg",sums[l]/255.0)#.astype(np.uint8))
    
        plt.imsave(f"part2.4-indepth/masked_apple_{l}.jpg",visualize_frequency(apples[l]))
        plt.imsave(f"part2.4-indepth/masked_orange.jpg",visualize_frequency(oranges[l]))
        plt.imsave(f"part2.4-indepth/masked_blends{l}.jpg",visualize_frequency(blends[l]))
    plt.imsave(f"part2.4-indepth/reconstructed_apple.jpg",np.clip(np.sum(apples,axis=0),0,1))
    plt.imsave(f"part2.4-indepth/reconstructed_orange.jpg",np.clip(np.sum(oranges,axis=0),0,1))
    plt.imsave(f"part2.4-indepth/reconstructed.jpg",np.clip(np.sum(blends,axis=0),0,1))

    #plt.imsave(f"part2.3/figure/sums_{len(laplacianImage1)}.jpg",np.clip(np.sum(blends,axis=0),0,1))

def part2_4_moon():
    def gaussianStack(N,I,K):
        if N==0:
            return []
        I=optimizedConvolution3D(I,K)
        return [I]+gaussianStack(N-1,I,K)
    def laplacianStack(gArr):
        ans = []
        for i in range(len(gArr)-1):
            ans.append(gArr[i]-gArr[i+1])
        ans.append(gArr[-1])
        return ans
    K = cv2.getGaussianKernel(13,2)
    K = np.outer(K,K.T)
    img1 = plt.imread("input/earth.jpg")/255.0
    img2 = plt.imread("input/moon2.jpg")/255.0
    gaussImage1 =[img1]+gaussianStack(5,img1,K)
    gaussImage2 = [img2]+gaussianStack(5,img2,K)
    laplacianImage1 = laplacianStack(gaussImage1)
    laplacianImage2=laplacianStack(gaussImage2)
    mask = np.zeros(img1.shape)
    mask[:, :img1.shape[1] // 2, :] = 1
    gaussMask=[mask]+gaussianStack(5,mask,K)
    def visualize_frequency(x):
            max_abs = np.max(np.abs(x))
        
            return np.clip(0.5 + x / (2 * max_abs), 0, 1)
    
    skio.imsave(f"part2.4-moon/mask.jpg",(gaussMask[0]*255).astype(np.uint8))
    #plt.imsave(f"part2.4-indepth/mask{i}.jpg",visualize_frequency(gaussMask[i]),cmap='gray')
    apples =[]
    oranges=[]
    prev=np.zeros(img1.shape)
    blends=[]
    sums=[]
    for i in range(len(laplacianImage1)):
        im1 = gaussMask[i]*laplacianImage1[i]
        im2 = (1-gaussMask[i])*laplacianImage2[i]
        blends.append(im1+im2)

        sums.append(np.sum([blends[-1],prev],axis=0))
        prev=sums[-1]
        apples.append(im1)
        oranges.append(im2)
    
    levels=range(len(laplacianImage1))
    
    
    plt.imsave(f"part2.4-moon/reconstructed.jpg",np.clip(np.sum(blends,axis=0),0,1)) 

def part2_4_third():
    def gaussianStack(N,I,K):
        if N==0:
            return []
        I=optimizedConvolution3D(I,K)
        return [I]+gaussianStack(N-1,I,K)
    def laplacianStack(gArr):
        ans = []
        for i in range(len(gArr)-1):
            ans.append(gArr[i]-gArr[i+1])
        ans.append(gArr[-1])
        return ans
    K = cv2.getGaussianKernel(13,2)
    K = np.outer(K,K.T)
    img1 = plt.imread("input/sphere.jpg")/255.0
    img2 = plt.imread("input/campanile.jpg")/255.0
    # img1 = img1[:, :, :3]
    # img2 = img2[:, :, :3]
    gaussImage1 =[img1]+gaussianStack(5,img1,K)
    gaussImage2 = [img2]+gaussianStack(5,img2,K)
    laplacianImage1 = laplacianStack(gaussImage1)
    laplacianImage2=laplacianStack(gaussImage2)
    newmask = plt.imread("input/fourthmask.jpg")/255.0
    mask = np.zeros((newmask.shape[0],newmask.shape[1],1))
    mask[:,:,0]=newmask
    gaussMask=[mask]+gaussianStack(5,mask,K)
    def visualize_frequency(x):
            max_abs = np.max(np.abs(x))
        
            return np.clip(0.5 + x / (2 * max_abs), 0, 1)
    
    #skio.imsave(f"part2.4-third/mask.jpg",(gaussMask[0]*255).astype(np.uint8))
    #plt.imsave(f"part2.4-indepth/mask{i}.jpg",visualize_frequency(gaussMask[i]),cmap='gray')
    apples =[]
    oranges=[]
    prev=np.zeros(img1.shape)
    blends=[]
    sums=[]
    for i in range(len(laplacianImage1)):
        im1 = gaussMask[i]*laplacianImage1[i]
        im2 = (1-gaussMask[i])*laplacianImage2[i]
        blends.append(im1+im2)

        sums.append(np.sum([blends[-1],prev],axis=0))
        prev=sums[-1]
        apples.append(im1)
        oranges.append(im2)
    
    levels=range(len(laplacianImage1))
    
    
    plt.imsave(f"part2.4-third/reconstructed.jpg",np.clip(np.sum(blends,axis=0),0,1)) 
part2_4_third()


