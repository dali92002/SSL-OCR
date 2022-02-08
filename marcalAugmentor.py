import cv2
import numpy as np
import random

def augmentor(img):
    TH,TW=img.shape

    param_gamma_low=.3
    #param_gamma_low=.5 # Nacho fixed
    param_gamma_high=2

    param_mean_gaussian_noise=0
    param_sigma_gaussian_noise=100**0.5

    param_kanungo_alpha=2 # params controlling how much foreground and background pixels flip state
    param_kanungo_beta=2
    param_kanungo_alpha0=1
    param_kanungo_beta0=1
    param_kanungo_mu=0
    param_kanungo_k=2

    param_min_shear=-.5 # here a little bit more shear to the left than to the right
    param_max_shear=.25

    param_rotation=3 # plus minus angles for rotation

    param_scale=.2 # one plus minus parameter as scaling factor

    param_movement_BB=6 # translation for cropping errors in pixels

    # add gaussian noise
    gauss = np.random.normal(param_mean_gaussian_noise,param_sigma_gaussian_noise,(TH,TW))
    gauss = gauss.reshape(TH,TW)
    gaussiannoise = np.uint8(np.clip(np.float32(img) + gauss,0,255))

    # randomly erode, dilate or nothing
    # we could move it also after binarization
    kernel=np.ones((3,3),np.uint8)
    #a=random.choice([1,2,3])
    a=random.choice([2,3]) # Nacho fixed
    #a = 3 # Nacho fixed
    if a==1:
        gaussiannoise=cv2.dilate(gaussiannoise,kernel,iterations=1)
    elif a==2:
        gaussiannoise=cv2.erode(gaussiannoise,kernel,iterations=1)

    # add random gamma correction
    gamma=np.random.uniform(param_gamma_low,param_gamma_high)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    gammacorrected = cv2.LUT(np.uint8(gaussiannoise), table)

    # binarize image with Otsu
    otsu_th,binarized = cv2.threshold(gammacorrected,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Kanungo noise
    dist = cv2.distanceTransform(1-binarized, cv2.DIST_L1, 3)  # try cv2.DIST_L1 for newer versions of OpenCV
    dist2 = cv2.distanceTransform(binarized, cv2.DIST_L1, 3) # try cv2.DIST_L1 for newer versions of OpenCV

    dist = dist.astype('float64') # Tro add
    dist2 = dist2.astype('float64') # Tro add

    P=(param_kanungo_alpha0*np.exp(-param_kanungo_alpha * dist**2)) + param_kanungo_mu
    P2=(param_kanungo_beta0*np.exp(-param_kanungo_beta * dist2**2)) + param_kanungo_mu
    distorted=binarized.copy()
    distorted[((P>np.random.rand(P.shape[0],P.shape[1])) & (binarized==0))]=1
    distorted[((P2>np.random.rand(P.shape[0],P.shape[1])) & (binarized==1))]=0
    closing = cv2.morphologyEx(distorted, cv2.MORPH_CLOSE, np.ones((param_kanungo_k,param_kanungo_k),dtype=np.uint8))

    # apply binary image as mask and put it on a larger canvas
    pseudo_binarized = closing * (255-gammacorrected)
    canvas=np.zeros((3*TH,3*TW),dtype=np.uint8)
    canvas[TH:2*TH,TW:2*TW]=pseudo_binarized
    points=[]
    count = 0 # Tro add
    while(len(points)<1):
        count += 1 # Tro add
        if count > 50: # Tro add
            break # Tro add

        # random shear
        shear_angle=np.random.uniform(param_min_shear,param_max_shear)
        M=np.float32([[1,shear_angle,0],[0,1,0]])
        sheared = cv2.warpAffine(canvas,M,(3*TW,3*TH),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_CUBIC)

        # random rotation
        M = cv2.getRotationMatrix2D((3*TW/2,3*TH/2),np.random.uniform(-param_rotation,param_rotation),1)
        rotated = cv2.warpAffine(sheared,M,(3*TW,3*TH),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_CUBIC)

        # random scaling
        scaling_factor=np.random.uniform(1-param_scale,1+param_scale)
        scaled = cv2.resize(rotated,None,fx=scaling_factor,fy=scaling_factor,interpolation=cv2.INTER_CUBIC)

        # detect cropping parameters
        points = np.argwhere(scaled!=0)
        points = np.fliplr(points)

    if len(points) < 1: # Tro add
        return pseudo_binarized

    r = cv2.boundingRect(np.array([points]))

    #random cropping
    deltax=random.randint(-param_movement_BB,param_movement_BB)
    deltay=random.randint(-param_movement_BB,param_movement_BB)
    x1=min(scaled.shape[0]-1,max(0,r[1]+deltax))
    y1=min(scaled.shape[1]-1,max(0,r[0]+deltay))
    x2=min(scaled.shape[0],x1+r[3])
    y2=min(scaled.shape[1],y1+r[2])
    final_image=np.uint8(scaled[x1:x2,y1:y2])

    return final_image

if __name__ == '__main__':
    imgName = 'p03-080-05-02.png'
    img = cv2.imread('/home/lkang/datasets/iam_final_words/words/'+imgName, 0)
    out_imgs = [cv2.resize(augmentor(img), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA) for i in range(20)]
    final_img = np.vstack((img, *out_imgs))
    rate = 800 / final_img.shape[0]
    final_img2 = cv2.resize(final_img, (int(final_img.shape[1]*rate), 800), interpolation=cv2.INTER_AREA)
    cv2.imshow('Marcal_V3', final_img2)
    cv2.waitKey(0)

