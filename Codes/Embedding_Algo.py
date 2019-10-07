import cv2
import random
import pywt
import numpy as np
from PIL import Image

location = "F:/NIIT University/4 Year/Machine Learning/Term Project/Dataset/"
#frames = []
#gray_frames = []
#sub_frames = []
# Function to extract frames 
def FrameCapture(path): 
    cap = cv2.VideoCapture(path) 
    count = 0
    success = 1

    while success:
        success, image = cap.read()
        #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(location + "Frames/image%d.png"%count,image)
        #frames.append(image)
        count += 1    
    print("Video was split into frames successfully")
    #print(gray_image)
    return count

def FrameSubtraction(nof):
    d = random.randrange(0,nof,1)
    rnd_frame_path = location+"Frames/image"+str(d)+".png"
    src1 = cv2.imread(rnd_frame_path)
    for i in range(0,nof,1):
        path2 = location+"Frames/image"+str(i)+".png" 
        src2 = cv2.imread(path2)
        im3 = src1-src2
        cv2.imwrite(location + "Sub_Frames/sub%d.png" % i, im3)
        #im3.delete()
    #print(sub_frames)
    return d

def RGB_Splitter(rf):
    src = cv2.imread(location + "Frames/image"+str(rf)+".png")
    red = src[:,:,2]
    green = src[:,:,1]
    blue = src[:,:,0]
    print(red)
    print(green)
    print(blue)
    return red, green, blue

def ApplyDWT(r,g,b):
    [LLR1,HHR1] = pywt.dwt(r,'db1')
    [LLG1,HHG2] = pywt.dwt(g,'db1')
    [LLB1,HHB2] = pywt.dwt(b,'db1')

    #Take the HH sub band and apply DWT again here
    [LLR2,HHR2] = pywt.dwt(LLR1,'db1')
    [LLG2,HHG2] = pywt.dwt(LLG1,'db1')
    [LLB2,HHB2] = pywt.dwt(LLB1,'db1')

    return HHR2,HHG2,HHB2

def ApplySVD(hr,hg,hb):
    ur, sr, vhr = np.linalg.svd(hr, full_matrices=False)
    ug, sg, vhg = np.linalg.svd(hg, full_matrices=False)
    ub, sb, vhb = np.linalg.svd(hb, full_matrices=False)
    return ur,sr,vhr,ug,sg,vhg,ub,sb,vhb


def RGB_Merger(c):
    rgbArray = np.zeros((512,512,3), 'uint8')
    rgbArray[..., 0] = (c[0]+c[1]+c[2])*256
    rgbArray[..., 1] = (c[3]+c[4]+c[5])*256
    rgbArray[..., 2] = (c[6]+c[7]+c[8])*256
    img = Image.fromarray(rgbArray,'RGB')
    img.save(location + 'cf.png')

def FrameConstruction(nof):
    
    p1 = location + 'cf.png'
    p2 = location + 'lg.png'
    src1 = cv2.imread(p1)
    src2 = cv2.imread(p2)
    im3 = src1 + src2
    cv2.imwrite(location+'wmk_frame.png',im3)

    for i in range(0, nof, 1):
        im_location = location + 'Sub_Frames/sub'+str(i) +'.png'
        src3 = cv2.imread(im_location)
        src3 = im3 + src3
        cv2.imwrite(location + "Sub_Frames/sub%d.png" % i, src3)
    
    return im3


