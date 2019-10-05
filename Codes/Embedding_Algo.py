import cv2
import random
import os
import pywt

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
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(location + "/Frames/gray_image%d.png"%count,gray_image)
        #frames.append(image)
        count += 1    
    print("Video was split into frames successfully")
    #print(gray_image)
    return count

def FrameSubtraction(nof):
    d = random.randrange(0,nof,1)
    rnd_frame_path = location + "Frames/gray_image"+str(d)+".png"
    src1 = cv2.imread(rnd_frame_path)
    for i in range(0,nof,1):
        path2 = location +"Frames/gray_image"+str(i)+".png" 
        src2 = cv2.imread(path2)
        im3 = src1-src2
        cv2.imwrite(location + "Sub_Frames/sub%d.png" % i, im3)
        #im3.delete()
    #print(sub_frames)
    return d

def RGB_Splitter(rf):
    src = cv2.imread(location + "Frames/gray_image"+str(rf)+".png")
    red = src[:,:,2]
    green = src[:,:,1]
    blue = src[:,:,0]
    return red, green, blue

def ApplyDWT(r,g,b):
    #DWT on the basis/random frame
    cAR,cDR = pywt.dwt(r,'db1')
    cAG,cDG = pywt.dwt(g,'db1')
    cAB,cDB = pywt.dwt(b,'db1')

    #Take the HH sub band and apply DWT again here
    #Take the result of this and apply SVD there
