import cv2
import random
import pywt
from numpy import diag,dot,zeros
import numpy as np
from PIL import Image
from scipy.linalg import svd

location = "F:/NIIT University/4 Year/Machine Learning/Term Project/Dataset/"
location2 = "F:/NIIT University/4 Year/Machine Learning/Term Project/Dataset/Logo.png"
frames = []
wmkd_frames = []

# Function to extract frames 
def FrameCapture(path): 
    cap = cv2.VideoCapture(path) 
    success, image = cap.read()
    count = 0
    success = 1
    while success:
        success, image = cap.read()
        if success:
            frames.append(image)
        count += 1  
    print("--------------Video was split into frames successfully and splitted into RGB frames --------------")
    return count

#Perfprming frame subtraction by selecting random frame from each channel and subtracting it over all channels
def Frame_Subtract(nof):
    sub_frames = []
    random_frame = random.choice(frames)
    for i in range(0,len(frames)):
        sub_frames.append(frames[i]-random_frame)
    print("-------------- Random frames from each channel were selected and subtracted accordingly --------------")
    return sub_frames, random_frame#,sub_green_frames,sub_blue_frames,dr,dg,db

def RGB_Splitter(rf):
    red = rf[:,:,2]
    green = rf[:,:,1]
    blue = rf[:,:,0]
    return red,green,blue

def ApplyDWT_Frames(r,g,b):
    LLR1,_ = pywt.dwt(r,'db1')
    LLG1,_ = pywt.dwt(g,'db1')
    LLB1,_ = pywt.dwt(b,'db1')

    #Take the HH sub band and apply DWT again here
    _,HHR2 = pywt.dwt(LLR1,'db1')
    _,HHG2 = pywt.dwt(LLG1,'db1')
    _,HHB2 = pywt.dwt(LLB1,'db1')

    print("-------------- DWT applied twice once on normal and then on LL sub band. Returning the HH sub bands --------------")
    return HHR2,HHG2,HHB2

def RGB_Splitter_Logo():
    src = cv2.imread(location2)
    blue, green, red = cv2.split(src)
    print("-------------- Logo splitted successfully --------------")
    return red, green, blue

def ApplyDWT_Logo(r,g,b):
    _, HHR = pywt.dwt(r,'db1')
    _, HHG = pywt.dwt(g,'db1')
    _, HHB = pywt.dwt(b,'db1') 
    print("-------------- DWT applied on Logo --------------")
    return HHR,HHG,HHB

def ApplySVD(hr,hg,hb):
    Ur, sr, VTr = svd(hr)
    Ug, sg, VTg = svd(hg)
    Ub, sb, VTb = svd(hb)
    print("-------------- Applying SVD on the HH sub bands successful --------------")
    return Ur,sr,VTr,Ug,sg,VTg,Ub,sb,VTb

def InverseSVD(u,s,vt,A):
    #print(u.shape)
    #print(s.shape)
    #print(vt.shape)
    
    Sigma = zeros((A.shape[0],A.shape[1]))
    #Sigma=np.pad(Sigma,(44,44),mode='constant')
    Sigma[:A.shape[1],:A.shape[1]] = diag(s)
    B = u.dot(Sigma.dot(vt))
    return B

def IDWT(mat):
    temp = pywt.idwt(None,mat, 'db1')
    return pywt.idwt(temp,None, 'db1')

def Reconstruct_Frame(wr,wg,wb):
    wmk_frame = cv2.merge([wr,wg,wb])
    cv2.imwrite(location+'Watermarked.png',wmk_frame)
    return wmk_frame

def Add_to_Subtracted_Frames(wmk_frame,n,sfs):
    for i in range(0,5):
        wmkd_frames[i] = sfs[i] + wmk_frame
        cv2.imwrite(location + 'wmk%d'%i, wmkd_frames[i])


    