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
red_frames = []
green_frames = []
blue_frames = []
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
            red = image[:,:,2]
            green = image[:,:,1]
            blue = image[:,:,0]    
            red_frames.append(red)
            green_frames.append(green)
            blue_frames.append(blue)
        count += 1  
    print("--------------Video was split into frames successfully and splitted into RGB frames --------------")
    return count

#Perfprming frame subtraSction by selecting random frame from each channel and subtracting it channel wise
def Frame_Subtract(nof):
    sub_red_frames = []
    sub_green_frames = []
    sub_blue_frames = []
    dr = random.choice(red_frames)
    dg = random.choice(green_frames)
    db = random.choice(blue_frames)
    
    for i in range(0,len(red_frames)):
        sub_red_frames.append(red_frames[i]-dr)
        sub_green_frames.append(green_frames[i]-dg)
        sub_blue_frames.append(green_frames[i]-db)
    
    print("-------------- Random frames from each channel were selected and subtracted accordingly --------------")
    return sub_red_frames,sub_green_frames,sub_blue_frames,dr,dg,db

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
    red = src[:,:,2]
    green = src[:,:,1]
    blue = src[:,:,0]
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

def Add_Sub_Frames(wr,wg,wb,sr,sg,sb):
    for i in range(0,len(sr)):
        sr[i]=sr[i]+wr
        sg[i]=sg[i]+wg
        sb[i]=sb[i]+wb
    print("So far so good")