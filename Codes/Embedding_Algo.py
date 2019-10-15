import cv2
import random
import pywt
from numpy import diag,dot,zeros,array,linalg
import numpy as np
from PIL import Image
from scipy.linalg import svd

location = "F:/NIIT University/4 Year/Machine Learning/Term Project/Dataset/"
location2 = "F:/NIIT University/4 Year/Machine Learning/Term Project/Dataset/Logo.png"
frames = []
wmkd_frames = []
alpha = 0.1

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
    Frames = array(frames)
    print("-------------- Video was split into frames successfully and splitted into RGB frames --------------")
    return count,Frames

#Perfprming frame subtraction by selecting random frame from each channel and subtracting it over all channels
def Frame_Subtract(nof,Frames):
    sub_frames = []
    random_frame = random.choice(Frames)
    for i in range(0,len(Frames)):
        sub_frames.append(frames[i]-random_frame)
    print("-------------- Random frames from each channel were selected and subtracted accordingly --------------")
    Sub_frames = array(sub_frames)
    return Sub_frames, random_frame#,sub_green_frames,sub_blue_frames,dr,dg,db

def RGB_Splitter(rf):
    blue, green, red = cv2.split(rf)
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
    return red,green,blue

def ApplyDWT_Logo(r,g,b):
    _, HHR = pywt.dwt(r,'db1')
    _, HHG = pywt.dwt(g,'db1')
    _, HHB = pywt.dwt(b,'db1') 
    print("-------------- DWT applied on Logo --------------")
    return HHR,HHG,HHB

def ApplySVD(mat):
    U, S, VT = svd(mat, full_matrices=True)
    print("-------------- Applying SVD on the HH sub bands successful --------------")
    return U,S,VT

def Singular_U_Adder(u1,u2):
    m,_ = u1.shape
    n,_ = u2.shape
    u3 = np.zeros((m,n))
    for y in range(0,n):
        for x in range(0,m):
            u3[y][x] = u1[y][x] + 0.1*u2[y][x]
    return u3

def Singular_S_Adder(s1,s2):
    m = s1.shape[0]
    n = s2.shape[0]
    s3 = np.zeros((min(m,n)),dtype=int)
    for y in range(0,min(m,n)):
            s3[y] = s1[y] + 0.1*s2[y]
    return s3

def Singular_VT_Adder(vt1,vt2):
    m,_ = vt1.shape
    n,_ = vt2.shape
    vt3 = np.zeros((m,m))
    for y in range(0,m):
        for x in range(0,m):
            vt3[y][x] = vt1[y][x] + 0.1*vt2[y][x]
    return vt3
 
def InverseSVD(u,s,vt):
    m,_ = u.shape
    n,_ = vt.shape
    Sigma = np.zeros((m,n))
    for i in range(min(m,n)):
        Sigma[i,i] = s[i]
    B = np.dot(u,np.dot(Sigma,vt))
    return B

def IDWT(mat):
    temp = pywt.idwt(None,mat, 'db1')
    return pywt.idwt(temp,None, 'db1')

def Reconstruct_Frame(wr,wg,wb):
    wmk_frame = cv2.merge([wr,wg,wb])
    cv2.imwrite(location+'Watermarked.png',wmk_frame)
    return wmk_frame

def Add_to_Subtracted_Frames(wmk_frame,n,sfs):
    a = sfs.shape[0]
    for i in range(0,5):
        image = sfs[i]+wmk_frame
        print('Watermark added to the subtracted frame')
        wmkd_frames.append(image)
        print('Added to the list')
        cv2.imwrite(location + 'Wmk%d.png' % i,image)
        print('Written to file system')