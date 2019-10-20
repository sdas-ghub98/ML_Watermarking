import cv2
import random
import pywt
from numpy import dot,zeros,array,ndarray,resize
from scipy.linalg import svd

location = "../Dataset/"
location2 = "../Dataset/Logo.png"

# Function to extract frames 
def FrameCapture(path): 
    frames = []
    red_frames =[]
    green_frames = []
    blue_frames = []
    cap = cv2.VideoCapture(path)  
    count = 0
    success = 1
    while success:
        success, image = cap.read()
        if success:
            frames.append(image)
            b,g,r = cv2.split(image)
            red_frames.append(r)
            green_frames.append(g)
            blue_frames.append(b)
            count += 1  
    rf = random.choice(frames)
    print("-------------- Video was split into frames successfully and splitted into RGB frames --------------")
    return count,red_frames, green_frames, blue_frames,rf

#Perfprming frame subtraction by selecting random frame from each channel and subtracting it over all channels
def Frame_Subtract(nof,Red,Green,Blue,rf):
    SR = []
    SG = []
    SB = []
    rb,rg,rr = cv2.split(rf)

    for i in range(0,nof):
        SR.append(cv2.subtract(Red[i],rb))
        SG.append(cv2.subtract(Green[i],rg))
        SB.append(cv2.subtract(Blue[i],rr))
    
    print("-------------- Random frames from each channel were selected and subtracted accordingly --------------")
    return SR,SG,SB

#Function that applies DWT twice on the frames
def ApplyDWT_Frames(rf):
    b,g,r = cv2.split(rf)

    LLR1,_ = pywt.dwt(r,'db1')
    LLG1,_ = pywt.dwt(g,'db1')
    LLB1,_ = pywt.dwt(b,'db1')
        
    #Take the HH sub band and apply DWT again here
    _,HHR2 = pywt.dwt(LLR1,'db1')
    _,HHG2 = pywt.dwt(LLG1,'db1')
    _,HHB2 = pywt.dwt(LLB1,'db1')

    print("-------------- DWT applied twice on frames (once on normal and then on LL sub band.) Returning the HH sub bands --------------")
    return HHR2,HHG2,HHB2
        
#Applying single round of DWT to the logo image
def ApplyDWT_Logo():
    wmk_image = cv2.imread(location2)
    gray_wmk = cv2.cvtColor(wmk_image,cv2.COLOR_BGR2GRAY)
    _,HHLogo = pywt.dwt(gray_wmk,'db1')
    print("-------------- DWT applied on Logo --------------")
    return HHLogo

# Function to apply SVD on the matrice passed as a parameter
def ApplySVD(mat):
    U, S, VT = svd(mat, full_matrices=True)
    print("-------------- Applying SVD on the HH sub bands successful --------------")
    return U,S,VT

#Function to add the singular matrix S
def Singular_S_Adder(s1,s2):
    a = resize(s2,(88,))
    s3 = s1 + a
    return s3
 
#Function to calculate inverse SVD
def InverseSVD(u1,vt1,u2,vt2,s3):
    m,_ = u1.shape
    n,_ = vt1.shape
    Sigma = zeros((m,n))
    for i in range(min(m,n)):
        Sigma[i,i] = s3[i]
    B = dot(u1,dot(Sigma,vt1))
    return B

#Function to calculate inverse DWT
def IDWT(mat):
    temp = pywt.idwt(mat, None,'db1')
    temp2 = pywt.idwt(None,temp,'db1') 
    return temp2

#Function to add the watermark on each channel of subtracted frames
def Add_to_Subtracted_Frames(wr,wg,wb,sbrf,sbgf,sbbf,nof):
    wmkd_frames = []
    temp = cv2.merge((wr,wg,wb))
    for i in range(0,len(sbrf)):
        wR = cv2.add(sbrf[i],wr,dtype=cv2.CV_64F)
        wG = cv2.add(sbgf[i],wg,dtype=cv2.CV_64F)
        wB = cv2.add(sbbf[i],wb,dtype=cv2.CV_64F)
        temp2 = cv2.merge((wR,wG,wB))
        final = cv2.add(temp,temp2,dtype=cv2.CV_64F)
        cv2.imshow('Final frame', final)
        cv2.waitKey(30)
        cv2.destroyAllWindows()
        wmkd_frames.append(final)


