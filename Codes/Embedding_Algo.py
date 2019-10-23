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
    # cv2.imshow('Red Frame',red_frames[0])
    # cv2.waitKey(1000)
    # cv2.imshow('Green Frame',green_frames[0])
    # cv2.waitKey(1000)
    # cv2.imshow('Blue Frame',blue_frames[0])
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    print("-------------- Video was split into frames successfully and splitted into RGB frames --------------")
    return count,red_frames, green_frames, blue_frames,rf

#Perfprming frame subtraction by selecting random frame from each channel and subtracting it over all channels
def Frame_Subtract(nof,Red,Green,Blue,rf):
    SR = []
    SG = []
    SB = []
    rb,rg,rr = cv2.split(rf)

    for i in range(0,nof,1):
        SR.append(cv2.subtract(Red[i],rr))
        SG.append(cv2.subtract(Green[i],rg))
        SB.append(cv2.subtract(Blue[i],rb))
    
    # cv2.imshow('Red Subtracted Frame',SR[0])
    # cv2.waitKey(1000)
    # cv2.imshow('Green Subtracted Frame',SG[0])
    # cv2.waitKey(1000)
    # cv2.imshow('Blue Subtracted Frame',SB[0])
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    print("-------------- Random frames from each channel were selected and subtracted accordingly --------------")
    return SR,SG,SB

#Function that applies DWT twice on the frames
def ApplyDWT_Frames(rf):
    b,g,r = cv2.split(rf)
    # cv2.imshow('Random frame',rf)
    # cv2.waitKey(1000)
    # cv2.imshow('Random red frame',r)
    # cv2.waitKey(1000)
    # cv2.imshow('Random green frame',g)
    # cv2.waitKey(1000)
    # cv2.imshow('Random blue frame',b)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()

    # Taking the original matrix and taking the cA(LL) values from DWT
    LLR1,_ = pywt.dwt(r,'db1')
    LLG1,_ = pywt.dwt(g,'db1')
    LLB1,_ = pywt.dwt(b,'db1')

    # cv2.imshow('Red DWT Frame',LLR1)
    # cv2.waitKey(1000)
    # cv2.imshow('Green DWT Frame',LLG1)
    # cv2.waitKey(1000)
    # cv2.imshow('Blue DWT Frame',LLB1)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()    

    #Take the HH sub band and apply DWT again here
    _,HHR2 = pywt.dwt(LLR1,'db1')
    _,HHG2 = pywt.dwt(LLG1,'db1')
    _,HHB2 = pywt.dwt(LLB1,'db1')

    # cv2.imshow('Red DWT Frame',HHR2)
    # cv2.waitKey(1000)
    # cv2.imshow('Green DWT Frame',HHG2)
    # cv2.waitKey(1000)
    # cv2.imshow('Blue DWT Frame',HHB2)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()

    print("-------------- DWT applied twice on frames (once on normal and then on LL sub band.) Returning the HH sub bands --------------")
    return HHR2,HHG2,HHB2
        
#Applying two rounds of DWT to the logo image
def ApplyDWT_Logo():
    wmk_image = cv2.imread(location2)
    gray_wmk = cv2.cvtColor(wmk_image,cv2.COLOR_BGR2GRAY)
    LLLogo,HHLogo = pywt.dwt(gray_wmk,'db1')
    # _,HHLogo = pywt.dwt(LLLogo,'db1')

    # cv2.imshow('Logo DWTized Frame',HHLogo)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    print("-------------- DWT applied on Logo --------------")
    return HHLogo

# Function to apply SVD on the matrice passed as a parameter
def ApplySVD(mat):
    U, S, VT = svd(mat, full_matrices=True)
    print("-------------- Applying SVD on the HH sub bands successful --------------")
    return U,S,VT

#Function to add the singular matrix S
def Singular_S_Adder(s1,s2):
    a = resize(s2,s1.shape)
    s3 = cv2.add(s1,a)
    # cv2.imshow('Watermarked S matrice',s3)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    return s3
 
#Function to calculate inverse SVD
def InverseSVD(u1,vt1,u2,vt2,s3):
    m,_ = u1.shape
    n,_ = vt1.shape
    #print(u1.shape,'\n\n',vt1.shape,'\n\n',u2.shape,'\n\n',vt2.shape,'\n\n',s3.shape)
    Sigma = zeros((m,n))
    for i in range(min(m,n)):
        Sigma[i,i] = s3[i]
    B = dot(u1,dot(Sigma,vt1))
    # print(B.shape)
    return B

#Function to calculate inverse DWT
def IDWT(mat):
    temp = pywt.idwt(mat,None,'db1')
    temp2 = pywt.idwt(None,temp, 'db1')
    print('-------------- Inverse DWT applied --------------')
    return temp2

#Function to add the watermark on each channel of subtracted frames
def Add_to_Subtracted_Frames(wr,wg,wb,sbrf,sbgf,sbbf,nof):
    wmkd_frames = []
    # print(wr.shape)
    # print(type(wr))
    # print(sbrf[0].shape)
    # print(type(sbrf[0]))
    for i in range(0,4):
        # wRGB = cv2.merge((sbrf[i],sbgf[i],sbbf[i]))
        # temp = cv2.add(wRGB,f,dtype=cv2.CV_64F) 
        # temp2 = cv2.merge((wR,wG,wB))
        # final = cv2.add(temp,temp2,dtype=cv2.CV_64F)
        t1 = cv2.add(sbrf[i],wr,dtype=cv2.CV_64F)
        t2 = cv2.add(sbgf[i],wg,dtype=cv2.CV_64F)
        t3 = cv2.add(sbrf[i],wb,dtype=cv2.CV_64F)
        cv2.imshow('Final red frame', t1)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cv2.imshow('Final green frame', t2)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cv2.imshow('Final blue frame', t3)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        # wmkd_frames.append(temp)