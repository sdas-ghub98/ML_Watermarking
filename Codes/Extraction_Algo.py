import cv2
import pywt
from scipy.linalg import svd
import numpy as np

location = '../Dataset/'

def FrameCapture(path):
    cap = cv2.VideoCapture(path)
    count = 0 
    success = 1
    success,image = cap.read()
    b,g,r = cv2.split(image)
    
    print("-------------- Taking the first video frame and splitting into RGB frames --------------")
    return r,g,b

def applyDWT(redframe,greenframe,blueframe):
    LLR1,_ = pywt.dwt(redframe,'db1')
    LLG1,_ = pywt.dwt(greenframe,'db1')
    LLB1,_ = pywt.dwt(blueframe,'db1')

    _,HHR2 = pywt.dwt(LLR1,'db1')
    _,HHG2 = pywt.dwt(LLG1,'db1')
    _,HHB2 = pywt.dwt(LLB1,'db1')
        
    print("-------------- 2 rounds of DWT applied and the HH values of each channel returned  --------------")
    return HHR2,HHG2,HHB2

def applySVD(mat):
    U,S,VT = svd(mat,full_matrices=True)
    return U,S,VT
    print("-------------- SVD applied on HH sub band successful --------------")

def Watermark_Processing(r,g,b):
    src = cv2.imread(location + 'logo.png')
    b2,g2,r2 = cv2.split(src)
    ur,sr,vtr = svd(r2,full_matrices=True)
    ug,sg,vtg = svd(g2,full_matrices=True)
    ub,sb,vtb = svd(b2,full_matrices=True)

    r3 = applyInverseSVD(ur,r,vtr)
    g3 = applyInverseSVD(ug,g,vtg)
    b3 = applyInverseSVD(ub,b,vtb)

    ir = applyIDWT(r3)
    ig = applyIDWT(g3)
    ib = applyIDWT(b3)

    res = cv2.merge((ib,ig,ir)).astype(uint8)
    return res

def applyInverseSVD(u,s,vt):
    m,_ = u.shape()
    n,_ = vt.shape()
    Sigma = np.zeros((m,n))
    for i in range(min(m,n)):
        Sigma[i,i]=s[i]
    B = np.dot(u,np.dot(s,vt))
    return B

def applyIDWT(mat):
    temp = pywt.idwt(None,mat,'db1')
    temp2 = pywt.idwt(temp,None,'db1')
    return temp2