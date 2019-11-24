import cv2
import pywt
from scipy.linalg import svd
from numpy import zeros,dot,uint8

location = '../Dataset/'

def FrameCapture(path):
    cap = cv2.VideoCapture(path)
    count = 0
    success = 1
    success,image = cap.read()
    b,g,r = cv2.split(image)
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

def applyInverseSVD(u,s,vt):
    m,_ = u.shape
    n,_ = vt.shape
    Sigma = zeros((m,n))
    for i in range(min(m,n)):
        Sigma[i,i] = s[i,i]
    B = dot(u,dot(Sigma,vt))
    # print(B.shape)
    return B

def applyIDWT(a,b,c):
    temp = pywt.idwt(b,a,'db1')
    temp2 = pywt.idwt(temp,c,'db1')
    return temp2

def GetOriginalUSVT():
    src = cv2.imread(location + 'logo.png')
    b2,g2,r2 = cv2.split(src)
    
    LLR3,HHR3 = pywt.dwt(r2,'db1')
    LLR4,HHR4 = pywt.dwt(LLR3,'db1')
    LLG3,HHG3 = pywt.dwt(g2,'db1')
    LLG4,HHG4 = pywt.dwt(LLG3,'db1')
    LLB3,HHB3 = pywt.dwt(b2,'db1')
    LLB4,HHB4 = pywt.dwt(LLB3,'db1')

    ur,_,vtr = svd(HHR4,full_matrices=True)
    ug,_,vtg = svd(HHG4,full_matrices=True)
    ub,_,vtb = svd(HHB4,full_matrices=True)

    return ur,vtr,ug,vtg,ub,vtb,LLR3,HHR3,LLG3,HHG3,LLB3,HHB3,LLR4,HHR4,LLG4,HHG4,LLB4,HHB4
def Watermark_Processing(r,g,b,ur,vtr,ug,vtg,ub,vtb,LLR3,HHR3,LLG3,HHG3,LLB3,HHB3,LLR4,HHR4,LLG4,HHG4,LLB4,HHB4):

    r3 = applyInverseSVD(ur,r,vtr)
    g3 = applyInverseSVD(ug,g,vtg)
    b3 = applyInverseSVD(ub,b,vtb)

    ir = applyIDWT(r3,LLR4,HHR3)
    ig = applyIDWT(g3,LLG4,HHG3)
    ib = applyIDWT(b3,LLB4,HHB3)

    res = cv2.merge((ib,ig,ir)).astype(uint8)
    return res