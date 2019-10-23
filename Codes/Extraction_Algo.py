import cv2
import pywt
from scipy.linalg import svd

location = '../Dataset/'

def FrameCapture(path):
    red_wframes = []
    green_wframes = []
    blue_wframes = []
    cap = cv2.VideoCapture(path)
    count = 0 
    success = 1
    while success:
        success,image = cap.read()
        if success:
            b,g,r = cv2.split(image)
            red_wframes.append(r)
            green_wframes.append(g)
            blue_wframes.append(b)
    
    print("-------------- Video was split into frames successfully and splitted into RGB frames --------------")
    return red_wframes,green_wframes,blue_wframes

def applyDWT(redframes,greenframes,blueframes):
    for i in range(0,redframes,1):
        LLR1,_ = pywt.dwt(redframes[i],'db1')
        LLG1,_ = pywt.dwt(greenframes[i],'db1')
        LLB1,_ = pywt.dwt(blueframes[i],'db1')

        _,HHR2 = pywt.dwt(LLR1,'db1')
        _,HHG2 = pywt.dwt(LLG1,'db1')
        _,HHB2 = pywt.dwt(LLB1,'db1')
        
        print("-------------- 2 rounds of DWT applied and the HH values of each channel returned  --------------")
        return HHR2,HHG2,HHB2

def applySVD(mat):
    U,S,VT = svd(mat,full_matrices=True)
    return U,S,VT
    print("-------------- SVD applied on HH sub band successful --------------")

# def Extractor