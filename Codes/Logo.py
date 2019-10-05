import cv2
import pywt
location = "F:/NIIT University/4 Year/Machine Learning/Term Project/Dataset/Logo.png"

def RGB_Splitter():
    src = cv2.imread(location)
    red = src[:,:,2]
    green = src[:,:,1]
    blue = src[:,:,0]
    return red, green, blue

def ApplyDWT_Logo(r,g,b):
    cAR,cDR = pywt.dwt(r,'db1')
    cAG,cDG = pywt.dwt(g,'db1')
    cAB,cDB = pywt.dwt(b,'db1') 
    return cAR,cDR,cAG,cDG,cAB,cDB
 
#def ApplySVD_Logo():


