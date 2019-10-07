import cv2
import pywt
import numpy as np
import Embedding_Algo as ea
from PIL import Image

location = "F:/NIIT University/4 Year/Machine Learning/Term Project/Dataset/Logo.png"

def RGB_Splitter():
    src = cv2.imread(location)
    red = src[:,:,2]
    green = src[:,:,1]
    blue = src[:,:,0]
    return red, green, blue

def ApplyDWT_Logo(r,g,b):
    [LLR, HHR] = pywt.dwt(r,'db1')
    [LLG, HHG] = pywt.dwt(g,'db1')
    [LLB, HHB] = pywt.dwt(b,'db1') 
    return HHR,HHG,HHB
 
def ApplySVD_Logo(r,g,b):
    ur, sr, vhr = np.linalg.svd(r, full_matrices=False)
    ug, sg, vhg = np.linalg.svd(g, full_matrices=False)
    ub, sb, vhb = np.linalg.svd(b, full_matrices=False)
    return ur,sr,vhr,ug,sg,vhg,ub,sb,vhb

def RGB_Merger_Logo(c):
    rgbArray = np.zeros((512,512,3), 'uint8')
    rgbArray[..., 0] = (c[0]+c[1]+c[2])*256
    rgbArray[..., 1] = (c[3]+c[4]+c[5])*256
    rgbArray[..., 2] = (c[6]+c[7]+c[8])*256
    img = Image.fromarray(rgbArray,'RGB')
    img.save(ea.location + 'lg.png')
