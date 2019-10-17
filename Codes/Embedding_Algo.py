import cv2
import random
import pywt
from numpy import dot,zeros,array,linalg,reshape,transpose
from scipy.linalg import svd


location = "../Dataset/"
location2 = "../Dataset/Logo.png"
alpha = 0.1
frames = []
wmkd_frames = []
red_frames =[]
green_frames = []
blue_frames = []

# Function to extract frames 
def FrameCapture(path): 
    cap = cv2.VideoCapture(path) 
    success, image = cap.read()
    count = 0
    success = 1
    while success:
        success, image = cap.read()
        if success:
            #frames.append(image)
            b,g,r = cv2.split(image)
            red_frames.append(r)
            green_frames.append(g)
            blue_frames.append(b)
            count += 1  
    #Frames = array(frames)
    Red = array(red_frames)
    Green = array(green_frames)
    Blue = array(blue_frames)
    print("-------------- Video was split into frames successfully and splitted into RGB frames --------------")
    return count,Red, Green, Blue

#Perfprming frame subtraction by selecting random frame from each channel and subtracting it over all channels
def Frame_Subtract(nof,Red,Green,Blue):
    sub_red_frames = []
    sub_green_frames = []
    sub_blue_frames = []
    dr = random.choice(Red)
    dg = random.choice(Green)
    db = random.choice(Blue)
    cv2.imwrite(location+'Random_Red.png',dr)
    cv2.imwrite(location+'Random_Green.png',dg)
    cv2.imwrite(location+'Random_Blue.png',db)

    for i in range(0,len(Red)):
        sub_red_frames.append(Red[i]-dr)
        sub_green_frames.append(Green[i]-dg)
        sub_blue_frames.append(Blue[i]-db)
    SR = array(sub_red_frames)
    SG = array(sub_green_frames)
    SB = array(sub_blue_frames)
    print("-------------- Random frames from each channel were selected and subtracted accordingly --------------")

    return SR,SG,SB,dr,dg,db

#Function that applies DWT twice on the frames
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

def Singular_U_Adder(u1,u2):
    m,_ = u1.shape
    n,_ = u2.shape
    u3 = zeros((m,n))
    for y in range(0,n):
        for x in range(0,m):
            u3[y][x] = u1[y][x] + 0.1*u2[y][x]
    return u3

def Singular_S_Adder(s1,s2):
    m = s1.shape[0]
    n = s2.shape[0]
    s3 = zeros((min(m,n)),dtype=int)
    for y in range(0,min(m,n)):
            s3[y] = s1[y] + 0.1*s2[y]
    return s3

def Singular_VT_Adder(vt1,vt2):
    m,_ = vt1.shape
    n,_ = vt2.shape
    vt3 = zeros((m,m))
    for y in range(0,m):
        for x in range(0,m):
            vt3[y][x] = vt1[y][x] + 0.1*vt2[y][x]
    return vt3
 
def InverseSVD(u,s,vt):
    m,_ = u.shape
    n,_ = vt.shape
    Sigma = zeros((m,n))
    for i in range(min(m,n)):
        Sigma[i,i] = s[i]
    B = dot(u,dot(Sigma,vt))
    return B

def IDWT(mat):
    return pywt.idwt(None,mat, 'db1')

def Reconstruct_Frame(wr,wg,wb):
    wmk_frame = cv2.merge([wb,wg,wr])
    cv2.imwrite(location+'Watermarked.png',wmk_frame)
    return wmk_frame

def Add_to_Subtracted_Frames(wmk_frame,n,sbrf,sbgf,sbbf):
    for i in range(0,5):
        sub_frame = cv2.merge([sbbf[i],sbgf[i],sbrf[i]])
        print(sub_frame.shape) # (264, 352, 3)
        print(wmk_frame.shape) # (264, 176, 3)
        #SHRUTI - Write the code here to add element by element
        wmkd_frames.append(wmk_frame + sub_frame)
        cv2.imwrite(location + 'Wmk%d.png'%i,wmkd_frames[i])
    
    