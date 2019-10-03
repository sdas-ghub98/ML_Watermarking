import cv2
import random
import os
import pywt

location = "F:\\NIIT University\\4 Year\\Machine Learning\\Term Project\\Dataset\\"
# Function to extract frames 
def FrameCapture(path): 
    vidObj = cv2.VideoCapture(path) 
    count = 0
    success = 1

    while success:
        success, image = vidObj.read() 
        cv2.imwrite(location +"Frames\\frame%d.png" % count, image)
        count += 1
    print("Video was split into frames successfully")
    return count

def FrameSubtraction(npf):
    d = random.randrange(0,nof,1)
    print(d)
    rnd_frame_path = location + "Frames\\frame"+str(d)+".png"
    src1 = cv2.imread(rnd_frame_path)
    for i in range(0,nof,1):
        path2 = location +"Frames\\frame"+str(i)+".png" 
        src2 = cv2.imread(path2)
        im3 = src2-src1
        cv2.imwrite(location + "Sub_Frames\\sub%d.png" % i, im3)
            #im3.delete()
    if(d):
        print("Random video frame was selected successfully and frames were subtracted")
    return d

def RGB_Splitter(random_frame_number):
    path = location + "Frames\\frame"+str(random_frame_number)+".png"
    rnd_frame = cv2.imread(path)
    red = rnd_frame[:,:,2]
    green = rnd_frame[:,:,1]
    blue = rnd_frame[:,:,0]
    return red, green, blue

#def ApplySVD(r,g,b):
    

# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    nof = FrameCapture(location + "Akiyo Video.mp4")
    #for i in range(0,nof,30):
    random_frame = FrameSubtraction(nof)
    r, g, b = RGB_Splitter(random_frame)
    #ApplySVD(r,g,b)
