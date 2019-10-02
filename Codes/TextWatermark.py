import cv2
import random
import os

# Function to extract frames 
def FrameCapture(path): 
    vidObj = cv2.VideoCapture(path) 
    count = 0
    success = 1

    while success:
        success, image = vidObj.read() 
        cv2.imwrite("F:\\NIIT University\\4 Year\\Machine Learning\\Term Project\\Dataset\\Frames\\frame%d.jpg" % count, image)
        count += 1

def FrameSubtraction():
    d = random.randrange(0,301,1)
    rnd_frame_path = "F:\\NIIT University\\4 Year\\Machine Learning\\Term Project\\Dataset\\Frames\\frame"+str(d)+".jpg"
    src1 = cv2.imread(rnd_frame_path)
    for i in range(0,301):
        if(i == d):
            continue
        else:
            path2 = "F:\\NIIT University\\4 Year\\Machine Learning\\Term Project\\Dataset\\Frames\\frame"+str(d)+".jpg" 
            src2 = cv2.imread(path2)
            im3 = src1 - src2
            cv2.imwrite("F:\\NIIT University\\4 Year\\Machine Learning\\Term Project\\Dataset\\Sub_Frames\\sub%d.jpg" % i, im3)
            #im3.delete()
    return src1

def RGB_Splitter(image):

# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture("F:\\NIIT University\\4 Year\\Machine Learning\\Term Project\\Dataset\\Akiyo Video.mp4")
    random_frame = FrameSubtraction()
    red, green, blue = RGB_Splitter(random_frame)
