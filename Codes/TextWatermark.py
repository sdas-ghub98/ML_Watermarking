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
    print("Video was split into frames successfully")

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
    if(d):
        print("Random video frame was selected successfully and frames were subtracted")
    return d

def RGB_Splitter(random_frame_number):
    path = "F:\\NIIT University\\4 Year\\Machine Learning\\Term Project\\Dataset\\Frames\\frame"+str(random_frame_number)+".jpg"
    rnd_frame = cv2.imread(path)
    red = rnd_frame[:,:,2]
    green = rnd_frame[:,:,1]
    blue = rnd_frame[:,:,0]
    """if(red):
        print("Red Split success!!")
    if(green):
        print("Green Split success!!")
    if(blue):
        print("Blue Split success!!")"""
    return red, green, blue

#def ApplySVD(r,g,b):


# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture("F:\\NIIT University\\4 Year\\Machine Learning\\Term Project\\Dataset\\Akiyo Video.mp4")
    random_frame = FrameSubtraction()
    r, g, b = RGB_Splitter(random_frame)
    #ApplySVD(r,g,b)
