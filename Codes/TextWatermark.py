import cv2 
  
# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
  
        # Saves the frames with frame-count 
        cv2.imwrite("frame%d.jpg" % count, image) 
  
        count += 1

#This function picks one random frame and splits into RGB values
def RGB_Splitter(path):

#This function will apply SVD transformation on the frame 
def SVD_applier():

#This will combine the RGB frames into one
def RGB_Combiner():

# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture("D:\\NIIT University\\4 Year\\Machine Learning\\Term Project\\Akiyo Video.mp4") 
