import Codes.Embedding_Algo as cea
import Codes.Logo as logo
if __name__ == '__main__': 
    # Calling the function 
    nof = cea.FrameCapture(cea.location + "Akiyo Video.mp4")
    random_frame = cea.FrameSubtraction(nof)
    r, g, b = cea.RGB_Splitter(random_frame)
    cea.ApplyDWT(r,g,b)
    #cea.ApplySVD(r,g,b)