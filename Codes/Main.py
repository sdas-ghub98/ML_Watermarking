import Embedding_Algo as cea
import Logo as logo
if __name__ == '__main__': 
    
    #Embedding algorithm
    nof = cea.FrameCapture(cea.location + "Akiyo Video.mp4")
    random_frame = cea.FrameSubtraction(nof)
    r1, g1, b1 = cea.RGB_Splitter(random_frame)
    a1,b1,c1,d1,e1,f1 = cea.ApplyDWT(r1,g1,b1)
    
    #cea.ApplySVD(r1,g1,b1)
    r2,g2,b2 = logo.RGB_Splitter()
    a2,b2,c2,d2,e2,f2 = logo.ApplyDWT_Logo(r2,g2,b2)
    #logo.ApplySVD_Logo

    #Extracting algorithm
