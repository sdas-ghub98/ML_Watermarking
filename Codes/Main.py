import Embedding_Algo as ea
import numpy as np
import cv2

if __name__ == '__main__': 
    
    #EMBEDDING ALGORITHM

    #Splitting the video frames and then splitting them on RGB frames
    nof,R,G,B,rf = ea.FrameCapture(ea.location + "Akiyo Video.mp4")
    
    #Performing frame subtraction on all the channels
    sbrf,sbgf,sbbf = ea.Frame_Subtract(nof,R,G,B,rf)

    #Applying two rounds of DWT on the random frame
    a1,b1,c1 = ea.ApplyDWT_Frames(rf)

    #Applying SVD on the random frame
    u1,s1,vt1 = ea.ApplySVD(a1)
    u2,s2,vt2 = ea.ApplySVD(b1)
    u3,s3,vt3 = ea.ApplySVD(c1)
    
    #Applying DWT once on the splitted logo and acquiring the HH sub band
    a2 = ea.ApplyDWT_Logo()

    #Applying SVD once on the HH sub DWT-ized logo
    u4,s4,vt4 = ea.ApplySVD(a2)
    
    #Now adding the singular matrices
    
    s7 = ea.Singular_S_Adder(s1,s4)
    s8 = ea.Singular_S_Adder(s2,s4)
    s9 = ea.Singular_S_Adder(s3,s4)

    print("-------------- Sums done! --------------")

    #Reconstructing three SVD matrixes for R,G and B channels separately
    dR = ea.InverseSVD(u1,vt1,u4,vt4,s7)
    dG = ea.InverseSVD(u2,vt2,u4,vt4,s8)
    dB = ea.InverseSVD(u3,vt3,u4,vt4,s9)
    
    print("-------------- Inverse SVDs calculcated on the RGB channels --------------")
    
    # #Treat these matrices as HH value and compute the inverse DWT twice on them
    eR = ea.IDWT(dR)
    eG = ea.IDWT(dG)
    eB = ea.IDWT(dB)

    print("-------------- Inverse DWT applied twice on the RGB channels --------------")

    # Reconstructing the watermarked frame
    #wmkd_red, wmkd_green, wmkd_blue = 
    ea.Add_to_Subtracted_Frames(eR,eG,eB,sbrf,sbgf,sbbf,nof)