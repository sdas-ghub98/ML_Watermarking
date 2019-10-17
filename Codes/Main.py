import Embedding_Algo as ea
import numpy as np
import cv2

if __name__ == '__main__': 
    
    #EMBEDDING ALGORITHM

    #Splitting the video frames and then splitting them on RGB frames
    nof,R,G,B = ea.FrameCapture(ea.location + "Akiyo Video.mp4")
    
    #Performing frame subtraction on all the channels
    sbrf,sbgf,sbbf,rr,rg,rb = ea.Frame_Subtract(nof,R,G,B)

    #Applying two rounds of DWT on the random frame
    a1,b1,c1 = ea.ApplyDWT_Frames(rr,rg,rb)

    #Applying SVD on the random frame
    u1,s1,vt1 = ea.ApplySVD(a1)
    u2,s2,vt2 = ea.ApplySVD(b1)
    u3,s3,vt3 = ea.ApplySVD(c1)
    
    #Applying DWT once on the splitted logo and acquiring the HH sub band
    a2 = ea.ApplyDWT_Logo()

    #Applying SVD once on the HH sub DWT-ized logo
    u4,s4,vt4 = ea.ApplySVD(a2)
    
    #Now adding the SVD matrices
    
    #For Red Frame
    u7 = ea.Singular_U_Adder(u1,u4)
    s7 = ea.Singular_S_Adder(s1,s4)
    vt7 = ea.Singular_VT_Adder(vt1,vt4)

    #For Green Frame
    u8 = ea.Singular_U_Adder(u2,u4)
    s8 = ea.Singular_S_Adder(s2,s4)
    vt8 = ea.Singular_VT_Adder(vt2,vt4)

    #For Blue Frame
    u9 = ea.Singular_U_Adder(u3,u4)
    s9 = ea.Singular_S_Adder(s3,s4)
    vt9 = ea.Singular_VT_Adder(vt3,vt4)

    print("-------------- Sums done! --------------")

    #Reconstructing three SVD matrixes for R,G and B channels separately
    
    d1 = ea.InverseSVD(u7,s7,vt7)
    d2 = ea.InverseSVD(u8,s8,vt8)
    d3 = ea.InverseSVD(u9,s9,vt9)

    print("-------------- Inverse SVDs calculcated on the RGB channels --------------")

    #Treat these matrices as High value and obtain the SVD
    e1 = ea.IDWT(d1)
    e2 = ea.IDWT(d2)
    e3 = ea.IDWT(d3)

    print("-------------- Inverse DWT applied twice on the RGB channels --------------")

    #Reconstructing the watermarked frame
    wmk_frame = ea.Reconstruct_Frame(e1,e2,e3)

    #Adding it to the subtracted frames
    ea.Add_to_Subtracted_Frames(wmk_frame,nof,sbrf,sbgf,sbbf)
