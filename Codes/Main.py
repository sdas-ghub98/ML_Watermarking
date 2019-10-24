import Embedding_Algo as ea
import numpy as np
import cv2

if __name__ == '__main__': 
    
    #EMBEDDING ALGORITHM

    #Splitting the video frames and then splitting them into RGB frames
    nof,R,G,B,rf = ea.FrameCapture(ea.location + "Akiyo Video.mp4")
    
    #Performing frame subtraction on all the channels and returning them as lists
    sbrf,sbgf,sbbf = ea.Frame_Subtract(nof,R,G,B,rf)

    #Applying two rounds of DWT on the random frame
    LLR1,HHR1,LLR2,HHR2,LLG1,HHG1,LLG2,HHG2,LLB1,HHB1,LLB2,HHB2 = ea.ApplyDWT_Frames(rf)

    #Applying SVD on the random frame
    u1,s1,vt1 = ea.ApplySVD(HHR2)
    u2,s2,vt2 = ea.ApplySVD(HHG2)
    u3,s3,vt3 = ea.ApplySVD(HHB2)
    
    # cv2.imshow('Red SVD U Frame',u1)
    # cv2.waitKey(1000)
    # cv2.imshow('Red S Frame',s1)
    # cv2.waitKey(1000)
    # cv2.imshow('Red VT Frame',vt1)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()

    # cv2.imshow('Green SVD U Frame',u2)
    # cv2.waitKey(1000)
    # cv2.imshow('Green S Frame',s2)
    # cv2.waitKey(1000)
    # cv2.imshow('Blue VT Frame',vt2)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()

    # cv2.imshow('Blue SVD U Frame',u3)
    # cv2.waitKey(1000)
    # cv2.imshow('Blue S Frame',s3)
    # cv2.waitKey(1000)
    # cv2.imshow('Blue VT Frame',vt3)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()

    
    # Applying DWT once on the splitted logo and acquiring the HH sub band
    a2 = ea.ApplyDWT_Logo()

    # Applying SVD once on the HH sub DWT-ized logo
    u4,s4,vt4 = ea.ApplySVD(a2)
    
    # #Now adding the singular matrices
    s7 = ea.Singular_S_Adder(s1,s4)
    s8 = ea.Singular_S_Adder(s2,s4)
    s9 = ea.Singular_S_Adder(s3,s4)

    print("-------------- Singular value of random frame and logo added --------------")

    # Reconstructing three SVD matrixes for R,G and B channels separately
    dR = ea.InverseSVD(u1,vt1,u4,vt4,s7)
    dG = ea.InverseSVD(u2,vt2,u4,vt4,s8)
    dB = ea.InverseSVD(u3,vt3,u4,vt4,s9)
    # cv2.imshow('ISVD R frame',dR)
    # cv2.waitKey(1000)
    # cv2.imshow('ISVD G frame',dG)
    # cv2.waitKey(1000)
    # cv2.imshow('ISVD B frame',dB)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    
    print("-------------- Inverse SVDs calculcated on the RGB channels --------------")
    
    # Treat these matrices as HH value and compute the inverse DWT twice on them
    eR = ea.IDWT(dR,LLR2,HHR1)
    eG = ea.IDWT(dG,LLG2,HHG1)
    eB = ea.IDWT(dB,LLB2,HHB1)
    
    #Merging the inverse DWT-ized channels into a single frame and converting the data type to uint8 to obtain optimal result
    f = cv2.merge((eB,eG,eR)).astype(np.uint8)
    
    # cv2.imshow('Reconstructed frame',f)
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()

    print("-------------- Inverse DWT applied twice on the RGB channels --------------")

    # Adding the watermarked frame to the subtracted frames
    watermarked_frames = ea.Add_to_Subtracted_Frames(f,sbrf,sbgf,sbbf,nof)

    print("-------------- Watermarked frames constructed --------------")

    # Creating video using these watermarked frames
    ea.Create_Video_From_Frames(watermarked_frames)

    print("-------------- Watermarked video constructed --------------")
    
