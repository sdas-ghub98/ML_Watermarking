import Embedding_Algo as ea
import numpy as np
import cv2

if __name__ == '__main__': 
    
    #EMBEDDING ALGORITHM

    #Splitting the video frames and then splitting them into RGB frames
    nof,R,G,B,rf = ea.FrameCapture(ea.location + "Akiyo Video.mp4")
    
    #Performing frame subtraction on all the channels and returning them as lists
    #r,g,b = The original random frame in 3 channels
    sbrf,sbgf,sbbf = ea.Frame_Subtract(nof,R,G,B,rf)

    #Applying two rounds of DWT on the random frame
    a1,b1,c1 = ea.ApplyDWT_Frames(rf)

    #Applying SVD on the random frame
    u1,s1,vt1 = ea.ApplySVD(a1)
    u2,s2,vt2 = ea.ApplySVD(b1)
    u3,s3,vt3 = ea.ApplySVD(c1)
    
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
    eR = ea.IDWT(dR)
    eG = ea.IDWT(dG)
    eB = ea.IDWT(dB)
    f = cv2.merge((eR,eG,eB))
    # cv2.imshow('IDWT R frame',eR)
    # cv2.waitKey(1000)
    # cv2.imshow('IDWT G frame',eG)
    # cv2.waitKey(1000)
    # cv2.imshow('IDWT B frame',eB)
    # cv2.waitKey(1000)
    cv2.imshow('IDWT Merged frame',f)
    cv2.waitKey(4000)
    cv2.destroyAllWindows()

    print("-------------- Inverse DWT applied twice on the RGB channels --------------")
    
    # If we take the random frame, split into RGB and then add the modified RGB values to the original values and merge again
    # b2,g2,r2 = cv2.split(rf)
    # R = cv2.add(eR,r2,dtype=cv2.CV_64F)
    # G = cv2.add(eG,g2,dtype=cv2.CV_64F)
    # B = cv2.add(eB,b2,dtype=cv2.CV_64F)
    # h = cv2.merge((R,G,B))
    # cv2.imshow('Watermarked Frame',h)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()

    # t = cv2.add(sbrf[0],f,cv2.CV_64F)
    # cv2.imshow('Red Subtracted Merged frame',t)
    # cv2.waitKey(1000)
    
    # cv2.destroyAllWindows()
    # t = cv2.add(sb,eB)
    # b2,g2,r2 = cv2.split(rf)
    # B = cv2.add(t,b2,dtype=cv2.CV_64F)
    # f = cv2.merge((r2,g2,B))
    # cv2.imshow('Watermarked random frame',f)
    # cv2.waitKey(4000)
    # cv2.destroyAllWindows()

    # Reconstructing the watermarked frame
    ea.Add_to_Subtracted_Frames(eR,eG,eB,sbrf,sbgf,sbbf,nof)