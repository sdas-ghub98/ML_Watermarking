import Embedding_Algo as ea
import Extraction_Algo as exa
import numpy as np
import cv2
import timeit

if __name__ == '__main__': 
##########################################################################################################################################    
    #EMBEDDING ALGORITHM

    start = timeit.default_timer()

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
    
    #Merging the watermarked channels after Inverse DWT
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
    
    stop = timeit.default_timer()

    print('Total time taken for embedding algo to work : ',stop-start,'seconds\n\n\n\n\n\n')


##########################################################################################################################################


    #EXTRACTION ALGORITHM

    #Taking the first frame of watermarked video and splitting it into R,G and B channels
    rw,gw,bw = exa.FrameCapture(exa.location + 'Watermarked Video.avi')
    print("-------------- Taking the first frame from watermarked video and splitting into RGB frames --------------")
    # cv2.imshow('Red frame',rw)
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()

    #Applying DWT twice on the frame
    HHWR,HHWG,HHWB = exa.applyDWT(rw,gw,bw)
    # cv2.imshow('HH Red',HHWR)
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()

    #Applying SVD on the HH sub band
    uwr,swr,vtwr = exa.applySVD(HHWR)
    # cv2.imshow('Red U',uwr)
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()
    uwg,swg,vtwg = exa.applySVD(HHWG)
    uwb,swb,vtwb = exa.applySVD(HHWB)

    #Taking the first frame of the non-watermarked video
    rnw,bnw,gnw = exa.FrameCapture(exa.location + 'Akiyo Video.mp4')
    # cv2.imshow('Red non watermarked',rnw)
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()
    print("-------------- Taking the first frame of non-watermarked video and splitting into RGB frames --------------")

    #Applying DWT twice on the non-watermarked frame
    HHNWR,HHNWG,HHNWB = exa.applyDWT(rnw,gnw,bnw)
    # cv2.imshow('Red non watermarked HH',HHNWR)
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()

    #Applying SVD on the non-watermarked frame
    unwr,snwr,vtnwr = exa.applySVD(HHNWR)
    unwg,snwg,vtnwg = exa.applySVD(HHNWG)
    unwb,snwb,vtnwb = exa.applySVD(HHNWB)

    #Subtracting the singular values
    red_logo = uwr - unwr
    green_logo = uwg - unwg
    blue_logo = uwb - unwb

    #Get SVD values from original logo file
    ur,vtr,ug,vtg,ub,vtb = exa.GetOriginalUSVT()

    # Reconstructing the watermark
    res = exa.Watermark_Processing(red_logo,green_logo,blue_logo,ur,vtr,ug,vtg,ub,vtb)

    cv2.imshow('Logo',res)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    
