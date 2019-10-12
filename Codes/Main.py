import Embedding_Algo as ea
import numpy as np

if __name__ == '__main__': 
    
    #EMBEDDING ALGORITHM

    #Splitting the video frames and then splitting them on RGB frames
    nof = ea.FrameCapture(ea.location + "Akiyo Video.mp4")
    
    #Performing frame subtraction on all the channeas
    #Receiving the sub frames matrices and random frames from R,G and B channea
    sr,sg,sb,rr,rg,rb = ea.Frame_Subtract(nof)

    #Applying two rounds of DWT on the random frame
    a1,b1,c1 = ea.ApplyDWT_Frames(rr,rg,rb)

    #Applying SVD on the random frame
    u1,s1,vt1,u2,s2,vt2,u3,s3,vt3 = ea.ApplySVD(a1,b1,c1)
    s1 = np.pad(s1,(18),mode='constant')
    s2 = np.pad(s2,(18),mode='constant')
    s3 = np.pad(s3,(18),mode='constant')
    vt1 = np.pad(vt1,(18,18),mode='constant')
    vt2 = np.pad(vt2,(18,18),mode='constant')
    vt3 = np.pad(vt3,(18,18),mode='constant')

    #Processing the logo by first doing the RGB split
    r2,g2,b2 = ea.RGB_Splitter_Logo()

    #Applying DWT once on the splitted logo and acquiring the HH sub band
    a2,b2,c2 = ea.ApplyDWT_Logo(r2,g2,b2)
    
    #Applying SVD once on the HH sub DWT-ized logo
    u4,s4,vt4,u5,s5,vt5,u6,s6,vt6, = ea.ApplySVD(a2,b2,c2)
    u4 = np.pad(u4,(9,9),mode='constant')
    u5 = np.pad(u5,(9,9),mode='constant')
    u6 = np.pad(u6,(9,9),mode='constant')
    s4 = np.zeros((124,))
    s5 = np.zeros((124,))
    s6 = np.zeros((124,))
    vt4 = np.zeros((124,))
    vt5 = np.zeros((124,))
    vt6 = np.zeros((124,))
    #Now adding the SVD matrices
    
    #For Red Frame
    u7 = u1 + u4
    s7 = s1 + s4
    vt7 = vt1 + vt4
    vt7 = np.pad(vt7,(0,0),mode='constant')

    #For Green Frame
    u8 = u2 + u5
    s8 = s2 + s5
    vt8 = vt2 + vt5

    #For Blue Frame
    u9 = u3 + u6
    s9 = s3 + s6
    vt9 = vt3 + vt6 

    print("-------------- Sums done! --------------")

    #Reconstructing three SVD matrixes for R,G and B channels separately
    d1 = ea.InverseSVD(u7,s7,vt7)
    d2 = ea.InverseSVD(u8,s8,vt8)
    d3 = ea.InverseSVD(u9,s9,vt9)

    #Applying inverse DWT to the reconstructed frames'''
