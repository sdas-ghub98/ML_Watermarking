import Embedding_Algo as cea
import Logo as logo
if __name__ == '__main__': 
    
    #Calling the embedding algorithm one by one

    #Splitting the video frames
    #nof = cea.FrameCapture(cea.location + "Akiyo Video.mp4")
    
    #Taking a random frame and subtracting it from the remaining frames
    #random_frame = cea.FrameSubtraction(nof)

    #Splitting the random frame into RGB
    r1, g1, b1 = cea.RGB_Splitter(131)
    
    #Applying DWT twice on the splitted random frame
    a1,b1,c1 = cea.ApplyDWT(r1,g1,b1)

    #Applying SVD on the HH band of the random frame
    ur,sr,vhr,ug,sg,vhg,ub,sb,vhb = cea.ApplySVD(r1,g1,b1)
    frames= []
    frames.extend((ur,sr,vhr,ug,sg,vhg,ub,sb,vhb))

    print("Frames processing done")
    
    #Processing the logo by first doing the RGB split
    r2,g2,b2 = logo.RGB_Splitter()

    #Applying DWT once on the splitted logo
    a2,b2,c2 = logo.ApplyDWT_Logo(r2,g2,b2)

    #Applying SVD once on the DWT-ized logo
    url,srl,vhrl,ugl,sgl,vhgl,ubl,sbl,vhbl = logo.ApplySVD_Logo(a2,b2,c2)
    logo_frame = []
    logo_frame.extend((url,srl,vhrl,ugl,sgl,vhgl,ubl,sbl,vhbl))
    
    print("Logo processing successful")


    #Adding the SVD frame and SVD logo together
    print("Putting the logo into the video frame...")
    #Converting the video frame to RGB 
    cea.RGB_Merger(frames)
    logo.RGB_Merger_Logo(logo_frame)
    
    # Once we get two separate RGB image saved, let us now add them together and then add the result to all the subtracted frames.
    wmk_frame = cea.FrameConstruction(300)

    #Constructing the video from these frames
    #cea.CreateVideo(wmk_frame)

    #Extracting algorithm
