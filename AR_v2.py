#!/usr/bin/env python

import numpy as np
import cv2

# ------------ Finding Projection Matrix --------------
def find_projective(H):
    # Intrinsic Matrix
    K = np.array([[1346.100595, 0, 932.1633975],
                  [0, 1355.933136, 654.8986796],
                  [0,           0,           1]])
    
    # Finding lambda
    lam = (np.linalg.norm(np.matmul(np.linalg.inv(K), H[:, 0])) + np.linalg.norm(np.matmul(np.linalg.inv(K), H[:, 1])))/2
    
    # Finding [r1 r2 t]
    Pt = (1.0/lam)*np.matmul(np.linalg.inv(K), H)
    det = np.linalg.det(Pt)
    if det <= 0:
        Pt = -1*Pt
    
    r1 = Pt[:,0]
    r2 = Pt[:,1]
    r3 = np.cross(Pt[:,0], Pt[:, 1]) # r3 = (r1 x r2)
    t = Pt[:,2]
    Rt = np.column_stack((r1, r2, r3, t))
    # Projection Matrix, P = K [r1 r2 (r1 x r2) t] 
    P = np.matmul(K, Rt)
    return P

# ---------------- Finding Homography Matrix -------------
def findHomography(X,Xp):
    
    A = np.asarray([[-X[0][0], -X[0][1], -1, 0, 0, 0, X[0][0]*Xp[0][0], X[0][1]*Xp[0][0], Xp[0][0]],
              [0, 0, 0, -X[0][0], -X[0][1], -1, X[0][0]*Xp[0][1], X[0][1]*Xp[0][1], Xp[0][1]],
	          [-X[1][0], -X[1][1], -1, 0, 0, 0, X[1][0]*Xp[1][0], X[1][1]*Xp[1][0], Xp[1][0]],
              [0, 0, 0, -X[1][0], -X[1][1], -1, X[1][0]*Xp[1][1], X[1][1]*Xp[1][1], Xp[1][1]],
	          [-X[2][0], -X[2][1], -1, 0, 0, 0, X[2][0]*Xp[2][0], X[2][1]*Xp[2][0], Xp[2][0]],
              [0, 0, 0, -X[2][0], -X[2][1], -1, X[2][0]*Xp[2][1], X[2][1]*Xp[2][1], Xp[2][1]],
	          [-X[3][0], -X[3][1], -1, 0, 0, 0, X[3][0]*Xp[3][0], X[3][1]*Xp[3][0], Xp[3][0]],
              [0, 0, 0, -X[3][0], -X[3][1], -1, X[3][0]*Xp[3][1], X[3][1]*Xp[3][1], Xp[3][1]]])
    # Use SVD to detrmine eigenvectors of A^T A
    U ,S, Vt = np.linalg.svd(A.astype(np.float32))

    # Take last eigenvector of A^T A scaled such that last element is 1
    # Solve Ax = 0 system
    x = Vt[8, :]/Vt[8,8] 

    # Reshape to 3x3 homography matrix
    H = x.reshape(3,3)
    return H

# ---------------- Warping AR Tag to make ready for decoding ------------
def warp_AR(H, img, maxWidth, maxHeight):
    H_inv = np.linalg.pinv(H)
    warped = np.zeros((maxWidth, maxHeight, 3), np.uint8)
    h, w = maxHeight, maxWidth
    indY,indX = np.indices((h,w))
    lin_homg_pts = np.stack((indX.ravel(), indY.ravel(), np.ones(indX.size)))
    # Use Inverse Warping
    source_img = H_inv.dot(lin_homg_pts)
    source_img /= source_img[2, :]
    X, Y = source_img[:2, :].astype(int)

    # Handling erraneous indices that go above or below image width and height
    for i in range(len(Y)):
        if Y[i] >= 1080:
            Y[i] = 1080 - 1
        if Y[i] < 0:
            Y[i] = 0
        if X[i] < 0:
            X[i] = 0
        if X[i] >=1920:
            X[i] = 1920 - 1
    # Copying pixels of AR Tag from frame of video to warped image of 128 x 128
    warped[indY.ravel(), indX.ravel(), :] = img[Y, X, :]    
    tag = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) # Converting to Gray image
    ret,tag = cv2.threshold(np.uint8(tag), 200, 255, cv2.THRESH_BINARY) # Converting to binary image
    return tag

# --------------- Warping and Superimposing -----------------
def warp_testudo(H, frame_img, testudo_img, maxWidth, maxHeight):
    H_inv = np.linalg.pinv(H)
    h, w = maxHeight, maxWidth
    indY,indX = np.indices((h,w))
    lin_homg_pts = np.stack((indX.ravel(), indY.ravel(), np.ones(indX.size)))
    # Using Inverse Warping
    source_img = H_inv.dot(lin_homg_pts)
    source_img /= source_img[2, :]
    X, Y = source_img[:2, :].astype(int)

    # Handling erraneous indices that go above or below image width and height 
    for i in range(len(Y)):
        if Y[i] >= 1080:
            Y[i] = 1080 - 1
        if Y[i] < 0:
            Y[i] = 0
        if X[i] < 0:
            X[i] = 0
        if X[i] >=1920:
            X[i] = 1920 - 1
    
    # Copy Pixels from Testudo image to video frame's AR Tag area
    frame_img[Y, X, :] = testudo_img[indY.ravel(), indX.ravel(), :]    
    return frame_img

# ------------------ Decoding the AR Tag ---------------
def decode_tag(tag):
    x = np.arange(0, 128, 16)
    y = np.arange(0, 128, 16)

    orientation = -1
    # Decode orientation based on location of white block in the space between 2x2 and 4x4 grid
    # Detect white block by checking whether mean >= 256/2 since 255 is highest value for all white pixels 
    if np.mean(np.reshape(tag[y[2]:y[2]+16, x[2]:x[2]+16],(256, 1))) >=128:
        orientation = np.pi
    elif np.mean(np.reshape(tag[y[2]:y[2]+16, x[5]:x[5]+16],(256, 1))) >=128:
        orientation = np.pi/2
    elif np.mean(np.reshape(tag[y[5]:y[5]+16, x[2]:x[2]+16],(256, 1))) >=128:
        orientation = 3*np.pi/2
    elif np.mean(np.reshape(tag[y[5]:y[5]+16, x[5]:x[5]+16],(256, 1))) >=128:
        orientation = 0
    else:
        print('Orientation Not found')
    
    # Choose inner 2x2 Grid
    x_bin = x[3:5]
    y_bin = y[3:5]

    # Based on Orientation, decode ID using binary representation (powers of 2)
    if orientation == 0:
        ID = 0

        if np.mean(np.reshape(tag[y_bin[0]:y_bin[0]+16, x_bin[0]:x_bin[0]+16],(256, 1))) >=128:
            ID += 1
        if np.mean(np.reshape(tag[y_bin[0]:y_bin[0]+16, x_bin[1]:x_bin[1]+16],(256, 1))) >=128:
            ID += 2
        if np.mean(np.reshape(tag[y_bin[1]:y_bin[1]+16, x_bin[1]:x_bin[1]+16],(256, 1))) >=128:
            ID += 4
        if np.mean(np.reshape(tag[y_bin[1]:y_bin[1]+16, x_bin[0]:x_bin[0]+16],(256, 1))) >=128:
            ID += 8
    elif orientation == np.pi/2:
        ID = 0

        if np.mean(np.reshape(tag[y_bin[0]:y_bin[0]+16, x_bin[0]:x_bin[0]+16],(256, 1))) >=128:
            ID += 2
        if np.mean(np.reshape(tag[y_bin[0]:y_bin[0]+16, x_bin[1]:x_bin[1]+16],(256, 1))) >=128:
            ID += 4
        if np.mean(np.reshape(tag[y_bin[1]:y_bin[1]+16, x_bin[1]:x_bin[1]+16],(256, 1))) >=128:
            ID += 8
        if np.mean(np.reshape(tag[y_bin[1]:y_bin[1]+16, x_bin[0]:x_bin[0]+16],(256, 1))) >=128:
            ID += 1
    elif orientation == np.pi:
        ID = 0

        if np.mean(np.reshape(tag[y_bin[0]:y_bin[0]+16, x_bin[0]:x_bin[0]+16],(256, 1))) >= 128:
            ID += 4
        if np.mean(np.reshape(tag[y_bin[0]:y_bin[0]+16, x_bin[1]:x_bin[1]+16],(256, 1))) >=128:
            ID += 8
        if np.mean(np.reshape(tag[y_bin[1]:y_bin[1]+16, x_bin[1]:x_bin[1]+16],(256, 1))) >=128:
            ID += 1
        if np.mean(np.reshape(tag[y_bin[1]:y_bin[1]+16, x_bin[0]:x_bin[0]+16],(256, 1))) >=128:
            ID += 2
    elif orientation == 3*np.pi/2:
        ID = 0

        if np.mean(np.reshape(tag[y_bin[0]:y_bin[0]+16, x_bin[0]:x_bin[0]+16],(256, 1))) >=128:
            ID += 8
        if np.mean(np.reshape(tag[y_bin[0]:y_bin[0]+16, x_bin[1]:x_bin[1]+16],(256, 1))) >=128:
            ID += 1
        if np.mean(np.reshape(tag[y_bin[1]:y_bin[1]+16, x_bin[1]:x_bin[1]+16],(256, 1))) >=128:
            ID += 2
        if np.mean(np.reshape(tag[y_bin[1]:y_bin[1]+16, x_bin[0]:x_bin[0]+16],(256, 1))) >=128:
            ID += 4
    else:
        ID = 0
        print("Orientation not identifiable and hence no ID found")
    return ID, orientation

if __name__ == '__main__':

    # Create a VideoCapture object
    cap = cv2.VideoCapture('1tagvideo.mp4') # input video
    

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frameSize = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    # VideoWriter Objects are created to record videos
    out_2a = cv2.VideoWriter('output_2a.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), frameSize)
    out_2b = cv2.VideoWriter('output_2b.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), frameSize)
    
    ref_img = cv2.imread("AR_Reference.jpg", cv2.IMREAD_GRAYSCALE)
    ref = cv2.resize(ref_img, (128, 128), interpolation = cv2.INTER_AREA)
    
    # ----------- 1b: Decoding Tag in Reference image --------------------
    ID_1b, orientation_1b = decode_tag(ref)
    print('------------ 1b: Decode Reference Tag Image ----------')
    print("ID: ", ID_1b)
    print("Orientation: ", orientation_1b)
    print('------------ 2a: Decode Frame Tag and superimpose testudo ----------')
    
    testudo = cv2.imread("testudo.png")
    t_h, t_w, channels = testudo.shape
    while(True):
        ret, frame = cap.read()
        if(ret == True):
            #------------------- 2a: Decoding and Testudo Image Imposing ------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale     
            bi = cv2.bilateralFilter(gray, 5, 75, 75) # Bilateral Filtering to smooth/blur image
            dst = cv2.cornerHarris(bi, 6, 7, 0.04) # Harris Corner Detection 
            dst = cv2.dilate(dst, None) # Dilation
            ret, dst = cv2.threshold(dst, 0.1*dst.max(), 255, 0)
            dst = np.uint8(dst)

            # -----------------  Getting corner coordinates from output of cornerHarris() ---------
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

            criteria =  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners = cv2.cornerSubPix(bi, np.float32(centroids), (5,5), (-1, -1), criteria)
            
            corners = corners[1:,:]
            
            # ---------------- Getting rid of detected Paper Corners ------------------
            corners_AR = None
            j = 0
            for i in corners:
                
                if int(i[0]) > int(1.075*min(corners[:,0])) and int(i[0]) < int(0.925*max(corners[:,0])) and int(i[1]) > int(1.075*min(corners[:,1])) and int(i[1]) < int(0.925*max(corners[:,1])):
                    cv2.circle(frame, (int(i[0]), int(i[1])), 3, (0, 255, 0), -1)
                    if j==0:
                        corners_AR = np.array([(i[0], i[1])])
                    else:
                        corners_AR = np.append(corners_AR, [(i[0], i[1])], axis=0)
                    j += 1

            # ------------------ Seperating 4 exterior corners of the tag from the interior points detected ---------
            exterior = None
            interior = None
            j= 0
            k = 0
            for i in corners_AR:
                if i[0] > min(corners_AR[:,0]) and i[0] < max(corners_AR[:,0]) and i[1] > min(corners_AR[:,1]) and i[1] < max(corners_AR[:,1]):
                    if j==0:
                        interior = np.array([(i[0], i[1])])
                    else:
                        interior = np.append(interior, [(i[0], i[1])], axis=0)
                    j += 1
                else:
                    if k==0:
                        exterior = np.array([(i[0], i[1])])
                    else:
                        exterior = np.append(exterior, [(i[0], i[1])], axis=0)
                    k += 1
                    cv2.circle(frame, (int(i[0]), int(i[1])), 3, (0, 0, 255), -1)
            
            if len(exterior) != 4:
                # Dealing with frames where corner detection fails to detect 4 points
                rect = np.copy(prev_exterior)
            else:
                # Sorting exterior corners of tag in Clockwise Order
                exterior = exterior[0:4, :]
                rect = np.zeros((4,2))
                s = exterior.sum(axis = 1)
                rect[0] = exterior[np.argmin(s)]  
                rect[2] = exterior[np.argmax(s)]
                diff = np.diff(exterior, axis = 1)
                rect[1] = exterior[np.argmin(diff)]
                rect[3] = exterior[np.argmax(diff)]
            
            ref_pts = np.array([[0,0], [127, 0], [127, 127], [0,127]]) # Corners of resized reference AR tag image
            
            # finding homography matrix between tag in video and in reference tag image
            H = findHomography(rect, ref_pts) 
        
            warped = warp_AR(H, frame, 128, 128) # Warping the AR tag in video frame
             

            ID, orientation = decode_tag(warped) # Decoding ID and orientation of AR tag
            
            # Set order of corners of testudo image and cube given for homography
            # according to orientation decoded
            if orientation == 0:
                testudo_pts = np.array([[0,0], [t_w, 0], [t_w, t_h],[0, t_h]])
                cube_pts = np.array([[0,0],[127, 0],[127, 127],[0, 127]])
            elif orientation == np.pi/2:
                testudo_pts = np.array([[t_w, 0], [t_w, t_h],[0, t_h], [0,0]])
                cube_pts = np.array([[127, 0],[127, 127],[0, 127], [0,0]])
            elif orientation == np.pi:
                testudo_pts = np.array([[t_w, t_h],[0, t_h], [0,0], [t_w, 0]])
                cube_pts = np.array([[127, 127],[0, 127], [0,0],[127, 0]])
            elif orientation == 3*np.pi/2:
                testudo_pts = np.array([[0, t_h], [0,0], [t_w, 0], [t_w, t_h]])
                cube_pts = np.array([[0, 127], [0,0],[127, 0],[127, 127]])

            if orientation != -1:
                # When Orientation is properly detected, print ID and orientation
                print("Orientaton: ", orientation)
                print("ID: ",ID)

            # finding homography matrix between tag in video and corners of Testudo image    
            H_T = findHomography(rect, testudo_pts)
            frame_testudo = np.copy(frame)
            # Warp Testudo image to superimpose on top of tag in video
            frame_testudo = warp_testudo(H_T, frame_testudo, testudo, t_w, t_h)
            
            out_2a.write(frame_testudo)
            
            #----------- 2b : Cube projection ------------
            
            # finding homography matrix between corners of virtual cube and corners of tag in video
            H_C = findHomography(cube_pts, rect)

            # Find Projection matrix
            P = find_projective(H_C)

            # Project Cube corners onto video frame
            x1, y1, z1 = np.matmul(P, np.array([0,0,0,1]))
            x2, y2, z2 = np.matmul(P, np.array([127,0,0,1]))
            x3, y3, z3 = np.matmul(P, np.array([127,127,0,1]))
            x4, y4, z4 = np.matmul(P, np.array([0,127,0,1]))
            x5, y5, z5 = np.matmul(P, np.array([0,127,-127,1]))
            x6, y6, z6 = np.matmul(P, np.array([0,0,-127,1]))
            x7, y7, z7 = np.matmul(P, np.array([127,0,-127,1]))
            x8, y8, z8 = np.matmul(P, np.array([127,127,-127,1]))

            # Draw Cube edges
            cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x2/z2),int(y2/z2)), (255,0,255), 2)
            cv2.line(frame,(int(x2/z2),int(y2/z2)),(int(x3/z3),int(y3/z3)), (255,0,255), 2)
            cv2.line(frame,(int(x3/z3),int(y3/z3)),(int(x4/z4),int(y4/z4)), (255,0,255), 2)
            cv2.line(frame,(int(x4/z4),int(y4/z4)),(int(x1/z1),int(y1/z1)), (255,0,255), 2)
            cv2.line(frame,(int(x4/z4),int(y4/z4)),(int(x5/z5),int(y5/z5)), (255,0,255), 2)
            cv2.line(frame,(int(x5/z5),int(y5/z5)),(int(x6/z6),int(y6/z6)), (255,0,255), 2)
            cv2.line(frame,(int(x6/z6),int(y6/z6)),(int(x1/z1),int(y1/z1)), (255,0,255), 2)
            cv2.line(frame,(int(x6/z6),int(y6/z6)),(int(x7/z7),int(y7/z7)), (255,0,255), 2)
            cv2.line(frame,(int(x7/z7),int(y7/z7)),(int(x2/z2),int(y2/z2)), (255,0,255), 2)
            cv2.line(frame,(int(x7/z7),int(y7/z7)),(int(x8/z8),int(y8/z8)), (255,0,255), 2)
            cv2.line(frame,(int(x8/z8),int(y8/z8)),(int(x3/z3),int(y3/z3)), (255,0,255), 2)
            cv2.line(frame,(int(x8/z8),int(y8/z8)),(int(x5/z5),int(y5/z5)), (255,0,255), 2)
            
            out_2b.write(frame)

            prev_exterior = np.copy(rect) # Store previous frame tag corner points
            cv2.imshow('cube_frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break 

    # When everything done, release the video capture and video write objects
    out_2a.release()
    out_2b.release()
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
