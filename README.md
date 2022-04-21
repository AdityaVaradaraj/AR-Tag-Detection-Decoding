# AR-Tag-Detection-Decoding
Detect and Decode AR tag in video and superimpose (a) Testudo image on it and (b) a 3D cube wireframe on it.

Input Video: https://drive.google.com/file/d/1MWJOLJcFJvRporEfZ-j_lPbGkVuRv0So/view

First Unzip the zip file into a folder.
There are 6 files in the zip folder:
1) AR_v2.py
2) AR_fft.py 
3) Report
4) This README.md
5) AR_Reference.jpg
6) testudo.png

Be sure to keep the 2 images and the input video given (1tagvideo.mp4) in same folder as the python files. AR_Reference.jpg is the sample AR Tag image provided in the Project 1 instruction manual.

Libraries used:
1) OpenCV (cv2) (Version 4.1.0)
2) scipy
3) numpy
4) matplotlib

For Problem 1(a), run AR_fft.py by typing python3 AR_fft.py in the terminal. You should see the output in a matplotlib window. Press the "q" key to terminate the program.

For Problem 1(b), 2(a) and 2(b), run AR_v2.py by typing python3 AR_fft.py in terminal. The ID and orientation of reference tag should be printed once in the terminal. Then, for each frame, the ID and orientation of tag in the frame should be printed in terminal. A video output of cube part (2(b)) is shown using cv2.imshow() in a named cv2 window. After whole video is finished, i.e., code is finished running, there should be 2 video files written into the same folder as the code. The videos are named as output_2a.mp4 and output_2b.mp4 corresponding to the outputs of problems 2(a) and 2(b). 

The pipeline and methods used are explained in detail in the report and can also be understood by reading the comments in the code.

Would request to view the output videos in 720p quality if viewing directly in google drive.
