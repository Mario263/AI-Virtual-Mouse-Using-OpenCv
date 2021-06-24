# AI Virtual Mouse Using OpenCv 
 This is an AI virtual mouse built using libraries and functions like OpenCV Numpy mediapipe etc. I have created a hand tracking module in which i have already described how it will track our hand , by using my module i have defined the location of Index finger and middle finger.

"""project owned by :- Abhishek Sharma

Hand Tracking Module owned by :- Abhishek Sharma
"""

# AI Virtual mouse main module
               
                import cv2
                import numpy as np
                import HandTrackingModule as htm
                import time
                import autopy

             #####################
             wCam, Hcam = 640, 480
             frameR = 20 # frame reduction
             smoothening = 3

             #####################

             plocX, PlocY = 0,0
             clocX, clocY = 0,0

             cap = cv2.VideoCapture(0)
             cap.set(3, wCam)
             cap.set(4, Hcam)
             pTime = 0
             detector = htm.handDetector(maxHands=1)
             wScr, hScr = autopy.screen.size()
             #print(wScr, hScr)
             while True:
                 # 1. to find the hand landmarks
                 success, img = cap.read()
                 img = detector.findHands(img)
                 lmList, bbox = detector.findPosition(img)
                 # 2. Get the tip of the index and middle finger
                 if len(lmList)!=0:
                     x1, y1 = lmList[8][1:]
                     x2, y2 = lmList[12][1:]
                 # 3. check which fingers are up
                     fingers = detector.fingersUp()
                     cv2.rectangle(img, (frameR, frameR), (wCam - frameR, Hcam - frameR), (255, 0, 255), 2)
                     #print(fingers)
                     # 4. only index finger : it is in moving mode
                     if fingers[1]==1 and fingers[2]==0:

                         # 5. Convert coordinates

                         x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
                         y3 = np.interp(y1, (frameR, Hcam-frameR), (0, hScr))
                         # 6. Smoothen Values
                         clocX = plocX + (x3 - plocX) / smoothening
                         clocY = PlocY + (y3 - PlocY) / smoothening
                         # 7. Move Mouse
                         autopy.mouse.move(wScr-clocX, clocY)
                         cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
                         plocX, PlocY = clocX, clocY
                    # 8. Both middle and index fingers are up : Clicking mode
                    if fingers[1] == 1 and fingers[2] == 1:
                        length, img, lineInfo = detector.findDistance(8, 12, img)
                        print(length)
                        # 9. Find Distance between fingers
                        if length < 45:
                            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                            # 10. click mouse if distance short
                            autopy.mouse.click()



                 # 11. frame rate
                 cTime = time.time()
                 fps = 1/(cTime-pTime)
                 pTime = cTime
                 cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                 # 12. Display
                 cv2.imshow("Image", img)
                 cv2.waitKey(1)


