# https://google.github.io/mediapipe/solutions/face_detection.html

import os, time, cv2, mediapipe as mp, utils.BGRColor as BGR

os.system('cls')

cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDrawing = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while cap.isOpened():
    _, img              = cap.read()
    imgToRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgToRGB)
    # print(results.detections)
    if results.detections:
        for keypoint, detection in enumerate(results.detections):
            # mpDrawing.draw_detection(img, detection)
            # print(detection)
            # print(keypoint, detection.score, relativeboundingbox.xmin, relativeboundingbox.ymin, 
            #       relativeboundingbox.width,relativeboundingbox.height)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, c = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            x, y, w, h = bbox
            print(bbox)
            # cv2.circle(img, (int(bboxC.xmin * iw), int(bboxC.ymin * ih)), 5, BGR.RED, 2) 
            # cv2.circle(img, (int(bboxC.xmin * iw) + int(bboxC.width * iw), int(bboxC.ymin * ih)), 5, BGR.RED, 2) 
            # cv2.circle(img, (int(bboxC.xmin * iw), int(bboxC.ymin * ih) + int(bboxC.height * ih)), 5, BGR.RED, 2) 
            # cv2.circle(img, (int(bboxC.xmin * iw) + int(bboxC.width * iw), int(bboxC.ymin * ih) + int(bboxC.height * ih)), 5, BGR.RED, 2) 
            cv2.rectangle(img, bbox, BGR.BLUE,2)
            cv2.putText(img, f'Score: {int(detection.score[0]*100)} %', (x,y-15), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, BGR.RED, 1)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,20), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, BGR.RED, 1)
    cv2.imshow('Face Estimation', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
