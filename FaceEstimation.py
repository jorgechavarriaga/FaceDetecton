# https://google.github.io/mediapipe/solutions/face_detection.html

import os, time, cv2, mediapipe as mp, utils.BGRColor as BGR

class FaceDetector():
    def __init__(self, modelSelection = 0, minDetectionConfidence = 0.5):
        self.modelSelection = modelSelection
        self.minDetectionConfidence = minDetectionConfidence
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDrawing = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionConfidence)        

    def findFaces(self, img, drawing = True):
        imgToRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgToRGB)
        bboxs = []
        if self.results.detections:
            for keypoint, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, c = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                x, y, w, h = bbox
                bboxs.append([keypoint, bbox, detection.score])
                if drawing:
                    self.fancyDraw(img, bbox)
                    cv2.putText(img, f'Score: {int(detection.score[0]*100)} %', (x,y-15), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, BGR.RED, 1)
        return img, bboxs        
                
        
    def fancyDraw(self, img, bbox, l = 25, t = 5):
        x0, y0, w, h = bbox
        x1, y1 = x0 + w, y0 +h 
        cv2.rectangle(img, bbox, BGR.BLUE,1)
        cv2.line(img, (x0,y0), (x0+l, y0), BGR.BLUE, t)
        cv2.line(img, (x0,y0), (x0, y0+l), BGR.BLUE, t)
        cv2.line(img, (x0+w,y0), (x0+w-l, y0), BGR.BLUE, t)
        cv2.line(img, (x0+w,y0), (x0+w, y0+l), BGR.BLUE, t)
        cv2.line(img, (x0,y0+h), (x0, y0+h-l), BGR.BLUE, t)
        cv2.line(img, (x0,y0+h), (x0+l, y0+h), BGR.BLUE, t)
        cv2.line(img, (x0+w,y0+h), (x0+w, y0+h-l), BGR.BLUE, t)
        cv2.line(img, (x0+w,y0+h), (x0+w-l, y0+h), BGR.BLUE, t)
        return img
        
def main():
    os.system('cls')            
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()
    while cap.isOpened():
        _, img = cap.read()
        img, bboxs = detector.findFaces(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20,20), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, BGR.RED, 1)
        cv2.imshow('Face Estimation', img)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

main()
        






