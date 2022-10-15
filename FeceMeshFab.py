import math
import cvzone
import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
success, img = cap.read()

imgFront = cv2.imread("iastech.png", cv2.IMREAD_UNCHANGED)
imgFront = cv2.resize(imgFront, (0, 0), None, 0.3, 0.3)
hf, wf, cf = imgFront.shape
hb, wb, cb = img.shape

pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
mp_drawing_styles = mp.solutions.drawing_styles

with mpFaceMesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as faceMesh:

    while True:
        success, img = cap.read()
        imgResult = cvzone.overlayPNG(img, imgFront, [0, 10])
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_IRISES, drawSpec)
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,drawSpec,drawSpec)

                for id, lm in enumerate(faceLms.landmark):
                    #print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    #cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, .5, (0, 255, 0), 1)
                    #print(id, x, y)
                    faces.append([id, x, y])
                #faces.append(face)
                    #if len(faces) == 468:
                        #print(faces[0])
                        #x1, y1 = faces[105][1:]
                        #x2, y2 = faces[23][1:]
                        #cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        #cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 1)
                        #cv2.circle(img, (x1, y1), 4, (0, 0, 255), cv2.FILLED, )
                        #cv2.circle(img, (x2, y2), 4, (0, 0, 255), cv2.FILLED)
                        #cv2.circle(img, (cx, cy),4, (0, 0, 255), cv2.FILLED)
                        #long = math.hypot(x2 - x1, y2 - y1)
                        #print(long)



        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
       # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 155), 1, 1)

        #cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
         #           3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)