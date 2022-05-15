import math
import cv2
import mediapipe as mp
import time
from Frames import CapFrame
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import snap7

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
mp_drawing_styles = mp.solutions.drawing_styles
detect = FaceMeshDetector(maxFaces=1)

# -------------------------------------------------------
# Configurações para conexção com PLC Siemens (S7-1500)
IP = '10.87.111.43'
RACK = 0
SLOT = 1

DB_NUMBER = 2
START_ADDRESS = 0
SIZE = 1

plc = snap7.client.Client()
plc.connect(IP, RACK, SLOT)
# -------------------------------------------------------

IrisVal = 0
FaceMove = False

with mpFaceMesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as faceMesh:

    while True:
        success, img = cap.read()
        img, faces = detect.findFaceMesh(img, draw=False)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)
        frame = []

# ----------------------------------------------------------------------------------------------------------------------
# ROTINA PARA DEFINIR A POSICAO CORRETA DO ROSTO EM RELACAO A CAMERA
        if faces:
            face = faces[0]
            ptLeft = face[145]
            ptRight = face[374]
            #cv2.line(img, ptLeft, ptRight, (0,200,0), 3)
            #cv2.circle(img, ptLeft, 5, (255,0,255), cv2.FILLED)
            #cv2.circle(img, ptRight, 5, (255,0,255), cv2.FILLED)
            w,_ = detect.findDistance(ptLeft, ptRight)
            W = 6.3     # Este valor e a distancia padrao do centro dos olhos em centimetros.

            ''' CALCULADO A DISTANCIA DO FOCO DA LENTE '''
            f = 530
            d = (W*f) / w

            ''' GERANDO IDENTIFICACAO PARA MOSTRAR SE A FACE ESTA DENTRO OU FORA DA AREA DE MEDICAO '''
            if d >= 25 and d <= 30:     # Distancia do rosto ate a camera em centimetros
                cvzone.putTextRect(img, 'OK', (face[10][0] - 25, face[10][1] - 50), scale=2, colorR=(0, 255, 0), colorT=0)
                FaceMove = True
            else:
                cvzone.putTextRect(img, 'NOK', (face[10][0] - 25, face[10][1] - 50), scale=2, colorR=(0, 0, 255))
                FaceMove = False
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# ROTINA PARA CAPTURAR OS MOVIMENTOS DO ROSTO
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_IRISES, drawSpec)
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)

                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    #cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, .5, (0, 255, 0), 1)
                    frame.append([id, x, y])

                    if len(frame) == 478:
                    # ---------------------------------------------------
                    # CONTROLE PELA SOBRANCELHA (LIGA INVERSOR)
                        sobrancelha = CapFrame(frame, 105, 23, img)
                        if FaceMove == True:
                            print(sobrancelha.cap_mov())
                            if sobrancelha.cap_mov() >= 70:
                                reading = plc.db_read(DB_NUMBER, START_ADDRESS, 1)
                                snap7.util.set_bool(reading, 0, 0, 1)
                                plc.db_write(DB_NUMBER, START_ADDRESS, reading)
                                print('Sobrancelha Levantada')
                            else:
                                reading = plc.db_read(DB_NUMBER, START_ADDRESS, 1)
                                snap7.util.set_bool(reading, 0, 0, 0)
                                plc.db_write(DB_NUMBER, START_ADDRESS, reading)
                                print('Sobrancelha Normal')
                    # ---------------------------------------------------
                    
                    # ---------------------------------------------------
                    # CONTROLE PELA BOCA (DESLIGA INVERSOR)
                        boca = CapFrame(frame, 17, 0, img)
                        if FaceMove == True:
                            print(boca.cap_mov())
                            if boca.cap_mov() >= 62:
                                reading = plc.db_read(DB_NUMBER, START_ADDRESS, 1)
                                snap7.util.set_bool(reading, 0, 1, 1)
                                plc.db_write(DB_NUMBER, START_ADDRESS, reading)
                                print('Boca Aberta')
                            else:
                                reading = plc.db_read(DB_NUMBER, START_ADDRESS, 1)
                                snap7.util.set_bool(reading, 0, 1, 0)
                                plc.db_write(DB_NUMBER, START_ADDRESS, reading)
                                print('Boca Fechada')
                    # ---------------------------------------------------

                    # ---------------------------------------------------   
                    # CONTROLE PELA IRIS (ESQUERDA DIMINUI E DIRETA AUMENTA)
                        iris = CapFrame(frame, 468, 168, img)
                        if FaceMove == True:
                            print(iris.cap_mov())
                            if iris.cap_mov() <= 50:
                                reading = plc.db_read(DB_NUMBER, START_ADDRESS, 1)
                                snap7.util.set_bool(reading, 0, 2, 1)
                                plc.db_write(DB_NUMBER, START_ADDRESS, reading)
                                print('Olho virado para Esquerda')
                            elif iris.cap_mov() >= 70:
                                reading = plc.db_read(DB_NUMBER, START_ADDRESS, 1)
                                snap7.util.set_bool(reading, 0, 3, 1)
                                plc.db_write(DB_NUMBER, START_ADDRESS, reading)
                                print('Olho virado para Direita')
                            else:
                                reading = plc.db_read(DB_NUMBER, START_ADDRESS, 1)
                                snap7.util.set_bool(reading, 0, 2, 0)
                                plc.db_write(DB_NUMBER, START_ADDRESS, reading)
                                
                                reading2 = plc.db_read(DB_NUMBER, START_ADDRESS, 1)
                                snap7.util.set_bool(reading2, 0, 3, 0)
                                plc.db_write(DB_NUMBER, START_ADDRESS, reading2)
                                print('Olho no Centro, nada Acontece')
                    # ---------------------------------------------------
    
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.imshow("Image", img)
        cv2.waitKey(1)
cap.release()
# ----------------------------------------------------------------------------------------------------------------------
