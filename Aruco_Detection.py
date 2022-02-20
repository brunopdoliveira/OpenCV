##########################################################################################
#                                Aruco Detection                                        #
##########################################################################################

# import libraries
import cv2
import cv2.aruco as aruco

video = cv2.VideoCapture(0)
print('Open Camera!')

# Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN

# Define id to find
id_to_find = 10

while (True):

    # Capture frame-by-frame
    ret, frame = video.read()
         # (rows,cols,channels) = frame.shape
         # print(frame.shape) #480x640

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    parameters = aruco.DetectorParameters_create()

    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    # Print 'X' in the center of the camera
        # cv2.putText(frame, "X", (cols/2, rows/2), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Aruco Colors
    borderColor = (0, 255, 0)

    image = aruco.drawDetectedMarkers(frame, corners, ids, borderColor)

    # Find ID Aruco defined and print
    if ids is not None:
        for i in range(len(ids)):
            if ids[0] == id_to_find:
                print(ids[0])
                print("Aruco Encontrado!")

    cv2.imshow('frame', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
