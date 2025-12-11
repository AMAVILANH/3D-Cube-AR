import cv2
import numpy as np

# Cargar parámetros de calibración
data = np.load("../calibracion/calibracion.npz")
K = data["K"]
dist = data["dist"]
chessboard_size = tuple(data["chessboard_size"])
square_size = float(data["square_size"])

# Crear objp igual que en calibración
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
objp *= square_size

# DEFINIR CUBO (75 mm)
cube_size = 75.0
cube = np.float32([
    [0,0,0],
    [cube_size,0,0],
    [cube_size,cube_size,0],
    [0,cube_size,0],
    [0,0,-cube_size],
    [cube_size,0,-cube_size],
    [cube_size,cube_size,-cube_size],
    [0,cube_size,-cube_size]
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if found:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        retp, rvec, tvec = cv2.solvePnP(objp, corners2, K, dist)

        # PROYECTAR CUBO
        imgpts, _ = cv2.projectPoints(cube, rvec, tvec, K, dist)
        imgpts = np.int32(imgpts).reshape(-1,2)

        # DIBUJAR BASE
        frame = cv2.drawContours(frame, [imgpts[:4]], -1, (0,255,0), 3)

        # DIBUJAR COLUMNAS
        for i,j in zip(range(4), range(4,8)):
            frame = cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[j]), (255,0,0), 3)

        # DIBUJAR TAPA
        frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0,0,255), 3)

    cv2.imshow("AR - Cubo", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
