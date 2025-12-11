import cv2
import numpy as np

# Cargar parámetros de calibración
data = np.load("../calibracion/calibracion.npz")
K = data["K"]
dist = data["dist"]
chessboard_size = tuple(data["chessboard_size"])
square_size = float(data["square_size"])

# Reconstruir objp (igual al código del profe)
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
objp *= square_size

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if found:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # ESTO ES LO MÁS IMPORTANTE:
        retp, rvec, tvec = cv2.solvePnP(objp, corners2, K, dist)

        cv2.drawChessboardCorners(frame, chessboard_size, corners2, found)

        # Mostrar pose en pantalla
        cv2.putText(frame, f"t = {tvec.ravel()}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Pose del Tablero", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
