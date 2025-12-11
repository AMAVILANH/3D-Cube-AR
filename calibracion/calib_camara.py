# calib_camara.py
import cv2
import numpy as np
import glob

# ------------------------------------------------------------
# Definir dimensiones del patrón de calibración 
# ------------------------------------------------------------
chessboard_size = (8, 5)  # Esquinas internas del tablero: (filas, columnas)
square_size = 25.0        # Tamaño de cada cuadrado en mm (ajustar si tu tablero es distinto)

# ------------------------------------------------------------
# Preparar los puntos 3D del patrón en coordenadas del mundo real
# ------------------------------------------------------------
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # puntos 3D en el mundo real
imgpoints = []  # puntos 2D en la imagen
valid_image_names = []

# ------------------------------------------------------------
# Cargar imágenes de calibración (ajusta la ruta si es necesario)
# ------------------------------------------------------------
images = glob.glob('../calibracion/calib_images/*.jpg')  # si ejecutas desde ar_cubo/, usa ruta relativa
if len(images) == 0:
    # intenta otra ruta local
    images = glob.glob('calib_images/*.jpg')
if len(images) == 0:
    raise FileNotFoundError("No se encontraron imágenes en 'calibracion/calib_images/'. Añade fotos del tablero.")

gray = None

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Advertencia: no se pudo leer {fname}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Buscar esquinas del tablero (misma llamada que usaba el profe)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp.copy())
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        valid_image_names.append(fname)

        # Dibuja y muestra — opcional (puedes comentar si no quieres ventanas)
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Detección de esquinas', img)
        print(f"Esquinas detectadas en: {fname}  (presiona cualquier tecla para continuar)")
        cv2.waitKey(500)  # 0 originalmente; aquí 500ms automático para agilizar
    else:
        print(f"No se detectaron esquinas en: {fname}")

cv2.destroyAllWindows()

# Verificar que hay suficientes imágenes válidas
if len(objpoints) < 3:
    raise RuntimeError("No hay suficientes imágenes válidas para calibrar. Se necesitan al menos 3 imágenes con esquinas detectadas.")

# ------------------------------------------------------------
# Calibración de cámara (igual que el profe)
# ------------------------------------------------------------
ret, K, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\nCalibración completada correctamente.")
print("\nMatriz intrínseca (K):\n", K)
print("\nCoeficientes de distorsión [k1, k2, p1, p2, k3]:\n", distCoeffs.ravel())

# ------------------------------------------------------------
# Calcular error de reproyección (muy importante para validar)
# ------------------------------------------------------------
mean_error = 0
total_points = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, distCoeffs)
    imgpoints_true = imgpoints[i].reshape(-1,2)
    imgpoints2 = imgpoints2.reshape(-1,2)
    error = cv2.norm(imgpoints_true, imgpoints2, cv2.NORM_L2)
    n = len(imgpoints2)
    mean_error += error
    total_points += n

mean_error = mean_error / total_points
print(f"\nError medio de reproyección (pixeles): {mean_error:.4f}")

# ------------------------------------------------------------
# Guardar calibración
# ------------------------------------------------------------
np.savez("calibracion.npz", K=K, dist=distCoeffs, rvecs=rvecs, tvecs=tvecs,
         chessboard_size=chessboard_size, square_size=square_size, valid_images=valid_image_names)

print("Parámetros guardados en 'calibracion.npz'.")
