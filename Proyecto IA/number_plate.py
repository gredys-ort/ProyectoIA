import cv2
import base64
import pymongo
from pymongo import MongoClient
import datetime
import os

print("OpenCV version:", cv2.__version__)

# Crear el directorio plates si no existe
if not os.path.exists("plates"):
    os.makedirs("plates")

# Ruta al archivo haarcascade
harcascade_path = "model/haarcascade_plate_number.xml"

# Configuración de la cámara
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

min_area = 500
count = 0

# Conexión a la base de datos MongoDB local
uri = "mongodb://localhost:27017"
client = MongoClient(uri)
db = client["plates"]
collection = db["plates"]

while True:
    success, img = cap.read()
    if not success:
        print("No se pudo acceder a la cámara")
        break

    plate_cascade = cv2.CascadeClassifier(harcascade_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    img_roi = None  # Inicializa img_roi

    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Numero de Placa", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y:y+h, x:x+w]
            cv2.imshow("ROI", img_roi)

    cv2.imshow("Extracción de placas con Inteligencia Artificial", img)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        if img_roi is not None:
            # Guarda la imagen en el sistema de archivos
            image_path = f"plates/image_{count}.jpg"
            cv2.imwrite(image_path, img_roi)
            print(f"Imagen guardada en: {image_path}")

            # Codifica la imagen en base64
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Guarda la imagen en la base de datos
            image_data = {
                "image_name": f"image_{count}.jpg",
                "image_data": encoded_image,
                "timestamp": datetime.datetime.now()
            }
            collection.insert_one(image_data)
            print("Imagen guardada en la base de datos")

            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Placa Guardada", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
            cv2.imshow("Resultado", img)
            cv2.waitKey(500)
            count += 1
        else:
            print("No se detectó ninguna placa para guardar.")
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
