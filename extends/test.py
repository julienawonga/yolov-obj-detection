import cv2

# Charger le modèle de détection pré-entraîné (remplacez le chemin par le vôtre)
model = cv2.dnn.readNet("yolov4-tiny_best.weights", "yolov4-tiny-custom.cfg")

# Charger les noms des classes (remplacez le chemin par le vôtre)
classes = ['pot']


# Configurer la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prétraitement de l'image pour l'entrée du modèle
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    model.setInput(blob)

    # Obtenir les détections
    outs = model.forward(model.getUnconnectedOutLayersNames())

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]

            if confidence > 0.5:  # Seuil de confiance
                # Coordonnées du rectangle de détection
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                # Dessiner le rectangle de détection
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(frame, classes[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Afficher le cadre avec les détections
    cv2.imshow("Object Detection", frame)

    # Appuyez sur la touche 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
