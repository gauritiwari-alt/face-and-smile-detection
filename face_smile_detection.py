import cv2

# Load face and smile detectors
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

smile_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # Draw rectangle around face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Extract face region
        gray_face = gray[y:y + h, x:x + w]
        color_face = frame[y:y + h, x:x + w]

        # Detect smiles inside the face
        smiles = smile_detector.detectMultiScale(
            gray_face,
            1.8,
            20
        )

        for (sx, sy, sw, sh) in smiles:

            # Draw rectangle around smile
            cv2.rectangle(
                color_face,
                (sx, sy),
                (sx + sw, sy + sh),
                (255, 0, 255),
                2
            )

    # Display result
    cv2.imshow("Face & Smile Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()
