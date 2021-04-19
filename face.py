import cv2

face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    read_success, frame = cam.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = face_data.detectMultiScale(gray_frame)
    eye_coordinates = eye_data.detectMultiScale(gray_frame)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(gray_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for (x2, y2, w2, h2) in eye_coordinates:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)

    # print("detected at -- {}".format(face_coordinates))
    print("detected at -- {}".format(eye_coordinates))

    cv2.imshow('Face Detector', gray_frame)
    cv2.waitKey(1)

    # PycharmProjects\FaceTracker
