import time

import cv2
import mediapipe as mp

cap = cv2.VideoCapture("Videos/1.mp4")
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.65)
mp_draw = mp.solutions.drawing_utils

previous_time = 0

while True:
    success, img = cap.read()

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mp_draw.draw_detection(img, detection)

            bounding_box_class = detection.location_data.relative_bounding_box
            img_height, img_width, img_channel = img.shape
            bounding_box = int(bounding_box_class.xmin * img_width), int(bounding_box_class.ymin * img_height), \
                           int(bounding_box_class.width * img_width), int(bounding_box_class.height * img_height)
            cv2.rectangle(img, bounding_box, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bounding_box[0], bounding_box[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    # current_time = time.time()
    # fps = 1 / (current_time - previous_time)
    # previous_time = current_time

    # cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imshow("Video", img)
    cv2.waitKey(1)

