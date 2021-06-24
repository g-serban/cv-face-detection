import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_detection_con=0.5):
        self.min_detection_con = min_detection_con
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(self.min_detection_con)
        self.mp_draw = mp.solutions.drawing_utils

    def find_faces(self, img, draw=True):
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(image_rgb)
        bounding_boxes = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):

                bounding_box_class = detection.location_data.relative_bounding_box

                img_height, img_width, img_channel = img.shape
                bounding_box = int(bounding_box_class.xmin * img_width), int(bounding_box_class.ymin * img_height), \
                               int(bounding_box_class.width * img_width), int(bounding_box_class.height * img_height)
                bounding_boxes.append([id, bounding_box, detection.score])

                cv2.rectangle(img, bounding_box, (255, 0, 255), 2)
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bounding_box[0], bounding_box[1] - 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return img, bounding_boxes


def main():
    cap = cv2.VideoCapture("Videos/1.mp4")
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bounding_boxes = detector.find_faces(img)
        print(bounding_boxes)

        cv2.imshow("Video", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()

# TODO 2h:03m