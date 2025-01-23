import cv2
import sys
from transformers import pipeline
from PIL import Image

# Parameters
face_confidence_threshold = 0.5
emotion_confidence_threshold = 0.5
scaleFactor = 1.8 # for screen size

# default camera, cmd arguments for other camera
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = "Emotion Detector"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Facial detection model (from OpenCV)
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
# Model parameters
in_width = 300
in_height = 300
mean = [104, 117, 123]
conf_threshold = 0.7

# Emotion recognition model (HuggingFace transformers)
emotions = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

while cv2.waitKey(1) != ord('q'):
    has_frame, frame = source.read()
    if not has_frame:
        break

    # Flip video and resize
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, dsize=None, fx=scaleFactor, fy=scaleFactor, interpolation=cv2.INTER_LINEAR)

    # Locate face in frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    best_guess_face = detections[0, 0, 0]
    face_confidence = best_guess_face[2]

    # Recognize emotion
    all_emotions = emotions(Image.fromarray(frame))
    best_emotion = all_emotions[0]
    emotion_confidence = best_emotion['score']

    # Annotate frame
    if face_confidence > face_confidence_threshold and emotion_confidence > emotion_confidence_threshold:
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        x_left_bottom = int(best_guess_face[3] * frame_width)
        y_left_bottom = int(best_guess_face[4] * frame_height)
        x_right_top = int(best_guess_face[5] * frame_width)
        y_right_top = int(best_guess_face[6] * frame_height)

        # Face rectangle
        cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
        # label = "Confidence: %.4f" % confidence
        label = "Emotion: " + str(best_emotion['label'])
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Label and background
        cv2.rectangle(
            frame,
            (x_left_bottom, y_left_bottom - label_size[1]),
            (x_left_bottom + label_size[0], y_left_bottom + base_line),
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.putText(frame, label, (x_left_bottom, y_left_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # show frame
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)