import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

IMG_SIZE = 200
MODEL_PATH = "asl_best_model.h5"

CLASSES = [
    'A','B','C','D','E','F','G',
    'H','I','J','K','L','M','N',
    'O','P','Q','R','S','T','U',
    'V','W','X','Y','Z','del','nothing','space'
]

model = tf.keras.models.load_model(MODEL_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)

def preprocess(img):
    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Square pad (ASL dataset images are perfectly square)
    h, w, _ = img.shape
    size = max(h, w)
    square = np.full((size, size, 3), 255, dtype=np.uint8)  # white pad

    offset_h = (size - h) // 2
    offset_w = (size - w) // 2
    square[offset_h:offset_h + h, offset_w:offset_w + w] = img

    # Resize to model size
    img = cv2.resize(square, (IMG_SIZE, IMG_SIZE))

    # Slight blur helps match ASL dataset smoothness
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Normalize
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

cam = cv2.VideoCapture(0)
print("Live inference started. Press q to exit.")

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction = "none"

    if result.multi_hand_landmarks:
        h, w, _ = frame.shape
        lm = result.multi_hand_landmarks[0]

        x = [p.x for p in lm.landmark]
        y = [p.y for p in lm.landmark]

        xmin = int(min(x)*w) - 40
        xmax = int(max(x)*w) + 40
        ymin = int(min(y)*h) - 40
        ymax = int(max(y)*h) + 40

        xmin, ymin = max(0,xmin), max(0,ymin)
        xmax, ymax = min(w,xmax), min(h,ymax)

        roi = frame[ymin:ymax, xmin:xmax]

        if roi.size > 0:
            img = preprocess(roi)
            pred = model.predict(img, verbose=0)[0]
            prediction = CLASSES[np.argmax(pred)]

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

    cv2.putText(frame, f"{prediction}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    cv2.imshow("ASL Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
