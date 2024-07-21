import cv2
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
file_name = 'coco.names'

try:
    model = cv2.dnn_DetectionModel(frozen_model, config_file)
except Exception as e:
    print(f"Error initializing the model: {e}")
    raise

# Read class labels from coco.names
classLabels = []
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print(classLabels)
print(len(classLabels))

# Continue with the rest of your code...


# Initialize the model
model = cv2.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)  # 255/2 = 127.5
model.setInputMean((127.5, 127.5, 127.5))  # mobilenet => [-1,1]
model.setInputSwapRB(True)

# Image detection example
img = cv2.imread('road_view2.jpg')
if img is None:
    raise FileNotFoundError("Image not found or cannot be read")

classIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[classInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale,
                color=(0, 255, 0), thickness=3)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# Video detection example
cap = cv2.VideoCapture("carv.mp4")
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise FileNotFoundError("Video file cannot be opened")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font,
                            fontScale=font_scale, color=(0, 255, 0), thickness=3)
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
