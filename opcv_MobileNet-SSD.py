import cv2 as cv

classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value

model = cv.dnn.readNetFromTensorflow('models/frozen_inference_graph.pb',\
                                      'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
cap = cv.VideoCapture(0)
cap.set(3, 640) # set video width
cap.set(4, 480) # set video height


if not cap.isOpened():
    print("Cannot open camera")
    exit()
print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# ret = cap.set(cv.CAP_PROP_FRAME_WIDTH,1280)
# ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT,720)

# print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    image = cv.flip(frame, 1)
    image_height, image_width, _ = image.shape
    
    model.setInput(cv.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
    output = model.forward()
    
    for detection in output[0, 0, :, :]:
         confidence = detection[2]
         if confidence > .5:
              class_id = detection[1]
              class_name=id_class_name(class_id,classNames)
              print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
              box_x = detection[3] * image_width
              box_y = detection[4] * image_height
              box_width = detection[5] * image_width
              box_height = detection[6] * image_height
              cv.rectangle(image, (int(box_x), int(box_y)), (int(box_width),\
                         int(box_height)), (23, 230, 210), thickness=1)
              cv.putText(image,str(class_name)+str(confidence) ,(int(box_x), int(box_y+.05*image_height)),\
                         cv.FONT_HERSHEY_SIMPLEX,(.005*image_width),(0, 0, 255))
        
        
    # Display the resulting frame
    cv.imshow('object detection', image)
    # press q to exit
    if cv.waitKey(1) == ord('q'):
        break
   
# When everything done, release the capture
cap.release()   
cv.destroyAllWindows()