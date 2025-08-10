import cv2
from ultralytics import YOLO
from sklearn.metrics import classification_report
import numpy as np

def detect_objects_in_photo(image_path, ground_truth=None):
    image_orig = cv2.imread(image_path)
   
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')

    results = yolo_model(image_orig)
    predictions = []
    confidences = []
    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.5:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                color = (0, int(cls[pos]), 255)
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
               
                # Store predictions and confidences
                predictions.append(int(cls[pos]))
                confidences.append(conf[pos])

    result_path = "./imgs/Test/teste.jpg"
    cv2.imwrite(result_path, image_orig)

    if ground_truth is not None:
        if len(predictions) > 0:
            # Use the prediction with the highest confidence
            max_conf_index = np.argmax(confidences)
            prediction = int(predictions[max_conf_index])
            report = classification_report([ground_truth[0]], [prediction], output_dict=True)
            print("Classification Report:")
            print(report)
        else:
            print("No predictions found.")

    return result_path

def detect_objects_in_video(video_path):
    # Load YOLO model
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/.pt')

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result_video_path = r"Results\detected_objects_video.mp4v"
    out = cv2.VideoWriter(result_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Run YOLO detection
        results = yolo_model(frame, verbose=False)  # Disable console output

        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                conf = boxes.conf[i].item()
                xmin, ymin, xmax, ymax = boxes.xyxy[i]

                if conf >= 0.5:
                    label = f"{result.names[cls]} {conf:.2f}"
                    color = (0, cls * 20 % 256, 255)

                    # Draw rectangle and label
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show result
        cv2.imshow('Detected Objects', frame)
        out.write(frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
    
    return result_video_path

def detect_objects_and_plot(path_orig, ground_truth=None):
    image_orig = cv2.imread(path_orig)
    
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    results = yolo_model(image_orig)

    predictions = []
    confidences = []
    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.5:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                color = (0, int(cls[pos]), 255)
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                
                # Store predictions and confidences
                predictions.append(int(cls[pos]))
                confidences.append(conf[pos])

    cv2.imshow("Teste", image_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if ground_truth is not None:
        if len(predictions) > 0:
            # Use the prediction with the highest confidence
            max_conf_index = np.argmax(confidences)
            prediction = int(predictions[max_conf_index])
            report = classification_report([ground_truth[0]], [prediction], output_dict=True)
            print("Classification Report:")
            print(report)
        else:
            print("No predictions found.")

# Example usage
ground_truth = [0]  # Example ground truth for one class
#detect_objects_and_plot("C:/Users/LENOVO/Downloads/Weapons-and-Knives/Weapons-and-Knives-Detector-with-YOLOv8-main/Results/img2.jpg", ground_truth)

detect_objects_in_video("C:/Users/LENOVO/Downloads/Weapons-and-Knives/5243195-hd_1920_1080_25fps.mp4")
