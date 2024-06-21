import cv2
from yolov9.models import YOLOv9
from yolov9.utils.general import non_max_suppression, scale_coords

# Function to capture image from Arduino sensor
def capture_image_from_arduino():
   
    image = cv2.imread("trial.jpg")
    return image

# Function to perform object detection using YOLOv9 model
def detect_objects(image):
    # Load YOLOv9 model
    model = YOLOv9(weights='best.pt')

    # Perform object detection
    results = model(image)

    # Apply non-max suppression
    results = non_max_suppression(results)

    return results

# Main function
def main():
    # Capture image from Arduino sensor
    image = capture_image_from_arduino()

    # Perform object detection
    results = detect_objects(image)

    # Display detected objects on image
    for result in results:
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in result:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f'{cls_pred}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with detected objects
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
