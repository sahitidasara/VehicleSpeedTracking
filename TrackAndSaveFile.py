
import cv2
from ultralytics import YOLO
from Speed import SpeedEstimator
from collections import defaultdict


# Load YOLOv8 model
model = YOLO("yolov10n.pt")

# Initialize global variable to store cursor coordinates
line_pts = [(0, 298), (1019, 298)] # line placement
names = model.model.names  # Dictionary of class names

# Initialize SpeedEstimator with the line points and model names
speed_obj = SpeedEstimator(reg_pts=line_pts, names=names)

# Mouse callback function to capture mouse movement
def RGB(event, x, y, flags, param):
    global cursor_point
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_point = (x, y)
        print(f"Mouse coordinates: {cursor_point}")

# Set up the window and attach the mouse callback function
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open the video file or webcam feed
cap = cv2.VideoCapture('Testing_Video.mp4')

count = 0
# Dictionary to store logged object speeds (to avoid duplicate logging)
logged_speeds = {}
# Dictionary to store cumulative speed data for averaging
cumulative_speed_data = defaultdict(lambda: {"total_speed": 0, "count": 0, "class_name": None})

with open("speed_log.txt", "w") as log_file:
    log_file.write("ObjectID, ObjectClass, Average Speed (m/h)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or cannot be read.")
            break

        count += 1
        if count % 3 != 0:  # Skip some frames for speed (optional)
            continue

        frame = cv2.resize(frame, (1020, 500))

        # Perform object tracking
        tracks = model.track(frame, persist=True, classes=[2, 7])

        # Estimate speed
        im0 = speed_obj.estimate_speed(frame, tracks)

        # Retrieve speed data
        speed_data = speed_obj.get_speed_data()

        # Accumulate speed data for each object
        for data in speed_data:
            obj_id = data["id"]
            speed_mph = data["speed"]
            obj_class_id = data["class_id"]  # Assuming class_id is part of `data`
            obj_class_name = names.get(obj_class_id, "Unknown")  # Lookup class name
            cumulative_speed_data[obj_id]["total_speed"] += speed_mph
            cumulative_speed_data[obj_id]["count"] += 1
            cumulative_speed_data[obj_id]["class_name"] = obj_class_name


        # Display the frame
        cv2.imshow("RGB", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Log the average speed for each tracked object
    for obj_id, speed_info in cumulative_speed_data.items():
        average_speed = speed_info["total_speed"] / speed_info["count"]
        class_name = speed_info["class_name"]
        print(f"Object ID {obj_id}, Class: {class_name}, Average Speed: {average_speed:.2f} m/h")
        log_file.write(f"{obj_id}, {class_name}, {average_speed:.2f}\n")


cap.release()
cv2.destroyAllWindows()
