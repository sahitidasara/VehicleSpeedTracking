import numpy as np
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score

# Example: {object_id: ground_truth_speed (m/h)}
# Initialize a dictionary to store ground truth speeds
ground_truth_speeds = {}

# Open the log file to read the data
with open("ActualSpeed.txt", "r") as log_file:
    # Skip the header line
    next(log_file)
    
    # Read the rest of the lines
    for line in log_file:
        # Split the line by comma to extract ObjectID, ObjectClass, and Speed
        obj_id, obj_class, speed = line.strip().split(", ")
        
        # Convert speed to float (assuming the speed is in m/h)
        ground_truth_speeds[int(obj_id)] = {
                "class": obj_class,
                "speed": speed
            }

# Now, ground_truth_speeds contains the object IDs and their corresponding ground truth speeds
#print("Ground Truth Speeds:", ground_truth_speeds)


# Store model's predicted speeds
predicted_speeds = {}
# Open the log file to read the data
with open("speed_log.txt", "r") as log_file:
    # Skip the header line
    next(log_file)
    
    # Read the rest of the lines
    for line in log_file:
        # Split the line by comma to extract ObjectID, ObjectClass, and Speed
        obj_id, obj_class, speed = line.strip().split(", ")
        
        # Convert speed to float (assuming the speed is in m/h)
        predicted_speeds[int(obj_id)] = {
                "class": obj_class,
                "speed": speed
            }
        
# Now, ground_truth_speeds contains the object IDs and their corresponding ground truth speeds


# Calculate Errors
absolute_errors = []
squared_errors = []
percentage_errors = []

# Loop through ground truth speeds and predicted speeds
for obj_id, true_data in ground_truth_speeds.items():
    true_speed = float(true_data["speed"])  # Ensure speed is treated as a float
    predicted_data = predicted_speeds.get(obj_id, {"speed": 0})  # Default to 0 if not found
    predicted_speed = float(predicted_data["speed"])  # Ensure predicted speed is treated as a float
    
    
    # Calculate errors
    absolute_errors.append(abs(predicted_speed - true_speed))
    squared_errors.append((predicted_speed - true_speed) ** 2)
    percentage_errors.append(abs(predicted_speed - true_speed) / true_speed * 100)

# Calculate MAE, RMSE, and MAPE
mae = np.mean(absolute_errors)
rmse = np.sqrt(np.mean(squared_errors))
mape = np.mean(percentage_errors)

print(f"Mean Absolute Error (MAE): {mae:.2f} m/h")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} m/h")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Set a threshold to calculate accuracy (e.g., 10% error tolerance)
threshold = 10  # 10% acceptable error
accuracy = np.mean([1 if abs(float(predicted_speeds.get(obj_id, {"speed": 0})["speed"]) - float(true_data["speed"])) <= (float(true_data["speed"]) * threshold / 100) else 0
                    for obj_id, true_data in ground_truth_speeds.items()])

print(f"Accuracy: {accuracy * 100:.2f}%")

y_true = [true_data["class"] for true_data in ground_truth_speeds.values()]
y_pred = [predicted_data["class"] for predicted_data in predicted_speeds.values()]

# Get precision, recall, F1 score with zero_division handling
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average=None, zero_division=0  # You can change `zero_division` to 1 to set to 1 instead
)

# Print precision, recall, and F1 for each class
classes = sorted(set(y_true))  # List of unique class names in ground truth
for i, class_name in enumerate(classes):
    print(f"Class: {class_name}")
    print(f"  Precision: {precision[i]:.2f}")
    print(f"  Recall: {recall[i]:.2f}")
    print(f"  F1 Score: {f1[i]:.2f}")
    print()


conf_matrix = confusion_matrix(y_true, y_pred, labels=["car", "truck"])

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["car", "truck"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Object Detection Classes")
plt.show()
