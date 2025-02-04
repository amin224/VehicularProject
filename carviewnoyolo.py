import carla
import random
import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# Spawn Audi A2 vehicle
vehicle_bp = blueprint_library.find('vehicle.audi.a2')
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(True)

# Set up the SceneFinal camera
camera_bp = blueprint_library.find('sensor.camera.rgb')  # SceneFinal is the default for RGB camera
camera_bp.set_attribute('fov', '90')  # Field of view
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('sensor_tick', '0.05')  # Capture frequency

# Position the camera on the vehicle
camera_transform = carla.Transform(carla.Location(x=0.30, y=0, z=1.30), carla.Rotation(pitch=0, yaw=0, roll=0))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Load YOLO model
model = YOLO('yolov8n.pt')  # YOLOv8 nano model for real-time detection

# Target classes: car, bus, truck, bicycle (YOLO class IDs)
TARGET_CLASSES = [2, 3, 5, 7]  # 2: car, 3: motorcycle, 5: bus, 7: truck


# Function to draw bounding boxes
def draw_boxes(image, results):
    image = image.copy()  # ✅ Ensure the image is writable

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Confidence score
                conf = box.conf[0]

                # Class ID
                class_id = int(box.cls[0])
                label = model.names[class_id]

                if label in ["car", "bus", "bicycle", "person"]:
                    # Draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label with confidence
                    cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image


# Process the image
def process_image(image):
    # print(f"Captured image {image.frame} at {image.timestamp}")

    # Convert CARLA image to numpy array
    img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
    img_array = img_array.reshape((image.height, image.width, 4))[:, :, :3]  # Remove alpha channel

    # YOLO detection
    results = model(img_array)

    # Draw bounding boxes
    img_with_boxes = draw_boxes(img_array, results)

    # Display the image
    cv2.imshow("YOLO Detection", img_with_boxes)
    cv2.waitKey(1)


camera.listen(lambda image: process_image(image))


def smooth_spectator_follow(camera):
    spectator = world.get_spectator()
    while True:
        # Get the current camera transform
        cam_transform = camera.get_transform()

        # Offset the spectator slightly behind and above the camera to reduce vibration
        offset_location = cam_transform.location + carla.Location(z=0.5, x=-1.0)
        offset_rotation = cam_transform.rotation

        # Smooth transition with interpolation
        spectator_transform = carla.Transform(offset_location, offset_rotation)
        spectator.set_transform(spectator_transform)

        time.sleep(0.1)  # Reduced update frequency for smoothness


# Start the smooth camera follow in a separate thread
threading.Thread(target=smooth_spectator_follow, args=(camera,), daemon=True).start()


# Run simulation
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    camera.stop()
    vehicle.destroy()
    cv2.destroyAllWindows()
    print("Simulation ended.")
