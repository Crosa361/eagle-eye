from flask import Flask, Response
import cv2
import threading
import time
import queue

app = Flask(__name__)

# Enable OpenCV optimizations
cv2.setUseOptimized(True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the video
VIDEO_PATH = "dummy_feed.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)  # Get video FPS

# Frame Queues (Separate for Input & Processed Frames)
raw_frame_queue = queue.Queue(maxsize=1)  # Stores raw frames
processed_frame_queue = queue.Queue(maxsize=1)  # Stores processed frames

# Configuration
N = 5  # Run face detection every Nth frame
FRAME_WIDTH = 640  # Resize width for faster processing
FRAME_HEIGHT = 360  # Resize height for faster processing

# Tracking Last Processed Frame Time
last_processed_time = 0


def video_capture():
    """ Continuously captures frames and places them into the raw queue """
    global last_processed_time

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))  # Resize for speed

        if raw_frame_queue.full():
            raw_frame_queue.get()  # Remove the oldest frame

        raw_frame_queue.put((frame, time.time()))  # Store the latest raw frame with timestamp


# Start video capture thread
video_thread = threading.Thread(target=video_capture, daemon=True)
video_thread.start()


def face_detection():
    """ Runs in a separate thread, processes frames, and adds bounding boxes """
    global last_processed_time

    while True:
        if raw_frame_queue.empty():
            continue  # Wait for frames

        frame, timestamp = raw_frame_queue.get()  # Retrieve latest raw frame

        # Process frame only if enough time has passed
        if timestamp - last_processed_time >= (1 / fps):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw bounding boxes around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            last_processed_time = timestamp  # Update last processed frame time

        if processed_frame_queue.full():
            processed_frame_queue.get()  # Remove oldest frame to prevent overflow

        processed_frame_queue.put(frame)  # Push the processed frame to the output queue


# Start face detection thread
detection_thread = threading.Thread(target=face_detection, daemon=True)
detection_thread.start()


def generate_frames():
    """ Flask route generator function for video streaming """
    while True:
        if processed_frame_queue.empty():
            continue  # Wait for frames

        frame = processed_frame_queue.get()  # Retrieve the latest processed frame

        # Encode frame as JPEG (Lower Quality for Speed)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_bytes = buffer.tobytes()

        # Yield frame to browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return "<h2>Optimized Video Feed with Face Detection</h2><img src='/video_feed' width='640'>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
