import cv2

video_path = "dummy_feed.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Reached end of video, restarting...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_count += 1
    print(f"Processing frame {frame_count}")

    cv2.imshow("Dummy Video Feed", frame)
    
    # Use a short wait time to process X11 events
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

