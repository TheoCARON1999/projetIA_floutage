from ultralytics import YOLO
import cv2
import numpy as np


global_filt = ["person", "car", "motorcycle"]

def play(video_path="videos/3people_dance.mp4", filt=None):
    # Filtering what to blur, add tags to blur them if they exist
    global global_filt
    if filt is not None:
        global_filt = filt

    # Load the YOLOv8 model
    model = YOLO('yolov8n-seg.pt')  # official model

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model.predict(frame, conf=0.25)
            # Visualize the results on the frame (not printing the boxes/masks/labels)
            annotated_frame = results[0].plot(boxes=False, masks=False)
            # annotated_frame = results[0].plot() #show mask + boxes

            # Make a mask around an object with segments and blurring it
            k = 0
            blurred_frame = cv2.blur(annotated_frame, (30, 30), 0)  # Making a blurred version of the frame
            mask = np.zeros((blurred_frame.shape[0], blurred_frame.shape[1], 3),
                            dtype=np.uint8)  # Making a big empty matrix
            if results[0].masks:
                for segs in results[0].masks.xy:  # for each mask containing segments
                    if (results[0].names[int(
                            results[0].boxes.cls[k].item())] in global_filt):  # Filtering what we don't want to blur
                        segs = segs.astype(np.int32, copy=False)  # float to int
                        mask = cv2.fillPoly(mask, [segs], (255, 255, 255))  # Fill the poly that will become the mask
                    k += 1
                annotated_frame = np.where(mask == (0, 0, 0), annotated_frame, blurred_frame)  # Apply the mask

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("press key 'q' to quit video !")
    print("enter 0 for camera or default for the default video.")
    video_path = input("chemin de la vidéo : ")
    if video_path == "" or video_path == "default":
        play()
    else:
        play(video_path)
