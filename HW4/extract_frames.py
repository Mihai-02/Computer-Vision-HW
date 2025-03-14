import cv2
import os

def extract_frames(video_path, num_frames=7, output_dir="frames"):
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the interval between frames to extract
    
    interval = total_frames // num_frames

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract frames
    extracted_frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)

        ret, frame = cap.read()

        if ret:
            frame_filename = os.path.join(output_dir, f"frame_{i+1}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_frames.append(frame)
        else:
            print(f"Error reading frame {i+1}")

    cap.release()

    return extracted_frames

video_path = 'video_1.mp4'
#video_path = 'video_2.mp4'
extracted_frames = extract_frames(video_path, num_frames=7, output_dir="frames_2")
print(f"Extracted {len(extracted_frames)} frames.")