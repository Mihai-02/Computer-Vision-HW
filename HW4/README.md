# Panorama Stitching

## Frame Extraction
The first step is extracting the frames from the video; this was done using the code in the file `extract_frames.py`. A total of **7 frames** were extracted from each video.

## Panorama Creation
Creating the panorama is done with the file `task.py`. Images are read from the folders `frames_1` or `frames_2`, and the grayscale version of the images is used throughout the panorama stitching process, except for the last step of determining the final image through concatenation.

The panorama stitching process consists of four steps:

## 1. Keypoint Detection
The first step, handled by the function `keypoint_detection()`, identifies distinctive features (keypoints) in each image using the **SIFT algorithm**.

### Parameters:
- **nfeatures**: The number of features to detect (`nfeatures=20000`).
- **Contrast threshold** and **edge threshold**: Used to fine-tune detection sensitivity.

Keypoints are identified and visualized on the original image. They represent visually significant parts of the image, such as corners or edges. These points are saved along with their corresponding descriptors, which describe the local neighborhood of the keypoints for later matching.

## 2. Descriptor Matching
The second step is performed by the function `descriptor_matching()`, which compares descriptors from consecutive frames to establish correspondence between features.

- Matches are identified using the **BFMatcher** algorithm with **L2 distance**.
- Matches are filtered using **Lowe's Ratio Test** (*distance < 0.8 × second-best distance*), ensuring only high-quality matches are retained.

## 3. Creating Homographies with RANSAC
The function `compute_homographies()` calculates the transformations required to align consecutive frames.

- Both **descriptor matching** and **homography computation** are handled in this function.
- For every pair of consecutive frames:
  - Points from matched keypoints are extracted from both images.
  - A **homography matrix** is calculated using `cv2.findHomography` with **RANSAC**, ensuring robust handling of outliers in the matches.
- The computed homography aligns the perspective of one frame with the next, allowing the images to overlap correctly.

## 4. Stitching the Images
The final step, managed by the function `stitch_images()`, warps and combines frames into a seamless panorama.

### Steps:
1. **Canvas Computation**
   - To ensure the entire panorama fits, the **cumulative transformation** for each frame is calculated.
   - The bounding box of all transformed frames is determined.
   - A **translation matrix** is applied to shift the panorama into a valid coordinate space.

2. **Frame Warping & Blending**
   - Each frame is warped onto the computed canvas using its cumulative homography.
   - The warped frames are blended into the panorama.
   - Overlapping regions are updated using the pixel values from the most recently added frame.

Throughout the process, intermediate results are visualized, providing insights into the stitching progress across consecutive frames.

