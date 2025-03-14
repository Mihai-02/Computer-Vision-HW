# Human Skin Segmentation

The process of image skin segmentation is done in four steps:

## 1. Thresholding the Image Based on a Range of Colors
The first step is handled by the function `filter_color()`, which applies two different color masks on the original image.

- The two ranges of colors used for filtering were obtained from manually analyzing multiple images of human faces.

## 2. Removing Noisy Components
The second step is performed by the function `remove_noise()`.

- First, all the connected components are determined using `cv2.connectedComponentsWithStats`.
- The function receives the image from step 1 and a **connectivity value of 8**, meaning the neighbors of a pixel are determined by looking at all horizontal, vertical, and diagonal directions.
- From all the identified components, only those with an **area greater than 300 pixels** are kept.
- The final image contains only the large structures.

## 3. Dilating and Eroding the Image
This step is performed by the function `closing()`.

- It receives as parameters:
  - **Kernel size**: Defines the structuring element used in morphological operations.
  - **Number of iterations**: Determines how many times the operations are applied.
- The same values are used for both dilation and erosion to preserve the original structures while removing small holes inside them.
- After testing multiple values, the best results were achieved with:
  - **Kernel size = 5**
  - **Iterations = 7**

## 4. Extracting the Elliptical Shapes
The final step is executed by the function `extract_faces()`.

- It takes as arguments:
  - The image from the previous step.
  - The number of faces in the image (`num_faces`).
  - The original image, over which ellipses are drawn to indicate detected faces.
- **Contours are identified** using `cv2.findContours` and sorted in decreasing order of area.
- Only the **num_faces largest components** are analyzed.
- Shapes are considered faces **only if their aspect ratio is smaller than 0.8**.
  - The bounding rectangle of the shape must be a **vertical rectangle** to contain an oval face shape.

