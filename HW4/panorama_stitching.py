import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists("Results"):
    os.mkdir("Results")
if not os.path.exists("Results/video_1"):
    os.mkdir("Results/video_1")
if not os.path.exists("Results/video_2"):
    os.mkdir("Results/video_2")
if not os.path.exists(f"Results/video_1/keypoint_matches"):
    os.mkdir(f"Results/video_1/keypoint_matches")
if not os.path.exists(f"Results/video_2/keypoint_matches"):
    os.mkdir(f"Results/video_2/keypoint_matches")






















def Difference_of_Gaussian(image, octaves=3, scales=5, sigma=1.6):
    k = 2**(1/scales)
    gaussian_pyramid = []
    
    for octave in range(octaves):
        octave_images = []
        current_image = image.copy() if octave == 0 else cv2.resize(current_image, (current_image.shape[1] // 2, current_image.shape[0] // 2), #interpolation=cv2.INTER_CUBIC)
        interpolation=cv2.INTER_NEAREST)

        #print(current_image)

        # current_image = current_image.astype(np.uint8)
        #current_image = np.array(current_image)

        for scale in range(scales+1):
            sigma_crt = sigma * k**scale
            blurred_image = cv2.GaussianBlur(current_image, (0,0), sigma_crt)
            octave_images.append(blurred_image)
    
        gaussian_pyramid.append(octave_images)

    dog = []
    oct = 1
    for octave_images in gaussian_pyramid:
        octave_dog = []
        for i in range(1, len(octave_images)):
            difference = cv2.subtract(octave_images[i], octave_images[i-1])
            octave_dog.append(difference)

            cv2.imwrite(f"Results/dog_octave{oct}_scale{i}.png", (difference * 255).astype(np.uint8))

        oct += 1

        dog.append(octave_dog)

    return dog

def keypoint_detection_manual(image, dog_pyramid, octaves=3, scales=5):
  keypoints = []
  descriptors = []
  
  contrast_threshold=0.2
  edge_threshold=20
  epsilon = 1e-6

  orb = cv2.ORB_create(nfeatures=30000)  # You can control the number of features
  sift = cv2.SIFT_create()

  dog_pyramid = [
    [cv2.normalize(img.astype(np.float32), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) for img in octave]
    for octave in dog_pyramid
  ]

  for octave_idx, octave in enumerate(dog_pyramid):
    for scale_idx in range(1, len(octave) - 1):
      crt = octave[scale_idx]
      prev = octave[scale_idx-1]
      next = octave[scale_idx+1]

      for y in range(1, crt.shape[0]-1):
        for x in range(1, crt.shape[1]-1):
          value = crt[y,x]

          if abs(value) < contrast_threshold:
            continue

          all_neighbors = (
            prev[y-1:y+2, x-1:x+2].flatten().tolist() +
            crt[y-1:y+2, x-1:x+2].flatten().tolist() +
            next[y-1:y+2, x-1:x+2].flatten().tolist()
          )

          if max(all_neighbors)==value or min(all_neighbors)==value:
            # Compute the Hessian matrix (second-order derivatives) for edge thresholding
            dxx = crt[y, x+1] + crt[y, x-1] - 2 * value
            dyy = crt[y+1, x] + crt[y-1, x] - 2 * value
            dxy = (crt[y+1, x+1] - crt[y+1, x-1] - crt[y-1, x+1] + crt[y-1, x-1]) / 4

            # Calculate the determinant and trace of the Hessian matrix
            det = dxx * dyy - dxy * dxy
            trace = dxx + dyy

            if det < 0 or (trace**2) / (det + epsilon) > (edge_threshold + 1)**2 / edge_threshold:
              continue

            keypoints.append((x, y, octave_idx, scale_idx))

             # Extract the descriptor for the keypoint using SIFT
            #kp = cv2.KeyPoint(x, y, 1)  # size=1 for simplicity
            #kp_list = [kp]
            #print(crt.shape)
            #_, des = sift.compute(cv2.convertScaleAbs(crt), kp_list)
            #descriptors.append(des.flatten())  # Flatten to 1D



    print(f"Octave {octave_idx}: {len(keypoints)} keypoints")

  keypoints = sorted(keypoints, key=lambda x: -x[2])  # Sort by contrast value
  keypoints = keypoints[:10000]

  unique_keypoints = list({(x, y, octave, scale) for x, y, octave, scale in keypoints})
  print(f"Unique: {len(unique_keypoints)} keypoints")

  keypoints_cv2 = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in unique_keypoints]
  #keypoints_cv2, descriptors = orb.compute(image, keypoints_cv2)

  _, des = sift.compute(image, keypoints_cv2)

  return keypoints_cv2, np.array(des)


def calculate_angle(pt1, pt2):
    delta_y = pt2[1] - pt1[1]
    delta_x = pt2[0] - pt1[0]
    angle = np.arctan2(delta_y, delta_x)
    return angle

def filter_matches_by_angle(keypoints1, keypoints2, matches, angle_threshold=0.1):
    filtered_matches = []

    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        
        angle = calculate_angle(pt1, pt2)

        if abs(angle) < angle_threshold:
            filtered_matches.append(match)

    return filtered_matches













def descriptor_matching(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    return good_matches

def draw_matches(img1, keypoints1, img2, keypoints2, matches):
  r, c = img1.shape[:2]
  r1, c1 = img2.shape[:2]

  output_img = np.zeros((max([r, r1]), c+c1, 3), dtype='uint8')
  output_img[:r, :c, :] = np.dstack([img1, img1, img1])
  output_img[:r1, c:c+c1, :] = np.dstack([img2, img2, img2])

  for match in matches:
    img1_idx = match.queryIdx
    img2_idx = match.trainIdx
    (x1, y1) = keypoints1[img1_idx].pt
    (x2, y2) = keypoints2[img2_idx].pt

    cv2.circle(output_img, (int(x1),int(y1)), 4, (0, 255, 255), 1)
    cv2.circle(output_img, (int(x2)+c,int(y2)), 4, (0, 255, 255), 1)

    cv2.line(output_img, (int(x1),int(y1)), (int(x2)+c,int(y2)), (0, 255, 255), 1)
    
  return output_img


def keypoint_detection(images_gray, out_path):
    sift = cv2.SIFT_create(nfeatures=20000, nOctaveLayers=4, contrastThreshold=0.08, edgeThreshold=10)

    keypoints_descriptors = []

    for i, image in enumerate(images_gray):
        keypoints, descriptors = sift.detectAndCompute(image, None)
        keypoints_descriptors.append((keypoints, descriptors))

        keyimg = cv2.drawKeypoints(image, keypoints, None, (255, 0, 255))

        if not os.path.exists(os.path.join(out_path, f"frame_{i}")):
            os.mkdir((os.path.join(out_path, f"frame_{i}")))

        img_path = os.path.join(out_path, f"frame_{i}/keypoints_{i}.png")
        cv2.imwrite(img_path, keyimg)

        plt.imshow(cv2.drawKeypoints(image, keypoints, None, (255, 0, 255)))
        plt.show()
    
    return keypoints_descriptors


def compute_homographies(keypoints_descriptors, images_gray, out_path):
    if not os.path.exists(os.path.join(out_path, "keypoint_matches")):
        os.mkdir(os.path.join(out_path, "keypoint_matches"))

    homographies = []

    for i in range(len(images_gray) - 1):
        kp1, desc1 = keypoints_descriptors[i]
        kp2, desc2 = keypoints_descriptors[i + 1]
        
        matches = descriptor_matching(desc1, desc2)

        # Get matching points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)
        homographies.append(H)

        matchimg = draw_matches(images_gray[i], kp1, images_gray[i+1], kp2, matches[0:1000:10])
        
        img_path = os.path.join(out_path, f"keypoint_matches/matches_{i}_{i+1}.png")
        cv2.imwrite(img_path, matchimg)
        
        plt.imshow(matchimg)
        plt.show()

    return homographies


def stitch_images(images, homographies, out_path):
    if not os.path.exists(os.path.join(out_path, "panorama_steps")):
        os.mkdir(os.path.join(out_path, "panorama_steps"))

    #Start the panorama from the first image
    panorama = images[0]

    # Initialize variables to track the bounding box of the final panorama
    min_x, min_y, max_x, max_y = 0, 0, panorama.shape[1], panorama.shape[0]

    for i, H in enumerate(homographies):
        trainimg = images[i+1].copy()
    
        if i > 0:
            cumulative_H = np.dot(cumulative_H, H)
        else:
            cumulative_H = H

        h, w = trainimg.shape[:2]
    
        # Get the corners of the trainimg (next image)
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T  # corners in homogeneous coordinates
    
        # Apply the homography to the corners to get the transformed corners
        transformed_corners = np.dot(cumulative_H, corners)
        transformed_corners /= transformed_corners[2, :]  # Convert back from homogeneous coordinates
    
        # Get the bounding box of the transformed corners
        min_x = min(min_x, transformed_corners[0].min())
        max_x = max(max_x, transformed_corners[0].max())
        min_y = min(min_y, transformed_corners[1].min())
        max_y = max(max_y, transformed_corners[1].max())

    width = int(max_x - min_x)
    height = int(max_y - min_y)

    # apply the translation to shift everything into positive coordinates
    translation_matrix = np.array([[1, 0, -min_x],
                               [0, 1, -min_y],
                               [0, 0, 1]])

    panorama_translated = cv2.warpPerspective(panorama, translation_matrix, (width, height))

    for i, H in enumerate(homographies):
        trainimg = images[i+1].copy()

        if i > 0:
            cumulative_H = np.dot(cumulative_H, H)
        else:
            cumulative_H = H

        result = cv2.warpPerspective(trainimg, np.dot(translation_matrix, cumulative_H), (width, height))

        # Blend the result into the panorama
        # copy the result image into the panorama
        mask = result > 0
        panorama_translated[mask] = result[mask]

        img_path = os.path.join(out_path, f"panorama_steps/step_{i}_{i+1}.png")
        cv2.imwrite(img_path, cv2.cvtColor(panorama_translated, cv2.COLOR_BGR2RGB))

        plt.imshow(panorama_translated)
        plt.show()

    return panorama_translated


#SELECT THE VIDEO
video=1

if video==1:
    images_bgr = [cv2.imread(f'frames_1/frame_{i}.jpg') for i in range(1,8)]  # First video
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_bgr]

    out_path = os.path.join("Results", "video_1")

    #DELETE
    #out_path = os.path.join("Results", "temp")


else:
    images_bgr = [cv2.imread(f'frames_2/frame_{i}.jpg') for i in range(1,7)]  # Second video; the final panorama looks weird when using all 7 frames
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_bgr]

    out_path = os.path.join("Results", "video_2")

images_gray = []
for i in images:
    images_gray.append(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY))

# MANUAL KEYPOINT DETECTOR - bad
# keypoints_descriptors = []
# for img in images_gray:
#     dog_pyramid = Difference_of_Gaussian(img)
#     #   dog.append(dog_pyramid)
#     keypoints_descriptors_image = keypoint_detection_manual(img, dog_pyramid)
#     keypoints_descriptors.append(keypoints_descriptors_image)


keypoints_descriptors = keypoint_detection(images_gray, out_path)
homographies = compute_homographies(keypoints_descriptors, images_gray, out_path)
panorama = stitch_images(images, homographies, out_path)

img_path = os.path.join(out_path, f"final_panorama.png")
cv2.imwrite(img_path, cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))

imgplot = plt.imshow(panorama)
plt.show()
