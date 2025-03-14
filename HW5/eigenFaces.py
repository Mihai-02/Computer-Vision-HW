import numpy as np
import os
import cv2

def load_and_process_images(folder, target_size=(40, 40)):
    images, labels = [], []

    for image_name in os.listdir(folder):
        image_path = os.path.join(folder, image_name)
        img = cv2.imread(image_path)

        label = int(image_name.split("_")[0])

        #Image processing: resizing and normalization
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, target_size).flatten()
        img = img / 255.0

        images.append(img)
        labels.append(label)

    images = np.array(images)
    return images, labels

def center_imgs(images):
    mean_img = np.mean(images, axis=0)
    return images - mean_img, mean_img

def pca(train_images, train_labels, m_components=20):
    centered_images, mean_face = center_imgs(train_images)          # the A matrix (transposed, compared to the paper)

    #Covariance matrix
    L = np.dot(centered_images, centered_images.T)

    eigenvalues, eigenvectors = np.linalg.eigh(L)

    print(eigenvectors.shape)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvectors = eigenvectors[:, :m_components]

    print(eigenvectors.shape)

    eigenfaces = np.dot(centered_images.T, eigenvectors)
 
    final_eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)

    projected_train = np.dot(centered_images, final_eigenfaces)

    return final_eigenfaces, mean_face, projected_train

def compute_class_vectors(train_projected, train_labels, num_classes):
    class_vectors = []

    train_labels = np.ravel(train_labels).astype(int)

    for k in range(1, num_classes+1):
        class_indices = np.where(train_labels == k)[0]
        #class_proj = train_projected[:, class_indices]  # Select projections of class k
        class_proj = train_projected[class_indices]  # Select samples of class k
        class_mean = np.mean(class_proj[:6], axis=0)  # Compute mean projection vector (avg of eigenface representation over a small nr of face images (according to the paper))
        class_vectors.append(class_mean)
    return np.stack(class_vectors, axis=0)

def project_test_image(test_image, eigenfaces, mean_face):
    test_centered = test_image - mean_face
    return np.dot(test_centered, eigenfaces)

def classify_test_image(test_proj, class_vectors, threshold=None):
    distances = [np.linalg.norm(test_proj - class_vec) for class_vec in class_vectors]
    min_dist = np.min(distances)
    if threshold is not None and min_dist > threshold:
        return "Unknown", min_dist
    predicted_class = np.argmin(distances) + 1
    return predicted_class, min_dist

train_images, train_labels = load_and_process_images("Faces/Train", target_size=(40,40))
eigenfaces, mean_face, train_projected = pca(train_images, train_labels, 30)

test_images, test_labels = load_and_process_images("Faces/Test", target_size=(40,40))

num_classes = len(np.unique(train_labels))
class_vectors = compute_class_vectors(train_projected, train_labels, num_classes)

accuracy = 0.0
num_images = 0
num_corrects = 0.0
for img, true_label in zip(test_images, test_labels):
    proj_img = project_test_image(img, eigenfaces, mean_face)

    predicted_label, distance = classify_test_image(proj_img, class_vectors, threshold=None)
    print(f"True: {true_label}, Predicted: {predicted_label}, Distance: {distance}")
    num_images += 1
    if true_label == predicted_label:
        num_corrects += 1

print("Accuracy: ", num_corrects / num_images)

