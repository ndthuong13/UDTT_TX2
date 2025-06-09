import cv2
import os
import numpy as np

def compute_hog(image_gray, cell_size=8, bin_size=9):
    image = np.float32(image_gray) / 255.0
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    h, w = image.shape
    n_cells_x = w // cell_size
    n_cells_y = h // cell_size

    hist_tensor = np.zeros((n_cells_y, n_cells_x, bin_size))

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cell_mag = magnitude[i*cell_size:(i+1)*cell_size,
                                 j*cell_size:(j+1)*cell_size]
            cell_ang = angle[i*cell_size:(i+1)*cell_size,
                             j*cell_size:(j+1)*cell_size]

            # chia góc thành các bin
            hist = np.zeros(bin_size)
            for m in range(cell_size):
                for n in range(cell_size):
                    mag = cell_mag[m, n]
                    ang = cell_ang[m, n] % 180  # HOG chỉ xét [0,180)
                    bin_idx = int(ang / (180 / bin_size))
                    hist[bin_idx] += mag
            hist_tensor[i, j, :] = hist

    return hist_tensor.ravel()

class SimpleKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        preds = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k_idxs = distances.argsort()[:self.k]
            k_labels = self.y_train[k_idxs]
            values, counts = np.unique(k_labels, return_counts=True)
            preds.append(values[np.argmax(counts)])
        return np.array(preds)

def edge_laplace(image_gray):
    lap = cv2.Laplacian(image_gray, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    return lap

def edge_sobel_gradient(image_gray):
    grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    return np.uint8(np.clip(gradient, 0, 255))


def load_images_and_labels(dataset_path, size=(64, 64)):
    images, labels = [], []
    label_dict = {}
    current_label = 0

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path): continue

        if folder not in label_dict:
            label_dict[folder] = current_label
            current_label += 1

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.resize(img, size)
            images.append(img)
            labels.append(label_dict[folder])
    return np.array(images), np.array(labels), label_dict


def compute_lbp(image_gray):
    h, w = image_gray.shape
    lbp_image = np.zeros((h-2, w-2), dtype=np.uint8)
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = image_gray[i, j]
            binary = ''
            binary += '1' if image_gray[i-1, j-1] >= center else '0'
            binary += '1' if image_gray[i-1, j  ] >= center else '0'
            binary += '1' if image_gray[i-1, j+1] >= center else '0'
            binary += '1' if image_gray[i  , j+1] >= center else '0'
            binary += '1' if image_gray[i+1, j+1] >= center else '0'
            binary += '1' if image_gray[i+1, j  ] >= center else '0'
            binary += '1' if image_gray[i+1, j-1] >= center else '0'
            binary += '1' if image_gray[i  , j-1] >= center else '0'
            lbp_value = int(binary, 2)
            lbp_image[i-1, j-1] = lbp_value
    hist, _ = np.histogram(lbp_image.flatten(), bins=256, range=(0, 256))
    return hist / (hist.sum() + 1e-7)


def kmeans_segmentation(image, k=3, max_iter=10):
    h, w, c = image.shape
    pixels = image.reshape(-1, 3).astype(np.float32)
    np.random.seed(0)
    centroids = pixels[np.random.choice(pixels.shape[0], k, replace=False)]

    for _ in range(max_iter):
        distances = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([pixels[labels == i].mean(axis=0) if len(pixels[labels == i]) > 0 else centroids[i] for i in range(k)])
        if np.allclose(centroids, new_centroids): break
        centroids = new_centroids

    segmented_img = centroids[labels].reshape(h, w, 3).astype(np.uint8)
    return segmented_img


def ahp_segmentation(image_gray, bins=4):
    hist, bin_edges = np.histogram(image_gray, bins=bins)
    thresholds = bin_edges[1:-1].astype(np.uint8)
    segmented = np.zeros_like(image_gray)
    for i in range(1, bins):
        segmented[(image_gray >= bin_edges[i-1]) & (image_gray < bin_edges[i])] = i * (255 // bins)
    return segmented


class SimpleLinearSVM:
    def __init__(self, lr=0.01, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.n_iters):
            for i, xi in enumerate(X):
                condition = y_[i] * (np.dot(xi, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(xi, y_[i]))
                    self.b -= self.lr * y_[i]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

class SimpleDecisionTree:
    def __init__(self, depth=1):
        self.depth = depth
        self.threshold = None
        self.feature_index = None
        self.left_label = None
        self.right_label = None

    def fit(self, X, y):
        m, n = X.shape
        best_gini = 1
        for feature in range(n):
            values = X[:, feature]
            threshold = np.mean(values)
            left = y[values < threshold]
            right = y[values >= threshold]
            gini = self.gini_index([left, right])
            if gini < best_gini:
                best_gini = gini
                self.threshold = threshold
                self.feature_index = feature
                self.left_label = self.majority_vote(left)
                self.right_label = self.majority_vote(right)

    def gini_index(self, groups):
        total = sum(len(group) for group in groups)
        gini = 0
        for group in groups:
            if len(group) == 0: continue
            score = sum((np.sum(group == c) / len(group))**2 for c in np.unique(group))
            gini += (1 - score) * (len(group) / total)
        return gini

    def majority_vote(self, group):
        values, counts = np.unique(group, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        preds = []
        for x in X:
            if x[self.feature_index] < self.threshold:
                preds.append(self.left_label)
            else:
                preds.append(self.right_label)
        return np.array(preds)

# images, labels, label_map = load_images_and_labels('dataset')
# X = np.array([compute_lbp(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in images])
# y = labels
#
# # Tách tập train/test
# split = int(0.8 * len(X))
# X_train, X_test = X[:split], X[split:]
# y_train, y_test = y[:split], y[split:]
#
# # SVM
# svm = SimpleLinearSVM()
# svm.fit(X_train, y_train)
# y_pred_svm = svm.predict(X_test)
# print("SVM Accuracy:", np.mean(y_pred_svm == y_test))
#
# # Tree
# tree = SimpleDecisionTree()
# tree.fit(X_train, y_train)
# y_pred_tree = tree.predict(X_test)
# print("Decision Tree Accuracy:", np.mean(y_pred_tree == y_test))
#
# # Phân đoạn
# test_img = images[0]
# gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
# seg_km = kmeans_segmentation(test_img, k=3)
# seg_ahp = ahp_segmentation(gray_img)
#
# cv2.imshow("Original", test_img)
# cv2.imshow("KMeans Segmentation", seg_km)
# cv2.imshow("AHP Segmentation", seg_ahp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Load dữ liệu
images, labels, label_map = load_images_and_labels('dataset')
gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# HOG feature
X_hog = np.array([compute_hog(img) for img in gray_images])
y = labels

# Train/test split
split = int(0.8 * len(X_hog))
X_train, X_test = X_hog[:split], X_hog[split:]
y_train, y_test = y[:split], y[split:]

# KNN
knn = SimpleKNN(k=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN Accuracy (HOG):", np.mean(y_pred_knn == y_test))

# Hiển thị ảnh biên
test_img = gray_images[0]
lap = edge_laplace(test_img)
grad = edge_sobel_gradient(test_img)

cv2.imshow("Original Gray", test_img)
cv2.imshow("Laplacian Edge", lap)
cv2.imshow("Sobel Gradient", grad)
cv2.waitKey(0)
cv2.destroyAllWindows()


