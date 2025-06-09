import os
import cv2
import numpy as np

# ==== Bước 1: Load dataset ====

def load_images_and_labels(dataset_path, size=(64, 64)):
    images, labels, label_dict = [], [], {}
    label_id = 0
    for folder in os.listdir(dataset_path):
        sub_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(sub_path): continue
        label_dict[folder] = label_id
        for file in os.listdir(sub_path):
            file_path = os.path.join(sub_path, file)
            img = cv2.imread(file_path)
            if img is None: continue
            img = cv2.resize(img, size)
            images.append(img)
            labels.append(label_id)
        label_id += 1
    return np.array(images), np.array(labels), label_dict

# ==== Bước 2: LBP ====
def compute_lbp(image_gray):
    h, w = image_gray.shape
    lbp = np.zeros((h-2, w-2), dtype=np.uint8)
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = image_gray[i, j]
            binary = [
                image_gray[i-1,j-1] >= center,
                image_gray[i-1,j  ] >= center,
                image_gray[i-1,j+1] >= center,
                image_gray[i  ,j+1] >= center,
                image_gray[i+1,j+1] >= center,
                image_gray[i+1,j  ] >= center,
                image_gray[i+1,j-1] >= center,
                image_gray[i  ,j-1] >= center,
            ]
            lbp[i-1, j-1] = sum([val << idx for idx, val in enumerate(binary)])
    hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
    return hist / (hist.sum() + 1e-6)

# ==== Bước 3: HOG ====
def compute_hog(image_gray, cell_size=8, bin_size=9):
    image = np.float32(image_gray) / 255.0
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    h, w = image.shape
    n_cells_x = w // cell_size
    n_cells_y = h // cell_size
    hog = np.zeros((n_cells_y, n_cells_x, bin_size))
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cell_mag = mag[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_ang = angle[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            hist = np.zeros(bin_size)
            for u in range(cell_size):
                for v in range(cell_size):
                    bin_idx = int(cell_ang[u,v] % 180 / (180/bin_size))
                    hist[bin_idx] += cell_mag[u,v]
            hog[i,j] = hist
    return hog.flatten()

# ==== Bước 4: SVM tuyến tính ====
class SimpleSVM:
    def __init__(self, lr=0.01, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters

    def fit(self, X, y):
        y_ = np.where(y == 0, -1, 1)
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.n_iters):
            for i, x in enumerate(X):
                condition = y_[i] * (np.dot(x, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2*self.lambda_param*self.w)
                else:
                    self.w -= self.lr * (2*self.lambda_param*self.w - y_[i]*x)
                    self.b -= self.lr * y_[i]

    def predict(self, X):
        return np.where(np.dot(X, self.w) + self.b >= 0, 1, 0)

# ==== Bước 5: Decision Tree đơn giản ====
class SimpleDecisionTree:
    def fit(self, X, y):
        self.feature = np.argmax(np.var(X, axis=0))
        threshold = np.mean(X[:, self.feature])
        self.threshold = threshold
        left = y[X[:, self.feature] < threshold]
        right = y[X[:, self.feature] >= threshold]
        self.left = np.argmax(np.bincount(left)) if len(left) else 0
        self.right = np.argmax(np.bincount(right)) if len(right) else 0

    def predict(self, X):
        return np.where(X[:, self.feature] < self.threshold, self.left, self.right)

# ==== Bước 6: KNN ====
class SimpleKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        preds = []
        for x in X:
            dists = np.linalg.norm(self.X - x, axis=1)
            idxs = np.argsort(dists)[:self.k]
            votes = self.y[idxs]
            preds.append(np.bincount(votes).argmax())
        return np.array(preds)

# ==== Bước 7: Phân đoạn ảnh: KMeans ====
def kmeans_segmentation(image, k=3, max_iter=10):
    pixels = image.reshape(-1, 3).astype(np.float32)
    centroids = pixels[np.random.choice(pixels.shape[0], k, replace=False)]
    for _ in range(max_iter):
        distances = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        for i in range(k):
            if np.any(labels == i):
                centroids[i] = pixels[labels == i].mean(axis=0)
    segmented = centroids[labels].reshape(image.shape).astype(np.uint8)
    return segmented

# ==== Bước 8: Phân đoạn ảnh AHP ====
def ahp_segmentation(image_gray, bins=4):
    hist, bin_edges = np.histogram(image_gray, bins=bins)
    seg = np.zeros_like(image_gray)
    for i in range(bins):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        seg[(image_gray >= lower) & (image_gray < upper)] = i * (255 // bins)
    return seg

# ==== Bước 9: Laplace / Gradient ====
def edge_laplace(gray):
    return cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))

def edge_gradient(gray):
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))

# ==== Main ====
if __name__ == "__main__":
    images, labels, label_map = load_images_and_labels("dataset")
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    print("Tổng ảnh:", len(images), "Số lớp:", len(label_map))

    # LBP
    X_lbp = np.array([compute_lbp(img) for img in gray_images])
    y = labels
    split = int(0.8 * len(X_lbp))
    X_train, X_test = X_lbp[:split], X_lbp[split:]
    y_train, y_test = y[:split], y[split:]

    # KNN
    knn = SimpleKNN(k=3)
    knn.fit(X_train, y_train)
    pred_knn = knn.predict(X_test)
    print("KNN Accuracy (LBP):", np.mean(pred_knn == y_test))

    # SVM
    if len(np.unique(y)) == 2:
        svm = SimpleSVM()
        svm.fit(X_train, y_train)
        pred_svm = svm.predict(X_test)
        print("SVM Accuracy (LBP):", np.mean(pred_svm == y_test))

    # Decision Tree
    tree = SimpleDecisionTree()
    tree.fit(X_train, y_train)
    pred_tree = tree.predict(X_test)
    print("Decision Tree Accuracy (LBP):", np.mean(pred_tree == y_test))

    # Phân đoạn ảnh
    img0 = images[0]
    gray0 = gray_images[0]

    seg_kmeans = kmeans_segmentation(img0)
    seg_ahp = ahp_segmentation(gray0)
    edge_lap = edge_laplace(gray0)
    edge_grad = edge_gradient(gray0)

    cv2.imshow("Original", img0)
    cv2.imshow("KMeans Segmentation", seg_kmeans)
    cv2.imshow("AHP Segmentation", seg_ahp)
    cv2.imshow("Edge Laplace", edge_lap)
    cv2.imshow("Edge Gradient", edge_grad)
    cv2.waitKey(0)