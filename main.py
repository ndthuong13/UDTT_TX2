import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ========== I. Trích chọn đặc trưng ==========

def extract_color_histogram(image, bins=(8, 8, 8)):
    """Trích xuất histogram màu từ ảnh RGB"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def extract_lbp_features(image, num_points=24, radius=3):
    """Trích xuất đặc trưng LBP"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, num_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3),
                             range=(0, num_points + 2))
    hist = hist.astype("float")
    return hist / (hist.sum() + 1e-7)

def extract_features(image):
    """Kết hợp đặc trưng LBP + Histogram màu"""
    color_hist = extract_color_histogram(image)
    lbp_hist = extract_lbp_features(image)
    return np.hstack([color_hist, lbp_hist])


# ========== II. Load dữ liệu ảnh & trích đặc trưng ==========
def load_dataset(dataset_path):
    X = []
    y = []
    for label in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, label)
        if not os.path.isdir(class_dir): continue
        for fname in os.listdir(class_dir):
            path = os.path.join(class_dir, fname)
            image = cv2.imread(path)
            if image is None: continue
            image = cv2.resize(image, (128, 128))
            features = extract_features(image)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

X, y = load_dataset("dataset")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== III. Huấn luyện và đánh giá bộ phân loại ==========

# 1. KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"[KNN] Accuracy: {acc_knn:.2f}")

# 2. Decision Tree
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)
print(f"[Decision Tree] Accuracy: {acc_tree:.2f}")


# ========== IV. Phân đoạn ảnh sử dụng KMeans ==========
def segment_kmeans(image, k=3):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(Z)
    segmented = kmeans.cluster_centers_[labels].reshape(image.shape).astype(np.uint8)
    return segmented

# ========== V. Phân đoạn ảnh sử dụng phát hiện biên (Canny) ==========
def edge_detection_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    return edges

# ========== VI. Hiển thị ví dụ phân đoạn ảnh ==========
sample_image = cv2.imread("dataset/dog/1.jpg")
sample_image = cv2.resize(sample_image, (256, 256))

seg_kmeans = segment_kmeans(sample_image, k=3)
edges = edge_detection_canny(sample_image)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Ảnh gốc")
plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))

plt.subplot(1,3,2)
plt.title("Phân đoạn (KMeans)")
plt.imshow(cv2.cvtColor(seg_kmeans, cv2.COLOR_BGR2RGB))

plt.subplot(1,3,3)
plt.title("Biên (Canny)")
plt.imshow(edges, cmap='gray')

plt.tight_layout()
plt.show()
