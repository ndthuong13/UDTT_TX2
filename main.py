class SimpleSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters

    def fit(self, X, y):
        y_ = np.where(y == 0, -1, 1)
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.n_iters):
            for i, x in enumerate(X):
                if y_[i] * (np.dot(x, self.w) + self.b) < 1:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y_[i] * x)
                    self.b -= self.lr * y_[i]
                else:
                    self.w -= self.lr * 2 * self.lambda_param * self.w

    def predict(self, X):
        return np.where(np.dot(X, self.w) + self.b >= 0, 1, 0)

def evaluate(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    acc = (tp + tn) / len(y_true)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return acc, precision, recall

# Giả định: bạn có feature_test.npy + tập huấn luyện từ trước
X_test = np.load("feature_test.npy")
y_test = np.load("label_test.npy")  # nếu có

X_train = np.load("feature_train.npy")
y_train = np.load("label_train.npy")

svm = SimpleSVM()
svm.fit(X_train, y_train)
pred = svm.predict(X_test)

acc, prec, rec = evaluate(y_test, pred)
print(f"Accuracy: {acc:.2f} | Precision: {prec:.2f} | Recall: {rec:.2f}")