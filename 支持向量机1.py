import glob
import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imagecodecs
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

def load_tiff_images(folder_path):
    """加载TIFF文件（自动处理多页、多光谱和动态范围）"""
    image_paths = glob.glob(f"{folder_path}/*.tif") + glob.glob(f"{folder_path}/*.tiff")
    images = []
    for path in image_paths:
        try:
            # 读取TIFF（支持多页和多光谱）
            tif = tifffile.imread(path)

            # 处理多页：取首页
            if tif.ndim == 3 and tif.shape[0] > 1:  # (Pages, H, W)
                frame = tif[0]
            else:
                frame = tif  # 单页 (H, W) 或 (H, W, C)

            # 动态范围归一化到0-255
            if frame.dtype != np.uint8:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # 单通道转RGB
            if frame.ndim == 2:  # (H, W)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[-1] == 1:  # (H, W, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[-1] == 4:  # 含Alpha通道
                frame = frame[..., :3]  # 去除Alpha

            images.append(frame)
        except Exception as e:
            print(f"跳过损坏文件 {path}: {e}")
    return images


# 加载所有TIFF图片
images = load_tiff_images("train")

def extract_color_histogram(image, bins=32):
    """提取HSV颜色直方图（输入需为RGB）"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])  # H通道
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])  # S通道
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    return np.hstack([hist_h, hist_s])

color_features = [extract_color_histogram(img) for img in images]



def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """提取GLCM纹理特征（输入需为单通道）"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    return np.array([contrast, energy, homogeneity])

texture_features = [extract_glcm_features(img) for img in images]


# 横向拼接特征
combined_features = np.hstack([np.array(color_features), np.array(texture_features)])

# 标准化
scaler = StandardScaler()
features_normalized = scaler.fit_transform(combined_features)

# 假设已有标签数据（示例标签）0为airplane,1为forest
labels = np.array([0]*320 + [1]*320)
indices = np.random.permutation(len(features_normalized))
features_normalized = features_normalized[indices]
labels = labels[indices]
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    features_normalized, labels, test_size=0.3, random_state=42
)

# 初始化SVM（关键参数：kernel, C, gamma）
svm_model = SVC(
    kernel='rbf',          # 径向基核函数（高斯径向基核）
    C=1.0,                 # 正则化参数（越大对误分类惩罚越强）
    gamma='scale',         # 核函数系数（'scale'自动计算）
    probability=True,      # 启用概率输出（可选）
    random_state=42
)

# 训练模型
svm_model.fit(X_train, y_train)

# 训练SVM（需启用概率输出）
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# 获取测试集的预测概率（正类的概率）
y_proba = svm_model.predict_proba(X_test)[:, 1]  # 取第二列（正类概率）

# 计算FPR, TPR, 阈值
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)  # 计算AUC值

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 随机猜测线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()