import glob
import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imagecodecs
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
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

def shuffle_data(features, labels):
    """同时置乱特征和标签"""
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    return features[indices], labels[indices]

# 假设已有标签数据（示例标签） 0为airplane,1为forest
labels = np.array([0]*320 + [1]*320)
# 置乱数据（在划分训练测试集之前）
combined_features, labels = shuffle_data(combined_features, labels)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    combined_features, labels, test_size=0.3, random_state=42
)

# 初始化随机森林
rf_model = RandomForestClassifier(
    n_estimators=200,      # 树的数量（越多效果越好，但计算量增大）
    max_depth=None,        # 树的最大深度（None表示不限制）
    min_samples_split=2,   # 分裂内部节点所需的最小样本数
    min_samples_leaf=1,    # 叶节点所需的最小样本数
    class_weight='balanced',  # 自动平衡类别权重（处理不平衡数据）
    random_state=42,
    n_jobs=-1             # 使用所有CPU核心并行训练
)

# 训练模型
rf_model.fit(X_train, y_train)

# 评估模型
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))

# 获取预测概率（正类的概率）
y_proba = rf_model.predict_proba(X_test)[:, 1]

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# 绘制曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc="lower right")
plt.show()