import glob
import tifffile
import cv2
import numpy as np
import joblib
import os
import imagecodecs
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


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


def shuffle_data(features, labels):
    """同时置乱特征和标签"""
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    return features[indices], labels[indices]





# 示例：训练随机森林分类器 0为airplane,1为forest
labels = np.array([0]*320 + [1]*320)
# 置乱数据
features_normalized, labels = shuffle_data(features_normalized, labels)
model = RandomForestClassifier(
    n_estimators=200,  # 树的数量（越多效果越好，但计算量增大）
    max_depth=None,  # 树的最大深度（None表示不限制）
    min_samples_split=2,  # 分裂内部节点所需的最小样本数
    min_samples_leaf=1,  # 叶节点所需的最小样本数
    class_weight='balanced',  # 自动平衡类别权重（处理不平衡数据）
    random_state=42,
    n_jobs=-1 ) # 使用所有CPU核心并行训练)
model.fit(features_normalized, labels)

# 创建模型保存目录
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

# 生成模型名称
model_filename = f"random_forest_model.joblib"

# 保存整个模型系统（包括标准化器）
model_system = {
    'model': model,
    'scaler': scaler,
    'feature_extractors': {
        'color_histogram': extract_color_histogram,
        'glcm_features': extract_glcm_features
    },
    'metadata': {
        'input_shape': combined_features.shape,
        'classes': model.classes_
    }
}

# 使用joblib保存
joblib.dump(model_system, os.path.join(model_dir, model_filename))


def load_and_predict(model_path, new_images_folder):
    """加载模型并对新图像进行预测"""

    # 加载模型系统
    model_system = joblib.load(model_path)

    # 获取各个组件
    model = model_system['model']
    scaler = model_system['scaler']
    extract_color = model_system['feature_extractors']['color_histogram']
    extract_glcm = model_system['feature_extractors']['glcm_features']

    # 加载新图像
    new_images = load_tiff_images(new_images_folder)

    # 提取特征（与训练时完全相同的流程）
    new_color_features = [extract_color(img) for img in new_images]
    new_texture_features = [extract_glcm(img) for img in new_images]
    new_combined = np.hstack([np.array(new_color_features), np.array(new_texture_features)])

    # 标准化（使用保存的scaler）
    new_features_normalized = scaler.transform(new_combined)

    # 预测
    predictions = model.predict(new_features_normalized)
    probabilities = model.predict_proba(new_features_normalized)

    return predictions, probabilities


# 使用示例
model_path = "saved_models/random_forest_model.joblib"  # 替换为实际路径
new_images_folder = "test"

predictions, probs = load_and_predict(model_path, new_images_folder)
print("预测结果:", predictions)
print("类别概率:", probs)