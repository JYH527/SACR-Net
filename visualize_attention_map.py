import numpy as np
import cv2
import os


def visualize_grid_attention_v2(img_path, save_path, attention_mask, ratio=1, cmap="jet", save_image=False,
                                quality=100):
    print("从以下路径加载图片尺寸信息: ", img_path)
    # 使用 OpenCV 加载图片尺寸
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法从 {img_path} 加载图片")

    orig_h, orig_w = img.shape[:2]
    img = None  # 释放图片资源

    # 调整注意力掩码大小并归一化
    mask = cv2.resize(attention_mask, (int(orig_w * ratio), int(orig_h * ratio)), interpolation=cv2.INTER_LINEAR)
    normed_mask = mask / (mask.max() + 1e-6)

    # 应用颜色映射生成热图
    normed_mask_uint8 = (normed_mask * 255).astype(np.uint8)
    if cmap.lower() == "jet":
        colormap = cv2.COLORMAP_JET
    elif cmap.lower() == "hot":
        colormap = cv2.COLORMAP_HOT
    else:
        raise ValueError(f"不支持的颜色映射: {cmap}")

    heatmap = cv2.applyColorMap(normed_mask_uint8, colormap)

    if save_image:
        os.makedirs(save_path, exist_ok=True)
        img_name = os.path.basename(img_path).split('.')[0] + "_heatmap.jpg"
        heatmap_save_path = os.path.join(save_path, img_name)
        print("保存热图至: " + heatmap_save_path)
        cv2.imwrite(heatmap_save_path, heatmap, [cv2.IMWRITE_JPEG_QUALITY, quality])
        heatmap = None  # 释放热图资源