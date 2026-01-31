# 风格迁移工具

一个基于 PyTorch 和 VGG19 模型的图像风格迁移工具，带有直观的 GUI 界面，支持中文路径，可在 CPU 上运行。

## 功能特性

- 🎨 **风格迁移**：将任意图像的风格应用到另一张图像上
- 🖼️ **GUI 界面**：直观的图形界面，操作简单易用
- 🌐 **中文支持**：完全支持中文路径和界面显示
- 💻 **CPU 兼容**：优化配置，可在无 GPU 的环境下运行
- 📁 **自定义保存**：支持自定义保存路径和文件名
- 📊 **对比图生成**：自动生成内容图、风格图、结果图的对比

## 环境配置

### 依赖项

该项目需要以下 Python 库：

- `opencv-python`：图像处理
- `numpy`：数值计算
- `matplotlib`：生成对比图
- `torch`：深度学习框架
- `torchvision`：提供预训练模型
- `tkinter`：GUI 界面（Python 标准库，无需单独安装）

### 安装方法

使用 pip 安装所需依赖：

```bash
pip install opencv-python numpy matplotlib torch torchvision
```

**注意**：
- 由于使用了 CPU 运行，无需安装 CUDA 版本的 PyTorch
- 对于 Windows 用户，建议在命令提示符或 PowerShell 中执行安装命令

## 快速开始

1. **克隆或下载**：将项目文件保存到本地目录
2. **安装依赖**：执行上述 pip 命令安装所需库
3. **运行程序**：直接运行 `work.py` 文件

```bash
python work.py
```

## 使用说明

1. **选择内容图像**：点击"内容图像"区域的"选择图片"按钮，选择要保留内容的图像
2. **选择风格图像**：点击"风格图像"区域的"选择图片"按钮，选择要提取风格的图像
3. **调整风格强度**：使用"风格强度"滑块调整风格迁移的强度（范围：100,000 - 500,000）
4. **选择保存位置**：点击"保存位置"区域的"选择文件夹"按钮，选择结果保存路径
5. **自定义文件名**：在"自定义文件名"输入框中输入结果文件的名称（可选）
6. **开始迁移**：点击"控制中心"区域的"开始风格迁移"按钮
7. **查看结果**：转换完成后，程序会显示保存路径，结果包括：
   - 风格迁移图：应用了风格的内容图像
   - 对比图：内容图、风格图、结果图的并排对比

## 技术实现

### 核心算法

- **基于 Gatys 算法**：使用预训练的 VGG19 模型提取图像特征
- **内容损失**：使用 MSE 损失保持内容结构
- **风格损失**：使用 Gram 矩阵 MSE 损失匹配风格纹理
- **优化器**：使用 Adam 优化器更新生成图像的像素值

### 关键参数

- **图像尺寸**：默认 256x256（CPU 优化）
- **迭代次数**：默认 3000 次
- **学习率**：默认 0.02
- **内容特征层**：conv4_2
- **风格特征层**：conv1_1, conv2_1, conv3_1, conv4_1, conv5_1

## 项目结构

```
├── work.py         # 主程序文件
└── README.md       # 项目说明文档
```

## 注意事项

1. **性能优化**：由于使用 CPU 运行，处理较大图像可能需要较长时间
2. **内存要求**：建议至少 4GB 内存
3. **图像格式**：支持 .jpg, .jpeg, .png, .bmp 格式的图像
4. **路径长度**：请避免使用过长的文件路径，以免保存失败
5. **错误处理**：程序包含异常处理，遇到错误会显示详细信息

## 示例

### 输入输出示例

| 内容图像 | 风格图像 | 风格迁移结果 |
|---------|---------|------------|
| ![荷花](hehua.jpg) | ![星空](starry_night.jpg) | ![荷花星空风格](hehua.star.jpg) |

### 对比图

为了更直观地展示风格迁移效果，工具会自动生成包含内容原图、风格参考图和迁移结果的对比图：

![风格迁移对比图](hehua.star_对比图.jpg)

### 效果说明

- **内容图像**：一张荷花照片，保留了原始的结构和主体
- **风格图像**：梵高的《星月夜》，提供了独特的星空纹理和色彩风格
- **风格迁移结果**：荷花保持了原始形态，同时融合了《星月夜》的笔触、色彩和纹理，创造出梦幻的艺术效果
- **对比图**：清晰展示了从原始照片到艺术风格的转变过程，风格强度设为 5.00

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 完整代码

以下是 `work.py` 的完整代码实现：

```python
# 导入计算机视觉库OpenCV，用于图像读取、预处理、格式转换等核心操作
import cv2
# 导入数值计算库NumPy，用于处理图像的数组操作
import numpy as np
# 导入绘图库Matplotlib，用于生成风格迁移对比图
import matplotlib.pyplot as plt
# 导入PyTorch核心库，用于构建神经网络、张量计算、优化器定义等
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 导入TorchVision库，提供预训练模型(VGG19)和图像变换工具
from torchvision import transforms, models
# 导入警告过滤库，屏蔽无关警告（如Matplotlib字体警告、PyTorch版本警告）
import warnings

# 全局过滤警告，避免干扰程序运行
warnings.filterwarnings('ignore')

# 导入GUI相关库
import tkinter as tk  # GUI主库，用于创建窗口和控件
from tkinter import filedialog, ttk, messagebox  # 子模块：文件选择框、美化控件、消息提示框
import threading  # 线程库，用于异步执行风格迁移（避免GUI卡死）
import os  # 系统库，用于路径处理、文件操作

# 1. 全局配置与设备初始化
# 强制使用CPU运行
DEVICE = torch.device("cpu")
# 获取系统桌面路径
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
# 风格迁移结果图的默认保存路径- 改为基础默认值，后续会被自定义覆盖
DEFAULT_OUTPUT_BASE = "风格迁移结果"
DEFAULT_COMPARE_BASE = "风格迁移对比图"


#  2. 图像预处理/后处理（兼容GUI）
def preprocess_image(img_path, img_size=256):  # 默认尺寸改为CPU适配的256
    """
    图像预处理函数（兼容中文路径）：将原始图像转为PyTorch张量，适配模型输入
    :param img_path: 图像文件路径（支持中文）
    :param img_size: 目标图像尺寸（CPU默认256x256，降低计算量）
    :return: 预处理后的Tensor（shape: [1, 3, img_size, img_size]）
    """
    try:
        # 解决OpenCV读取中文路径失败问题：先读取为字节数组，再解码
        img_np = np.fromfile(img_path, dtype=np.uint8)
        # 解码字节数组为BGR格式图像（OpenCV默认格式）
        img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        # 校验图像是否读取成功
        if img_bgr is None:
            raise Exception("图像读取失败")

        # 步骤1：调整图像尺寸（保持长宽比，按最短边缩放到目标尺寸）
        h, w = img_bgr.shape[:2]  # 获取图像原始高、宽
        scale = img_size / min(h, w)  # 计算缩放比例
        new_h, new_w = int(h * scale), int(w * scale)  # 计算新尺寸
        # 双三次插值缩放（画质更优）
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), cv2.INTER_CUBIC)

        # 步骤2：居中裁剪到目标尺寸（保证图像为正方形，适配模型输入）
        top = (new_h - img_size) // 2  # 顶部裁剪偏移量
        left = (new_w - img_size) // 2  # 左侧裁剪偏移量
        img_bgr = img_bgr[top:top + img_size, left:left + img_size]

        # 步骤3：格式转换：BGR→RGB（适配Matplotlib/TorchVision）
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # 定义图像变换流水线：转为Tensor + 归一化（适配VGG19预训练模型）
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将numpy数组(0-255)转为Tensor(0-1)，维度从HWC→CHW
            # ImageNet数据集归一化参数（VGG19预训练时使用的均值和标准差）
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 执行变换，并增加batch维度（模型要求输入为[batch, channel, height, width]）
        img_tensor = transform(img_rgb).unsqueeze(0)
        # 将张量移至CPU（强制指定，核心修改点）
        return img_tensor.to(DEVICE, torch.float)
    except Exception as e:
        # 捕获预处理异常，向上抛出（便于GUI显示错误信息）
        raise Exception(f"预处理失败：{str(e)}")


def deprocess_image(tensor):
    """
    图像后处理函数：将模型输出的Tensor转回OpenCV可用的BGR格式图像
    :param tensor: 模型输出张量（shape: [1, 3, img_size, img_size]）
    :return: BGR格式numpy数组（0-255，可直接保存/显示）
    """
    # 步骤1：移除batch维度，移至CPU，转为numpy数组
    img = tensor.squeeze(0).cpu().detach().numpy()
    # 步骤2：维度转换：CHW→HWC（适配OpenCV）
    img = img.transpose(1, 2, 0)
    # 步骤3：反归一化（恢复图像原始像素范围）
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    # 步骤4：裁剪像素值到0-1范围，再转为0-255的uint8类型（图像标准格式）
    img = np.clip(img, 0, 1) * 255
    img = img.astype(np.uint8)
    # 步骤5：RGB→BGR（适配OpenCV保存/显示）
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# VGG19特征提取器
class VGG19FeatureExtractor(nn.Module):
    """
    VGG19特征提取器：加载预训练VGG19，提取指定层的内容/风格特征
    核心改进：使用固定层索引（避免动态命名匹配错误），强制部署到CPU
    """

    def __init__(self):
        super().__init__()  # 继承nn.Module的初始化
        # 适配PyTorch版本（2.0+修改了预训练模型加载方式）
        if torch.__version__ >= "2.0.0":
            # 新版PyTorch：指定权重参数加载预训练VGG19
            vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        else:
            # 旧版PyTorch：直接加载预训练VGG19
            vgg19 = models.vgg19(pretrained=True).features
        # 强制将模型部署到CPU
        vgg19 = vgg19.to(DEVICE)

        # 固定层索引（VGG19的特征层结构固定，索引不会变）
        self.content_idx = 21  # conv4_2层：提取内容特征（保留图像结构）
        self.style_idxs = [0, 5, 10, 19, 28]  # conv1_1到conv5_1：提取风格特征（纹理/色彩）
        # 裁剪VGG19到conv5_1层（后续层无意义，减少计算量）
        self.vgg = vgg19[:max(self.style_idxs) + 1]

    def forward(self, x):
        """
        前向传播：提取内容特征和风格特征
        :param x: 预处理后的图像张量
        :return: 内容特征张量、风格特征列表
        """
        content_feat = None  # 初始化内容特征
        style_feats = []  # 初始化风格特征列表
        # 遍历VGG19层，逐层计算特征
        for idx, layer in enumerate(self.vgg):
            x = layer(x)  # 执行当前层计算
            # 提取内容特征（仅conv4_2层）
            if idx == self.content_idx:
                content_feat = x
            # 提取风格特征（conv1_1/2_1/3_1/4_1/5_1层）
            if idx in self.style_idxs:
                style_feats.append(x)
        # 断言校验：确保特征提取成功（避免后续计算报错）
        assert content_feat is not None, "内容特征提取失败"
        assert len(style_feats) == 5, "风格特征提取失败"
        return content_feat, style_feats


# 4. 风格迁移核心函数（适配GUI）
def style_transfer_core(content_path, style_path, style_weight, save_dir, custom_name, img_size=256, epochs=3000):
    """
    风格迁移核心逻辑（供GUI调用）：基于Gatys算法，融合内容图和风格图
    核心修改：固定为CPU参数（img_size=256，epochs=3000，学习率0.02）
    :param content_path: 内容图路径
    :param style_path: 风格图路径
    :param style_weight: 风格强度（数值越大，风格越浓）
    :param save_dir: 保存文件夹路径
    :param custom_name: 自定义文件名
    :param img_size: 图像尺寸（CPU固定256）
    :param epochs: 迭代次数（CPU固定3000）
    :return: 结果图保存路径、对比图保存路径
    """
    try:
        # 步骤1：预处理内容图和风格图
        content_tensor = preprocess_image(content_path, img_size)
        style_tensor = preprocess_image(style_path, img_size)
        # 初始化生成图像（以内容图为起点，后续优化像素值），开启梯度计算
        gen_tensor = content_tensor.clone().requires_grad_(True)

        # 步骤2：加载特征提取器，设置为评估模式（冻结参数，不训练）
        extractor = VGG19FeatureExtractor()
        extractor.eval()  # 评估模式：禁用Dropout/BatchNorm等训练层
        # 冻结VGG19所有参数（仅优化生成图像的像素，不优化特征提取器）
        for param in extractor.parameters():
            param.requires_grad = False

        # 步骤3：预提取内容图的内容特征、风格图的风格特征（避免重复计算）
        content_target, _ = extractor(content_tensor)
        _, style_target = extractor(style_tensor)

        # 步骤4：定义优化器（仅优化生成图像的像素值）
        # CPU固定学习率0.02
        lr = 0.02
        # 使用Adam优化器（收敛快、稳定性好）
        optimizer = optim.Adam([gen_tensor], lr=lr)

        # 步骤5：迭代优化（最小化内容损失+风格损失）
        for epoch in range(epochs):
            optimizer.zero_grad()  # 清空梯度（避免累积）
            # 提取生成图像的内容特征和风格特征
            gen_content, gen_style = extractor(gen_tensor)

            # 计算内容损失（MSE损失：保持生成图与内容图的结构一致）
            c_loss = F.mse_loss(content_target, gen_content)
            # 计算风格损失（格拉姆矩阵MSE：匹配生成图与风格图的纹理/色彩）
            s_loss = 0
            for s_feat, g_feat in zip(style_target, gen_style):
                # 计算格拉姆矩阵（捕捉风格的纹理特征，忽略空间位置）
                _, c, h, w = s_feat.size()
                s_gram = torch.mm(s_feat.view(c, h * w), s_feat.view(c, h * w).t()) / (c * h * w)
                g_gram = torch.mm(g_feat.view(c, h * w), g_feat.view(c, h * w).t()) / (c * h * w)
                s_loss += F.mse_loss(s_gram, g_gram)

            # 总损失：内容损失（权重1） + 风格损失（权重style_weight）
            total_loss = 1 * c_loss + style_weight * s_loss
            total_loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新生成图像的像素值

        # 步骤6：构建最终保存路径（自定义+默认兼容）
        # 处理自定义名称：为空则用默认名
        if not custom_name.strip():
            output_name = DEFAULT_OUTPUT_BASE
            compare_name = DEFAULT_COMPARE_BASE
        else:
            output_name = custom_name.strip()
            compare_name = f"{custom_name.strip()}_对比图"
        # 拼接完整路径（自动加.jpg后缀）
        output_path = os.path.join(save_dir, f"{output_name}.jpg")
        compare_path = os.path.join(save_dir, f"{compare_name}.jpg")

        # 步骤7：后处理生成图像，并保存（兼容中文路径）
        gen_img = deprocess_image(gen_tensor)
        # 保存结果图：先编码为jpg格式，再写入文件（解决中文路径保存失败）
        cv2.imencode('.jpg', gen_img)[1].tofile(output_path)

        # 步骤8：生成对比图（传入风格强度，用于标题显示）
        generate_compare_image(content_tensor, style_tensor, gen_tensor, compare_path, style_weight)
        # 返回保存路径（供GUI显示）
        return output_path, compare_path
    except Exception as e:
        # 捕获迁移过程中的异常，向上抛出
        raise Exception(f"风格迁移失败：{str(e)}")


def generate_compare_image(content_tensor, style_tensor, gen_tensor, save_path, style_weight):
    """
    生成内容图、风格图、结果图的对比图，并保存
    :param content_tensor: 内容图张量
    :param style_tensor: 风格图张量
    :param gen_tensor: 生成图张量
    :param save_path: 对比图保存路径
    :param style_weight: 风格强度（用于标题显示）
    """
    # 后处理三张图像，转为OpenCV可用格式
    content_img = deprocess_image(content_tensor)
    style_img = deprocess_image(style_tensor)
    gen_img = deprocess_image(gen_tensor)
    # 统一风格图尺寸（与内容图/结果图一致，CPU固定256）
    style_img = cv2.resize(style_img, (256, 256), cv2.INTER_CUBIC)

    # 转换为RGB格式（适配Matplotlib显示）
    content_rgb = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)
    style_rgb = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
    gen_rgb = cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR)

    # 设置Matplotlib中文字体（避免乱码）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 创建画布（尺寸18x6英寸，分辨率100）
    plt.figure(figsize=(18, 6), dpi=100)

    # 子图1：内容原图
    plt.subplot(1, 3, 1)  # 1行3列，第1个位置
    plt.imshow(content_rgb)
    plt.axis('off')  # 关闭坐标轴
    plt.title("内容原图", fontsize=14)

    # 子图2：风格参考图
    plt.subplot(1, 3, 2)  # 1行3列，第2个位置
    plt.imshow(style_rgb)
    plt.axis('off')
    plt.title("风格参考图", fontsize=14)

    # 子图3：生成结果图
    plt.subplot(1, 3, 3)  # 1行3列，第3个位置
    plt.imshow(gen_rgb)
    plt.axis('off')
    # 标题显示风格强度（归一化显示，更直观）
    plt.title(f"风格迁移结果（强度：{style_weight / 100000:.2f}）", fontsize=14)

    # 调整子图间距，避免重叠
    plt.tight_layout()
    # 保存对比图（bbox_inches='tight'：去除空白边框）
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 关闭画布，释放内存


#  5. GUI 界面封装
class StyleTransferGUI:
    """风格迁移GUI主类：封装所有界面控件和交互逻辑"""

    def __init__(self, root):
        """
        初始化GUI
        :param root: tkinter主窗口对象
        """
        self.root = root  # 保存主窗口引用
        self.root.title("风格迁移工具")  # 修改标题
        self.root.geometry("800x500")  # 修改窗口尺寸
        self.root.resizable(False, False)  # 固定窗口大小（不可缩放）

        # 初始化全局变量（存储用户选择的路径和参数）
        self.content_path = None  # 内容图路径（初始为None）
        self.style_path = None  # 风格图路径（初始为None）
        # 风格强度变量（tkinter变量，绑定滑块，默认值100000）
        self.style_weight = tk.IntVar(value=100000)
        # 新增：保存文件夹路径（默认桌面）
        self.save_dir_path = tk.StringVar(value=DESKTOP_PATH)
        # 新增：自定义文件名（默认空）
        self.custom_filename = tk.StringVar(value="")

        # 构建GUI界面（调用内部方法）
        self._build_ui()

    def _build_ui(self):
        """构建GUI界面控件（内部方法，仅类内调用）"""
        # 1. 内容图像选择区域（带边框的标签框）
        frame_content = ttk.LabelFrame(self.root, text="内容图像", padding=10)
        frame_content.place(x=20, y=20, width=300, height=100)  # 定位：x,y坐标 + 宽高

        # 显示当前选择的内容图文件名（初始显示"未选择图片"）
        self.label_content = ttk.Label(frame_content, text="未选择图片")
        self.label_content.pack(side=tk.LEFT, padx=5)  # 左对齐，左右内边距5

        # 内容图选择按钮（点击触发_select_content_img方法）
        btn_content = ttk.Button(frame_content, text="选择图片", command=self._select_content_img)
        btn_content.pack(side=tk.RIGHT, padx=5)  # 右对齐，左右内边距5

        # 2. 风格图像选择区域（结构同内容图）
        frame_style = ttk.LabelFrame(self.root, text="风格图像", padding=10)
        frame_style.place(x=20, y=140, width=300, height=100)

        self.label_style = ttk.Label(frame_style, text="未选择图片")
        self.label_style.pack(side=tk.LEFT, padx=5)

        btn_style = ttk.Button(frame_style, text="选择图片", command=self._select_style_img)
        btn_style.pack(side=tk.RIGHT, padx=5)

        # 3. 风格强度调整区域
        frame_weight = ttk.LabelFrame(self.root, text="风格强度", padding=10)
        frame_weight.place(x=20, y=260, width=300, height=100)

        # 风格强度滑块（范围10万~50万，水平方向，绑定style_weight变量）
        scale_weight = ttk.Scale(frame_weight, from_=100000, to=500000,
                                 variable=self.style_weight, orient=tk.HORIZONTAL,
                                 # 滑块拖动时触发_update_weight_label方法，实时更新显示
                                 command=lambda v: self._update_weight_label())
        scale_weight.pack(fill=tk.X, padx=5, pady=5)  # 水平填充，内边距5

        # 显示当前风格强度（格式化显示，带千位分隔符）
        self.label_weight = ttk.Label(frame_weight, text=f"当前强度：{self.style_weight.get():,}")
        self.label_weight.pack()

        # 4. 保存路径选择区域
        frame_save_dir = ttk.LabelFrame(self.root, text="保存位置", padding=10)
        frame_save_dir.place(x=20, y=380, width=300, height=80)

        # 显示当前保存文件夹（仅显示最后一级目录，避免过长）
        self.label_save_dir = ttk.Label(frame_save_dir, text=os.path.basename(self.save_dir_path.get()))
        self.label_save_dir.pack(side=tk.LEFT, padx=5)

        # 保存路径选择按钮
        btn_save_dir = ttk.Button(frame_save_dir, text="选择文件夹", command=self._select_save_dir)
        btn_save_dir.pack(side=tk.RIGHT, padx=5)

        # 5. 自定义文件名区域
        frame_filename = ttk.LabelFrame(self.root, text="自定义文件名", padding=10)
        frame_filename.place(x=350, y=20, width=430, height=100)

        # 文件名输入框
        entry_filename = ttk.Entry(frame_filename, textvariable=self.custom_filename, font=("SimHei", 10))
        entry_filename.pack(fill=tk.X, padx=5, pady=5)  # 水平填充，内边距5
        # 提示标签
        ttk.Label(frame_filename, text="提示：为空则使用默认名称，无需输入.jpg后缀").pack(padx=5)

        # 6. 控制与状态区域（右侧，调整y坐标）
        frame_control = ttk.LabelFrame(self.root, text="控制中心", padding=10)  # 标注CPU版本
        frame_control.place(x=350, y=140, width=430, height=320)

        # 开始风格迁移按钮（点击触发_start_transfer方法）
        self.btn_start = ttk.Button(frame_control, text="开始风格迁移", command=self._start_transfer)
        self.btn_start.pack(pady=20)  # 上下内边距20

        # 状态提示标签（显示当前程序状态，字体放大）
        self.label_status = ttk.Label(frame_control, text="状态：等待选择图片", font=("SimHei", 12))  # 标注CPU
        self.label_status.pack(pady=20)

        # 结果路径提示标签（换行显示，最大宽度400）
        self.label_result = ttk.Label(frame_control, text="", wraplength=400)
        self.label_result.pack(pady=10)

    def _select_content_img(self):
        """内容图选择按钮回调：打开文件选择框，保存选择的路径"""
        # 打开文件选择框，限定图像格式
        path = filedialog.askopenfilename(
            title="选择内容图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        # 若用户选择了文件（未取消）
        if path:
            self.content_path = path  # 保存路径
            # 更新标签显示（仅显示文件名，避免路径过长）
            self.label_content.config(text=os.path.basename(path))
            # 更新状态提示
            self.label_status.config(text="状态：已选择内容图，等待选择风格图")

    def _select_style_img(self):
        """风格图选择按钮回调（逻辑同内容图）"""
        path = filedialog.askopenfilename(
            title="选择风格图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        if path:
            self.style_path = path
            self.label_style.config(text=os.path.basename(path))
            self.label_status.config(text="状态：图片已选，可开始转换")

    # 选择保存文件夹的回调方法
    def _select_save_dir(self):
        """保存路径选择按钮回调：打开文件夹选择框"""
        dir_path = filedialog.askdirectory(title="选择保存文件夹")
        if dir_path:
            self.save_dir_path.set(dir_path)
            # 更新标签显示（仅显示最后一级目录）
            self.label_save_dir.config(text=os.path.basename(dir_path))
            self.label_status.config(text="状态：已选择保存文件夹")

    def _update_weight_label(self):
        """风格强度滑块回调：实时更新强度显示标签"""
        weight = self.style_weight.get()  # 获取当前滑块值
        # 更新标签（格式化显示，带千位分隔符）
        self.label_weight.config(text=f"当前强度：{int(weight):,}")

    def _start_transfer(self):
        """开始迁移按钮回调：校验参数，开启线程执行迁移"""
        # 校验：用户是否选择了内容图和风格图
        if not self.content_path or not self.style_path:
            messagebox.warning("提示", "请先选择内容图和风格图！")  # 弹出警告框
            return

        # 禁用开始按钮（避免重复点击）
        self.btn_start.config(state=tk.DISABLED)
        # 更新状态提示
        self.label_status.config(text="状态：正在转换中...")
        self.label_result.config(text="")  # 清空结果提示

        # 开启后台线程执行风格迁移（避免GUI卡死）
        thread = threading.Thread(
            target=self._transfer_worker,  # 线程执行的方法
            args=(
                self.content_path,
                self.style_path,
                self.style_weight.get(),
                self.save_dir_path.get(),  # 传递保存文件夹
                self.custom_filename.get()  # 传递自定义文件名
            )
        )
        thread.daemon = True  # 设置为守护线程（主窗口关闭时，线程自动退出）
        thread.start()  # 启动线程

    def _transfer_worker(self, content_path, style_path, style_weight, save_dir, custom_name):
        """风格迁移工作线程（后台执行，不阻塞GUI）"""
        try:
            # CPU固定参数
            img_size = 256
            epochs = 3000

            # 执行风格迁移核心逻辑（传入保存路径和自定义名称）
            output_path, compare_path = style_transfer_core(
                content_path, style_path, style_weight, save_dir, custom_name, img_size, epochs
            )

            # 更新GUI状态（必须用after方法，确保在主线程执行）
            self.root.after(0, self._update_status,
                            "状态：转换完成！",  # 状态提示
                            # 结果路径提示（换行显示）
                            f"结果已保存：\n1. 风格迁移图：{output_path}\n2. 对比图：{compare_path}")
        except Exception as e:
            # 捕获异常，更新GUI显示错误信息
            self.root.after(0, self._update_status,
                            f"状态：转换失败！",
                            f"错误原因：{str(e)}")
        finally:
            # 恢复开始按钮（无论成功/失败，都启用）
            self.root.after(0, lambda: self.btn_start.config(state=tk.NORMAL))

    def _update_status(self, status, result):
        """更新GUI状态和结果标签（主线程执行）"""
        self.label_status.config(text=status)  # 更新状态
        self.label_result.config(text=result)  # 更新结果路径


# 6. 运行GUI
if __name__ == "__main__":
    # 创建tkinter主窗口
    root = tk.Tk()
    # 实例化GUI类
    app = StyleTransferGUI(root)
    # 全局设置tkinter字体（解决中文显示问题）
    root.option_add("*Font", "SimHei 9")
    # 启动GUI主循环（阻塞，直到窗口关闭）
    root.mainloop()
```

---

**如有任何问题或建议，欢迎联系作者。**