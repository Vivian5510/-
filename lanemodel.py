import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
from paddle.vision.transforms import Compose, ToTensor, Normalize
import os
import json
import random
import cv2
import numpy as np
from PIL import Image

# 优化后的卷积神经网络模型
class OptimizedCnnModel(nn.Layer):
    def __init__(self):
        super(OptimizedCnnModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2D(3, 32, 5, stride=2),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.Conv2D(32, 64, 5, stride=2),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 128, 3, stride=2),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 256, 3, stride=1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv2D(256, 512, 3, stride=1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 2),
        )

    def forward(self, inputs):
        x = self.features(inputs)
        return x

# 数据增强方法定义
def color_filter_autumn(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_AUTUMN)
    return im_color

def color_filter_bone(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_BONE)
    return im_color

def color_filter_winter(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_WINTER)
    return im_color

def apply_hue(img):
    low, high, prob = [-18, 18, 0.5]
    if np.random.uniform(0., 1.) < prob:
        return img

    delta = np.random.uniform(low, high)
    u = np.cos(delta * np.pi)
    w = np.sin(delta * np.pi)
    bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
    tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                     [0.211, -0.523, 0.311]])
    ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                      [1.0, -1.107, 1.705]])
    t = np.dot(np.dot(ityiq, bt), tyiq).T
    img = np.dot(img, t)
    img = np.array(img).astype(np.uint8)
    return img

def apply_saturation(img):
    low, high, prob = [0.5, 1.5, 0.5]
    if np.random.uniform(0., 1.) < prob:
        return img
    delta = np.random.uniform(low, high)

    gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
    gray = gray.sum(axis=2, keepdims=True)
    gray *= (1.0 - delta)
    img *= delta
    img += gray
    img = np.array(img).astype(np.uint8)
    return img

def apply_contrast(img):
    low, high, prob = [0.5, 1.5, 0.5]
    if np.random.uniform(0., 1.) < prob:
        return img
    delta = np.random.uniform(low, high)

    img *= delta
    img = np.array(img).astype(np.uint8)
    return img

def apply_brightness(img):
    low, high, prob = [0.5, 1.5, 0.5]
    if np.random.uniform(0., 1.) < prob:
        return img
    delta = np.random.uniform(low, high)

    img += delta
    img = np.array(img).astype(np.uint8)
    return img

def apply_hflip(img):
    return cv2.flip(img, 1)

color_maps = [
    apply_hue,
    apply_saturation,
    apply_contrast,
    apply_brightness,
    apply_hflip
]

def gen_random_ind():
    return random.randint(0, 4)

# 自定义数据集类
class MyDataSet(Dataset):
    def __init__(self, data_paths, transform=None):
        super(MyDataSet, self).__init__()
        self.data_list = []
        self.data = []
        for data_dir, label_path in data_paths:
            label_path = os.path.join(data_dir, label_path)
            with open(label_path, encoding='utf-8') as f:
                data_set = json.loads(f.read())
                for data in data_set:
                    image_name = data["img_path"]
                    label = data["state"]
                    image_path = os.path.join(data_dir, image_name)
                    self.data_list.append([image_path, label])
        self.transform = transform
        self.flag_load_all = False

    def load_alldata(self):
        if not self.flag_load_all:
            for image_path, label in self.data_list:
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image = img.resize((128, 128), Image.Resampling.LANCZOS)
                image = np.array(image).astype(np.float32)
                self.data.append([image, label])
            self.flag_load_all = True

    def __getitem__(self, index):
        image = None
        label = None
        if self.flag_load_all:
            image, label = self.data[index]
        else:
            image_path, label = self.data_list[index]
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((128, 128), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32)
            
        id = gen_random_ind()
        image = color_maps[id](image)

        if self.transform is not None:
            image = self.transform(image)
        label = np.array(label[1:])
        if id == 4:
            label = 0 - label
        label = paddle.to_tensor(label, dtype="float32")
        return image, label

    def __len__(self):
        return len(self.data_list)


import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
from paddle.io import DataLoader
import datetime
import paddle.vision.transforms as T  # 引入PaddlePaddle的transforms库
from paddle.vision.transforms import Compose, Resize, RandomResizedCrop, CenterCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, Pad, RandomAffine, Grayscale, ColorJitter, Normalize, ToTensor  # 导入常用的图像变换操作

def get_dataset(data_paths):
    # 图像数据处理方法：包含归一化和张量转换，以及各种数据增强策略
    transform4data = Compose([
        Resize((128, 128)),                      # 调整图像大小
        RandomResizedCrop(size=128, scale=(0.8, 1.0)),  # 随机裁剪并调整大小
        RandomHorizontalFlip(),                  # 随机水平翻转
        RandomVerticalFlip(),                    # 随机垂直翻转
        RandomRotation(degrees=15),              # 随机旋转
        Pad(padding=4),                          # 边缘填充
        RandomCrop(size=(128, 128)),             # 随机裁剪
        CenterCrop(size=(128, 128)),             # 中心裁剪
        RandomAffine(degrees=15, shear=10),      # 随机仿射变换，包括旋转和剪切
        Grayscale(num_output_channels=3),        # 转换为灰度图像
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机调整亮度、对比度、饱和度、色调
        BrightnessTransform(value=0.2),          # 调整亮度
        SaturationTransform(value=0.2),          # 调整饱和度
        ContrastTransform(value=0.2),            # 调整对比度
        HueTransform(value=0.2),                 # 调整色调
        RandomNoise(prob=0.5),                   # 随机添加噪声
        Normalize(mean=[127.5], std=[127.5]),    # 归一化
        ToTensor()                               # 转换为张量
    ])

    train_custom_dataset = MyDataSet(data_paths, transform=transform4data)
    print("read all data to memory")
    train_custom_dataset.load_alldata()  # 一次性读取所有数据到内存中
    return train_custom_dataset


def train():
    # 按照 年月日_小时 时间格式定义模型保存路径
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H")
    model_parmas_path, model_opt_path, checkpoint_path = get_model_path(time_str)
    
    # 定义模型
    cnn_model = CnnModel()
    
    # 获取数据，并把数据转为 paddle 数据格式
    train_paths, eval_paths = get_data_paths()
    train_custom_dataset = get_dataset(train_paths)
    eval_custom_dataset = get_dataset(eval_paths)
    print("Data read complete")
    
    # 定义数据加载器
    train_loader = DataLoader(train_custom_dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=0)
    test_loader = DataLoader(eval_custom_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=0)
    print("DataLoader defined")
    
    # 定义保存最后一次训练的检查点
    final_checkpoint = dict()
    
    # 学习率调度器和优化器
    lr_scheduler = optim.lr.CosineAnnealingDecay(learning_rate=0.001, T_max=50)
    opt = optim.AdamW(learning_rate=lr_scheduler, parameters=cnn_model.parameters(), weight_decay=0.01)
    
    # 损失函数
    loss_fn = nn.L1Loss()

    print("Training started...")
    epoch_num = train_cfg["epochs"]

    for epoch_id in range(epoch_num):
        start_time = datetime.datetime.now()
        print(f"Epoch {epoch_id}, start time: {start_time}")
        
        # 将模型及其所有子层设置为训练模式。这只会影响某些模块，如Dropout和BatchNorm。
        cnn_model.train()
        loss_sum = 0
        count = 0
        
        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data
            predicts = cnn_model(x_data)
            
            # 计算损失
            loss = loss_fn(predicts, label=y_data)
            print(f"\tBatch {batch_id}, loss MAE: {float(loss)}")
            
            # 反向传播和参数更新
            loss.backward()
            opt.step()
            opt.clear_grad()
            
            # 保存检查点信息
            final_checkpoint["loss"] = loss
            loss_sum += float(loss)
            count += 1

        # 更新学习率
        lr_scheduler.step()

        avg_loss = loss_sum / count
        print(f"Train epoch completed, cost time: {datetime.datetime.now() - start_time}, avg MAE loss: {avg_loss}\n")
        
        # 每10个epoch进行一次评估
        if epoch_id % 10 == 0:
            print(f"Eval epoch {epoch_id}, start time: {datetime.datetime.now()}")
            cnn_model.eval()
            eval_mae_loss_sum = 0
            eval_mse_loss_sum = 0
            eval_count = 0
            
            for batch_id, data in enumerate(test_loader()):
                x_data, y_data = data
                predicts = cnn_model(x_data)
                
                # 计算损失
                loss_mae = F.l1_loss(predicts, y_data)
                loss_mse = F.mse_loss(predicts, y_data)
                print(f"\tBatch {batch_id}, loss MAE: {float(loss_mae)}, MSE: {float(loss_mse)}")
                
                eval_mae_loss_sum += float(loss_mae)
                eval_mse_loss_sum += float(loss_mse)
                eval_count += 1

            avg_eval_mae = eval_mae_loss_sum / eval_count
            avg_eval_mse = eval_mse_loss_sum / eval_count
            print(f"Eval completed, cost time: {datetime.datetime.now() - start_time}, avg MAE loss: {avg_eval_mae}, avg MSE loss: {avg_eval_mse}\n")
            
        # 保存模型参数
        paddle.save(cnn_model.state_dict(), model_parmas_path)
    
    # 保存优化器参数和检查点信息
    paddle.save(opt.state_dict(), model_opt_path)
    paddle.save(final_checkpoint, checkpoint_path)


