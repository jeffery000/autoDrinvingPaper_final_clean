# Autopilot-Uisee-keras-udacity
 Uisee autopilot competition by keras from hnu Kivl.
## 1.本地训练与测试的使用方法（仅需代码和数据集）
### 1.1流程图
![](process_main.png)
### 1.2 步骤
- 主目录下见流程图
- no_seg下与流程图类似
- no_pic_esti下 先run.py,保存模型，再run_lstm_conv.py（需要载入卷积模型）

## 2.与模拟器连接测试
### 2.1运行环境
本地模拟器测试的运行环境及依赖库包括如下（非唯一）：
- 计算机配置：win10 + I5 + GTX1060
- Anaconda_python-3.6.2
- Cuda-10.0, cudnn-v7.6.5.32
- 依赖库：tensorflow-gpu==1.13.1, keras==2.3.1, numpy==1.17.0

### 2.2文件说明
- Uisee666.py 为测试代码。
- KivlNet_part1.h5和KivlNet_part2.h5为训练好的模型。

### 2.3运行说明
- 打开模拟器
- Uisee666.py和两个分离训练模型文件放在同一个目录，通过cmd运行命令：

  `python3 Uisee666.py`

## 3 主要文件说明
### 3.1 主目录下文件
主目录下功能主要为一些工具函数以及基于vgg分割的卷积网络预测
- baseline_nvidia_11_14.py  训练+交叉验证 的主代码
- baseline_nvidia_prediction.py  仅预测,输出所有预测值
- lane_segment_by_vgg.py  用vgg生成分割图
- merge_data_dir.py 合并数据文件夹
- joint_vgg_kivl.py  凭借两部分网络,vgg+kivl
- down_sample_imgs.py   下采样数据集(减少样本数量)
- network.py  两部分网络的定义（pytorch才有）
- plt_loss_curve.py 绘制log文件夹下的训练图
- plt_data_compare.py  预测和实际的对比曲线图
- udacity2uisee.py 将udacity数据集转为uisee格式

### 3.2 no_pcture_esti目录
该下功能主要为一些lstm方法预测及lstm+conv方法预测，不重新训练conv，只读取h5权重，都包含画图功能
- run.py  lstm方法预测
- run_lstm_conv.py lstm+卷积方法

### 3.3 no_seg目录
去掉vgg分割网络，直接端到端的方法，各个文件名和主目录下功能一样
