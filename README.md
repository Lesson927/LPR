# 基于YOLOV10与PaddleOCR的车牌识别系统（YOLOv10_PaddleOCR_License_Plate_Recognition）

## 功能简介

- **车牌检测**： 用YOLOv10模型检测车牌
- **字符识别**:  用PaddleOCR识别字符
- **图像识别**:  单张图像检测识别
- **视频识别**:  识别视频中的车牌
- **实时识别**:  摄像头实时采集识别
- **图像文件夹识别**:  识别文件夹的图像并返回相应结果

## 用到的技术

- **YOLOv10**
- **PaddleOCR**
- **OS**
- **Streamlit**
- **OpenCV**
- **NumPy**
- **PIL**

## 环境配置

conda virtual environment is recommended. 
```
conda create -n myenv python=3.10
conda activate myenv
pip install paddlepaddle
pip install paddleocr
pip install streamlit
pip install ultralytics
```

### YOLOv10模型说明

我训练的模型在这个路径 YOLOv10_PaddleOCR_License_Plate_Recognition\train\weights\best.pt. 
当然你也可以自己训练模型，只需要把自己的模型在hhh.py中的第14行进行替换即可。


### 运行程序

在终端运行以下代码:

```
cd...
streamlit run hhh.py
```

### 图像识别

![LP-output](https://github.com/Lesson927/LPR/blob/main/images/output1.png)

### 文件夹处理

![LP-output](https://github.com/Lesson927/LPR/blob/main/images/output2.png)


