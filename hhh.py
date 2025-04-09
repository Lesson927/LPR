# -*- coding: utf-8 -*-
import os
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageFont, ImageDraw
from paddleocr import PaddleOCR

# 设置OCR语言为中文，可同时识别中文和英文
ocr = PaddleOCR(use_angle_cls=True, lang="ch")

# 加载YOLO模型，您可以替换为自己的训练模型路径
model = YOLO(r"F:\LPR\YOLOv10_PaddleOCR_License_Plate_Recognition-main\train\weights\best.pt")

import re

def process_image(image, apply_rotation=True):
    results = model(image)

    # 设置车牌标签和置信度阈值
    license_plate_label = "license"
    confidence_threshold = 0.5

    if license_plate_label in model.names.values():
        label_index = list(model.names.values()).index(license_plate_label)
    else:
        return image

    if label_index is not None:
        for result in results:
            boxes = result.boxes.xyxy
            classes = result.boxes.cls
            confidences = result.boxes.conf
            for i, cls in enumerate(classes):
                if int(cls) == label_index and confidences[i] > confidence_threshold:
                    x1, y1, x2, y2 = map(int, boxes[i])
                    license_plate_image = image[y1:y2, x1:x2]
                    
                    # 使用OCR进行车牌识别
                    ocr_result = ocr.ocr(license_plate_image, cls=True)

                    # 确保 ocr_result 不为 None，并且是一个有效的列表，且包含需要的识别信息
                    if ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0 and \
                            isinstance(ocr_result[0], list) and len(ocr_result[0]) > 0:
                        license_plate_number = ocr_result[0][0][1][0]

                        
                        # 清洗车牌号，只保留汉字、字母和数字
                        cleaned_license_plate_number = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '', license_plate_number)
                        
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)

                        # 设置支持中文的字体路径
                        font_path = "simfang.ttf"  # 请确保此路径下的字体支持中文
                        font_size = 50
                        font = ImageFont.truetype(font_path, font_size)
                        pil_image = Image.fromarray(image)
                        draw = ImageDraw.Draw(pil_image)
                        draw.text((x1 - 40, y1 - font_size), f"{cleaned_license_plate_number.upper()}",
                                  font=font, fill=(255, 0, 0))
                        image = np.array(pil_image)
                    else:
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)

    return image


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "output_video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            processed_frame = process_image(frame, apply_rotation=False)
            frame_placeholder.image(processed_frame, channels="BGR")
            out.write(processed_frame)
        except:
            continue

    cap.release()
    out.release()

def process_live_feed():
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            processed_frame = process_image(frame, apply_rotation=False)
            frame_placeholder.image(processed_frame, channels="BGR")
        except:
            continue

    cap.release()

import re

def clean_license_plate(license_plate):
    """清洗车牌号，去除不必要的符号，只保留汉字、字母和数字"""
    return re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '', license_plate)

def extract_license_plate(image):
    results = model(image)

    # 设置车牌标签和置信度阈值
    license_plate_label = "license"
    confidence_threshold = 0.5

    if license_plate_label in model.names.values():
        label_index = list(model.names.values()).index(license_plate_label)
    else:
        return None

    if label_index is not None:
        for result in results:
            boxes = result.boxes.xyxy
            classes = result.boxes.cls
            confidences = result.boxes.conf
            for i, cls in enumerate(classes):
                if int(cls) == label_index and confidences[i] > confidence_threshold:
                    x1, y1, x2, y2 = map(int, boxes[i])
                    license_plate_image = image[y1:y2, x1:x2]
                    return license_plate_image

    return None

def process_folder(folder_path, true_labels):
    total_images = 0
    correct_predictions = 0

    # 遍历文件夹中的所有图像文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, file_name)
            image = cv2.imread(image_path)

            # 提取车牌区域
            license_plate_image = extract_license_plate(image)
            
            if license_plate_image is not None:
                # 从文件名获取真实车牌号
                true_license_plate = true_labels.get(file_name, None)

                if true_license_plate is not None:
                    # 使用OCR识别车牌
                    ocr_result = ocr.ocr(license_plate_image, cls=True)
                    # 检查 ocr_result 是否为有效结果
                    if ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0:

                        if isinstance(ocr_result[0], list) and len(ocr_result[0]) > 0:
                            recognized_license_plate = ocr_result[0][0][1][0]

                            # 清洗识别结果和真实车牌号
                            cleaned_recognized_plate = clean_license_plate(recognized_license_plate)
                            cleaned_true_plate = clean_license_plate(true_license_plate)

                            # 比较识别结果与真实车牌号
                            if cleaned_recognized_plate == cleaned_true_plate:
                                correct_predictions += 1
                            else:
                                st.write(f"文件 {file_name} 识别错误")    
                        else:
                            st.write(f"文件 {file_name} 的 OCR 结果无效")
                    else:
                        st.write(f"文件 {file_name} 的 OCR 结果为空")
                else:
                    st.write(f"文件 {file_name} 没有对应的真实车牌号")
            else:
                st.write(f"文件 {file_name} 未检测到车牌区域")

            total_images += 1

    # 计算准确率
    accuracy = correct_predictions / total_images if total_images > 0 else 0
    st.write(f"总图片数量: {total_images}")
    st.write(f"正确识别数量: {correct_predictions}")
    st.write(f"识别准确率: {accuracy * 100:.2f}%")

st.title("车牌识别")
option = st.sidebar.selectbox("选择识别类型", ("图像识别", "视频识别", "实时检测识别", "批量文件夹识别"))

if option == "图像识别":
    uploaded_file = st.file_uploader("导入图像", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        processed_image = process_image(image)
        st.image(processed_image, caption='识别处理完成')

elif option == "视频识别":
    video_file = st.file_uploader("导入视频", type=["mp4", "avi", "mov"])
    if video_file is not None:
        video_bytes = video_file.read()
        video_path = f"temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        process_video(video_path)
        os.remove(video_path)

elif option == "实时检测识别":
    process_live_feed()

elif option == "批量文件夹识别":
    folder_path = st.text_input("输入文件夹路径")
    true_labels = {
        "云A07SL9.jpg":"云A07SL9",
        "云A526EG.jpg":"云A526EG",
        "云LZV026.jpg":"云LZV026",
        "京CX8888.jpg":"京CX8888",
        "京GK8888.jpg":"京GK8888",
        "京NAN389.jpg":"京NAN389",
        "京ND0684.jpg":"京ND0684",
        "京PHE352.jpg":"京PHE352",
        "冀A0N2A6.jpg":"冀A0N2A6",
        "冀B911ZB.jpg":"冀B911ZB",
        "冀DBJ888.jpg":"冀DBJ888",
        "冀RC8266.jpg":"冀RC8266",
        "吉BCM767.jpg":"吉BCM767",
        "吉D12407.jpg":"吉D12407",
        "宁BP1168.jpg":"宁BP1168",
        "宁EU5085.jpg":"宁EU5085",
        "川A00001.jpg":"川A00001",
        "川A380EA.jpg":"川A380EA",
        "川A4D883.jpg":"川A4D883",
        "川A524MF.jpg":"川A524MF",
        "川A5907K.jpg":"川A5907K",
        "川A662HS.jpg":"川A662HS",
        "川A69E03.jpg":"川A69E03",
        "川A7DJ91.jpg":"川A7DJ91",
        "川A8H3J6.jpg":"川A8H3J6",
        "川A8PJ79.jpg":"川A8PJ79",
        "川AF741Z.jpg":"川AF741Z",
        "川AGL097.jpg":"川AGL097",
        "川AKF205.jpg":"川AKF205",
        "川AN203B.jpg":"川AN203B",
        "川AR712T.jpg":"川AR712T",
        "川ATQ621.jpg":"川ATQ621",
        "川AVH325.jpg":"川AVH325",
        "川AW2P25.jpg":"川AW2P25",
        "川AX912N.jpg":"川AX912N",
        "川AXB392.jpg":"川AXB392",
        "川AY2D79.jpg":"川AY2D79",
        "川BZN559.jpg":"川BZN559",
        "川GAZ670.jpg":"川GAZ670",
        "川J71111.jpg":"川J71111",
        "川K6A278.jpg":"川K6A278",
        "川Q79867.jpg":"川Q79867",
        "川SY3456.jpg":"川SY3456",
        "川W587AV.jpg":"川W587AV",
        "新A933PK.jpg":"新A933PK",
        "新AEA927.jpg":"新AEA927",
        "新AK3057.jpg":"新AK3057",
        "新AW6312.jpg":"新AW6312",
        "新AY9229.jpg":"新AY9229",
        "晋DLY520.jpg":"晋DLY520",
        "晋M77777.jpg":"晋M77777",
        "桂K88888.jpg":"桂K88888",
        "沪CXS990.jpg":"沪CXS990",
        "沪EC1288.jpg":"沪EC1288",
        "津ADX9521.jpg":"津ADX9521",
        "津DPA222.jpg":"津DPA222",
        "津E22602.jpg":"津E22602",
        "津KEJ156.jpg":"津KEJ156",
        "津RB7992.jpg":"津RB7992",
        "浙A9412B.jpg":"浙A9412B",
        "浙A98888.jpg":"浙A98888",
        "浙BD25158.jpg":"浙BD25158",
        "浙D5Y6K3.jpg":"浙D5Y6K3",
        "浙D976J0.jpg":"浙D976J0",
        "浙F6CN16.jpg":"浙F6CN16",
        "浙HB1111.jpg":"浙HB1111",
        "浙JD636R.jpg":"浙JD636R",
        "浙JN8888.jpg":"浙JN8888",
        "渝A9D9K6.jpg":"渝A9D9K6",
        "渝AHG207.jpg":"渝AHG207",
        "渝B5M551.jpg":"渝B5M551",
        "渝BSE100.jpg":"渝BSE100",
        "湘A3685C.jpg":"湘A3685C",
        "湘A90CJ8.jpg":"湘A90CJ8",
        "湘D6EF99.jpg":"湘D6EF99",
        "湘E43A07.jpg":"湘E43A07",
        "琼A99999.jpg":"琼A99999",
        "琼AVW306.jpg":"琼AVW306",
        "琼B0HH06.jpg":"琼B0HH06",
        "琼BD12127.jpg":"琼BD12127",
        "甘AS8B12.jpg":"甘AS8B12",
        "甘F35306.jpg":"甘F35306",
        "甘H5A220.jpg":"甘H5A220",
        "皖A22T43.jpg":"皖A22T43",
        "皖AJH155.jpg":"皖AJH155",
        "皖K00000.jpg":"皖K00000",
        "皖Q18632.jpg":"皖Q18632",
        "粤A83CU8.jpg":"粤A83CU8",
        "粤AKQ131.jpg":"粤AKQ131",
        "粤B44444.jpg":"粤B44444",
        "粤CFM951.jpg":"粤CFM951",
        "粤ESB945.jpg":"粤ESB945",
        "粤GQQ001.jpg":"粤GQQ001",
        "粤BNB001.jpg":"粤BNB001",
        "粤XK8888.jpg":"粤XK8888",
        "苏AF06226.jpg":"苏AF06226",
        "苏AH21B5.jpg":"苏AH21B5",
        "苏B669GP.jpg":"苏B669GP",
        "苏E7382B.jpg":"苏E7382B",
        "蒙C99999.jpg":"蒙C99999",
        "藏A2222B.jpg":"藏A2222B",
        "藏AR3089.jpg":"藏AR3089",
        "藏CB5555.jpg":"藏CB5555",
        "豫H00351.jpg":"豫H00351",
        "辽000250.jpg":"辽000250",
        "辽A88888.jpg":"辽A88888",
        "辽B07559.jpg":"辽B07559",
        "辽BR7777.jpg":"辽BR7777",
        "鄂AAP378.jpg":"鄂AAP378",
        "鄂AEB506.jpg":"鄂AEB506",
        "鄂Q33073.jpg":"鄂Q33073",
        "闽A0111E.jpg":"闽A0111E",
        "闽A1035D.jpg":"闽A1035D",
        "闽AWF197.jpg":"闽AWF197",
        "闽C4298S.jpg":"闽C4298S",
        "闽DD896E.jpg":"闽DD896E",
        "闽HB1508.jpg":"闽HB1508",
        "陕A67LB0.jpg":"陕A67LB0",
        "陕AG9Z10.jpg":"陕AG9Z10",
        "陕AH931Z.jpg":"陕AH931Z",
        "陕AM31B1.jpg":"陕AM31B1",
        "陕AZ0E83.jpg":"陕AZ0E83",
        "陕C44448.jpg":"陕C44448",
        "陕K75555.jpg":"陕K75555",
        "青AF609F.jpg":"青AF609F",
        "鲁AA888M.jpg":"鲁AA888M",
        "鲁AD06666.jpg":"鲁AD06666",
        "鲁BQ3T88.jpg":"鲁BQ3T88",
        "鲁BVK895.jpg":"鲁BVK895",
        "鲁CAH882.jpg":"鲁CAH882",
        "鲁CB4111.jpg":"鲁CB4111",
        "鲁HSB299.jpg":"鲁HSB299",
        "鲁J40579.jpg":"鲁J40579",
        "鲁QA8888.jpg":"鲁QA8888",
        "鲁U22159.jpg":"鲁U22159",
        "鲁V8080Z.jpg":"鲁V8080Z",
        "黑AD03210.jpg":"黑AD03210",
        "黑AF1949.jpg":"黑AF1949",
        "黑AJP250.jpg":"黑AJP250",
        "黑EAN999.jpg":"黑EAN999",
        "川ABZ1150.jpg":"川ABZ1150",
        "京BD01113.jpg":"京BD01113",
        "京Q58A77.jpg":"京Q58A77",
        "沪AF71017.jpg":"沪AF71017",
        "浙ABZ9369.jpg":"浙ABZ9369",
        "鲁B325DE.jpg":"鲁B325DE",
        "皖S66666.jpg":"皖S66666",
        "川ADF5918.jpg":"川ADF5918",
        "川AF80789.jpg":"川AF80789",
        "鲁P908H0.jpg":"鲁P908H0",
    }  # 示例真实车牌号字典，您可以根据实际情况修改
    if folder_path and os.path.exists(folder_path):
        process_folder(folder_path, true_labels)
