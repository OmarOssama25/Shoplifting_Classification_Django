import json
import cv2
import numpy as np
import os
from keras.models import model_from_json
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tempfile

# تحميل الموديل من config.json وملف الأوزان
def load_model_from_files(config_path, weights_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_json = json.dumps(config)
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    return model

# دالة لمعالجة الفيديو والتنبؤ
def process_and_predict_video(video_file, model):
    # حفظ الفيديو في ملف مؤقت
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        for chunk in video_file.chunks():
            temp_file.write(chunk)
        temp_file_path = temp_file.name

    # استخدام OpenCV لقراءة الفيديو من الملف المؤقت
    cap = cv2.VideoCapture(temp_file_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    frames = []
    
    # استخراج الفريمات من الفيديو
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read or video is empty.")  # طباعة رسالة تفيد بعدم وجود فريمات
            break
        if frame is None:
            print("Frame is None.")  # في حالة عدم وجود فريم صالح
            break

        resized_frame = cv2.resize(frame, (128, 128))
        normalized_frame = resized_frame / 255.0
        frames.append(normalized_frame)

    cap.release()

    # تحقق من وجود فريمات
    print(f"Extracted {len(frames)} frames from video.")  # طباعة عدد الفريمات المستخرجة

    if len(frames) == 0:
        print("No frames extracted from video.")
        return None  # أو قم بإرجاع قيمة مناسبة للإشارة إلى الخطأ

    # تقليل أو زيادة عدد الفريمات إلى 100
    if len(frames) < 100:
        if frames:  # التأكد من أن frames ليست فارغة
            frames += [frames[-1]] * (100 - len(frames))  # إضافة آخر فريم لتعبئة النقص
    elif len(frames) > 100:
        frames = frames[:100]  # تقليل الفريمات إلى 100

    # تحويل الفريمات إلى مصفوفة numpy
    video_data = np.array(frames)
    video_data = np.expand_dims(video_data, axis=0)  # إضافة بعد batch

    # إجراء التنبؤ باستخدام الموديل
    predictions = model.predict(video_data)

    # الحصول على الدقة (الاحتمالية)
    predicted_probability = predictions[0][0]

    # حذف الملف المؤقت بعد المعالجة
    os.remove(temp_file_path)

    # إرجاع الدقة
    return predicted_probability

# دالة معالجة الفيديو
def process_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']

        # تحميل النموذج
        config_path = 'templates/config.json'  # تأكد من إدخال المسار الصحيح
        weights_path = 'templates/model.weights.h5'  # تأكد من إدخال المسار الصحيح
        model = load_model_from_files(config_path, weights_path)

        # إجراء التنبؤ
        predicted_probability = process_and_predict_video(video_file, model)

        if predicted_probability is None:
            return render(request, 'result.html', {'predicted_label': 'Error: No valid frames extracted from the video.'})

        # تحديد التصنيف بناءً على الاحتمالية
        if predicted_probability >= 0.5:
            predicted_label = f"Shop Lifter (Probability: {predicted_probability:.2f})"
        else:
            predicted_label = f"Non Shop Lifter (Probability: {predicted_probability:.2f})"

        return render(request, 'result.html', {'predicted_label': predicted_label})

    return render(request, 'home.html')
