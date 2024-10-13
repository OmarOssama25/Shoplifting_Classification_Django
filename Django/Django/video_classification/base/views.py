import json
import cv2
import numpy as np
import os
from keras.models import model_from_json
from django.shortcuts import render
import tempfile

def load_model_from_files(config_path, weights_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_json = json.dumps(config)
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    return model

def process_and_predict_video(video_file, model, frame_skip=5):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        for chunk in video_file.chunks():
            temp_file.write(chunk)
        temp_file_path = temp_file.name

    cap = cv2.VideoCapture(temp_file_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read or video is empty.")
            break

        if frame_count % frame_skip == 0:
            resized_frame = cv2.resize(frame, (150, 150))
            normalized_frame = resized_frame / 255.0
            frames.append(normalized_frame)

        frame_count += 1

    cap.release()

    print(f"Extracted {len(frames)} frames from video.")

    if len(frames) == 0:
        print("No frames extracted from video.")
        return None

    if len(frames) < 100:
        frames += [frames[-1]] * (100 - len(frames))
    elif len(frames) > 100:
        frames = frames[:100]

    video_data = np.array(frames)
    video_data = np.expand_dims(video_data, axis=0)

    predictions = model.predict(video_data)

    os.remove(temp_file_path)

    return predictions[0][0]

def process_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']

        config_path = 'templates/config.json'
        weights_path = 'templates/model.weights.h5'
        model = load_model_from_files(config_path, weights_path)

        predicted_probability = process_and_predict_video(video_file, model)

        if predicted_probability is None:
            return render(request, 'result.html', {'predicted_label': 'Error: No valid frames extracted from the video.'})

        if predicted_probability >= 0.5:
            predicted_label = f"Shop Lifter (Probability: {predicted_probability:.2f})"
        else:
            predicted_label = f"Non Shop Lifter (Probability: {predicted_probability:.2f})"

        return render(request, 'result.html', {'predicted_label': predicted_label})

    return render(request, 'home.html')
