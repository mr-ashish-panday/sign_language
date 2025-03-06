from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load your trained model
model = load_model('action_detectiondem.h5')
mp_holistic = mp.solutions.holistic
actions = np.array(['accident', 'hungry', 'disaster', 'toilet', 'medicine'])

def process_frame(image_bytes):
    # Convert base64 to OpenCV image
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # MediaPipe processing
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        image, results = mediapipe_detection(image, holistic)
        keypoints = extract_keypoints(results)
    
    return keypoints

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    images = data.get('images', [])
    
    if len(images) != 30:
        return jsonify({'error': 'Exactly 30 frames required'}), 400

    # Process all frames
    sequence = []
    for img in images:
        image_data = img.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        keypoints = process_frame(image_bytes)
        sequence.append(keypoints)

    # Model prediction
    sequence = np.expand_dims(sequence, axis=0)
    prediction = model.predict(sequence)[0]
    predicted_action = actions[np.argmax(prediction)]

    return jsonify({'prediction': predicted_action})

if __name__ == '__main__':
    app.run(debug=True, port=5000)