# Sign Language Interpreter using Action Detection and LSTM

This project utilizes TensorFlow, Keras, and MediaPipe to build a real-time sign language interpreter. By detecting keypoints (face, hand, and pose landmarks) and processing them through an LSTM-based deep learning model, the system decodes and predicts sign language sequences from video.

## Key Features
- **MediaPipe Holistic Keypoints**: Detect face, hand, and pose landmarks.
- **Action Detection with LSTM**: Handle sequences of keypoints using LSTM layers.
- **Real-Time Prediction**: Predict sign language from live video input.

## Steps
1. **Extract MediaPipe Keypoints**: Use MediaPipe to detect landmarks (face, hands, and pose).
2. **Data Collection**: Record and save keypoint sequences for training.
3. **Data Preprocessing**: Prepare the keypoint data and create labels for training.
4. **Build LSTM Model**: Train a deep learning model using LSTM layers to process the keypoint sequences.
5. **Real-Time Prediction**: Use the trained model to predict sign language in real-time using a webcam or video input.
6. **Performance Evaluation**: Evaluate the model with a confusion matrix and test its performance in real-time.

## How the Code Works
- **Keypoint Extraction**: MediaPipe Holistic is used to extract keypoints from face, hand, and pose landmarks.
- **Data Collection and Labeling**: Keypoints are stored in sequence, labeled for various signs, and saved for training the model.
- **Model Training**: A sequential deep learning model with LSTM layers is created using TensorFlow/Keras to predict actions (signs) from the keypoint sequences.
- **Real-Time Prediction**: The trained model processes real-time keypoint data from a video stream to predict the corresponding sign language gestures.
- **Evaluation**: The model's accuracy is measured using a confusion matrix and live tests.

## How to Run
1. Install dependencies:
   ```bash
   pip install mediapipe tensorflow opencv-python
   ```
2. Collect keypoints and preprocess the data.
3. Train the LSTM model.
4. Run real-time predictions using webcam input.
