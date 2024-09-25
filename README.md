# Saudi Sign Gesture Recognition System

## Introduction

This project is a **Saudi Sign Gesture Recognition System** that detects hand gestures and converts them into corresponding speech using the **Google Cloud Text-to-Speech API**. The system recognizes both **sign language gestures** (using one hand) and **numbers** (using two hands). The detected gestures or numbers are then converted into audio, and the system plays the spoken output in English/Arabic.

The core of the system relies on **MediaPipe** for hand landmark detection, **pre-trained machine learning models** for recognizing gestures and numbers, and Google’s **Text-to-Speech (TTS)** service for converting the recognized gestures/numbers into spoken words. Both the sign and number models are trained using **multilayer perceptrons** on a large dataset and have undergone **hyperparameter tuning** to optimize performance.

## How the System Works

1. **Hand Detection and Tracking**:
   - The system captures real-time video from a webcam using **OpenCV**.
   - It uses **MediaPipe** to detect hand landmarks and track hand movements. It works with either one or two hands at a time.

2. **Gesture and Number Recognition**:
   - For **sign language gestures**: If only one hand is detected, the system predicts the hand gesture using a pre-trained model (`finalized_model_hyp_onlysigns.sav`), which classifies the gesture based on hand landmarks. Some preprocessing is carried out before passing the data to the model.
   - For **numbers**: If two hands are detected, the system uses another pre-trained model (`numbers_model_iter2.sav`) to predict the number represented by the hand signs.

3. **Text-to-Speech Conversion**:
   - Once a gesture or number is recognized and held for 5 seconds, the corresponding text is passed to the Google Cloud TTS API, which synthesizes speech in English/Arabic.
   - The synthesized audio is saved as an MP3 file and played back through the speakers.

This system is a powerful tool for enhancing communication through gesture recognition and speech synthesis, making it particularly useful for educational and assistive technologies.

---

# Project: Audio Models Testing (English to Arabic Translation)

## Files for Testing

Please download the following files for testing:

- **Jupyter Notebook for Audio Models (English to Arabic)**
- **Model Files**
- **Google Cloud Text-to-Speech API**: A JSON file for Google TTS is included.

### Prerequisites

- **Python Version**: The project was developed using Python 3.6.
- **Dependencies**: 
   - You may need to install **ffmpeg** for audio processing. [Here is a YouTube tutorial](#) for installation guidance.
   - **MediaPipe** was installed earlier, and its versions may differ. Please reinstall if necessary.

### Testing Instructions

1. Download the provided Jupyter Notebook and model files.
2. Ensure **ffmpeg** and all necessary dependencies are installed.
3. Follow the instructions in the notebook to test the audio translation models.

### Troubleshooting

- If there are any errors during the tests, please let me know.

### Recording Test Results

Once you have completed the tests, kindly record the results (whether the tests worked or not) in this [Google Spreadsheet](#).

### Image-Based Testing

For image-based testing, you can use the images from [my blog](#). Please note that the images for number recognition will be updated soon.

---

Thank you for your efforts in testing the system!
