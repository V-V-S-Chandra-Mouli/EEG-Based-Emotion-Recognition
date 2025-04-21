# EEG-Based Emotion Recognition using Deep Learning Techniques

This project focuses on recognizing human emotions using EEG (Electroencephalogram) signals through the application of deep learning methods, specifically Convolutional Neural Networks (CNNs). EEG provides a direct, real-time insight into brain activity and is a valuable tool for emotion recognition, especially for applications in brain-computer interfaces, mental health monitoring, and adaptive systems.

The EEG data used in this project is sourced from well-known datasets like DEAP or SEED, which include recordings from multiple subjects exposed to emotional stimuli such as videos or music. The raw EEG signals undergo preprocessing including bandpass filtering, normalization, and segmentation. In certain cases, these signals are converted into spectrograms or 2D feature maps to be compatible with CNN models.

The proposed CNN model automatically learns spatial and temporal features from the EEG inputs, removing the need for manual feature engineering. The architecture includes multiple convolutional layers, pooling layers, and fully connected layers that culminate in emotion classification output. Emotion classes may include binary states (e.g., high/low valence) or multi-class categories (e.g., happy, sad, relaxed, etc.).

The model is trained using a categorical cross-entropy loss function and optimized using the Adam optimizer. Performance is evaluated based on metrics such as accuracy, precision, recall, and confusion matrix. The CNN demonstrated superior performance compared to traditional machine learning models, showing deep learning's effectiveness in capturing complex EEG patterns.

Technologies used in this project include Python, TensorFlow (or PyTorch), NumPy, and Matplotlib. The codebase is structured into directories for raw and preprocessed data, model training scripts, utility functions, and result visualization. A sample training script (`train_model.py`) and environment file (`requirements.txt`) are provided for reproducibility.

Planned enhancements for the project include the integration of LSTM or Transformer architectures to better capture temporal dependencies, subject-adaptive models for personalized emotion recognition, and a potential user interface for real-time emotion feedback.

This project was carried out under the mentorship of Dr. Mohammed Farukh Hashmi at the Networks Laboratory, Department of Electronics and Communication Engineering, NIT Warangal. The work demonstrates the promising future of deep learning in EEG-based emotion recognition and its application in intelligent systems.

