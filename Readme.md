# Sign Language to Text Conversion using Transformers

<!-- ![Project Demo](demo.gif) *Replace with actual demo GIF if available* -->

## Project Overview

This project converts dynamic sign language gestures into text using transformer-based deep learning. It captures sequences of body and hand movements through a camera, processes them using MediaPipe for landmark detection, and employs a transformer model to interpret the gestures in real-time.

## Key Features

- **Dynamic Gesture Recognition**: Captures sign language gestures across multiple frames
- **Transformer Architecture**: Uses self-attention mechanisms to understand temporal patterns
- **Real-time Processing**: Works with live camera feed for immediate feedback
- **Data Collection Tool**: Includes a GUI application for creating custom datasets
- **Normalized Coordinates**: Handles varying distances and orientations

## File Structure

```
Sign-Language-To-Text-Conversion/
├── dataset_creation_tool.py     # GUI for recording and saving gesture data
├── transformerTrain.py          # Script for training the transformer model
├── transformerTestRealtime.py   # Real-time gesture recognition script
├── requirements.txt             # Python dependencies
├── gesture_data/                # Directory for saved gesture datasets
│   └── [gesture_label]/         # Subdirectories for each gesture class
│       └── [session_id]/        # Individual recording sessions containing:
│           ├── original.avi               # Raw video
│           ├── labeled.avi                # Video with landmarks
│           ├── stickman_with_coords.avi   # Simplified stick figure
│           ├── stickman_clean.avi         # Clean stick figure
│           ├── original_coords.npy        # Original coordinates
│           └── normalized_coords.npy      # Normalized coordinates
└── README.md                    # This file
```

## Dataset Creation Tool

The `dataset_creation_tool.py` provides a GUI interface for recording and saving gesture data:

### How to Use:
1. Run the script: `python dataset_creation_tool.py`
2. Click "Start Recording" to begin capturing frames
3. Perform your gesture in front of the camera
4. Click "Stop Recording" when finished
5. Enter a label for your gesture when prompted
6. Click "Save Dataset" to store the recording

### Data Format:
Each recording session saves:
- Original and labeled videos
- Normalized coordinate data (3D positions relative to body center)
- Stick figure visualizations
- Numpy arrays containing:
  - Original coordinates with visibility scores
  - Normalized coordinates (body-centered and scaled)
  - Body-only coordinates (without hands)

## Model Training

The `transformerTrain.py` script handles model training:

### Key Components:
- Transformer architecture with self-attention
- Positional encoding for temporal sequences
- Custom data loader for gesture sequences
- Training and validation loops

### Training Process:
1. Organize your data in `gesture_data/[labels]` folders
2. Configure hyperparameters in the script
3. Run: `python transformerTrain.py`
4. Model checkpoints will be saved periodically

## Real-time Testing

The `transformerTestRealtime.py` script provides live gesture recognition:

### Features:
- Real-time camera processing
- On-the-fly landmark detection
- Sequence prediction using trained transformer
- Visual feedback of recognized gestures

### Usage:
1. Train your model first using `transformerTrain.py`
2. Run: `python transformerTestRealtime.py`
3. Perform gestures in front of the camera
4. View real-time predictions

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

Key dependencies (see requirements.txt for complete list):
- Python 3.7+
- TensorFlow/PyTorch (depending on implementation)
- MediaPipe
- OpenCV
- NumPy
- Tkinter

## Future Improvements

- Expand vocabulary of recognized signs
- Improve robustness to lighting and background variations
- Add user customization for new gestures
- Optimize for mobile deployment
- Incorporate facial expression recognition

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.