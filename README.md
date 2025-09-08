# Fruit Detector Application

## Overview
The Fruit Detector Application is a real-time fruit detection system that utilizes a Convolutional Neural Network (CNN) model to identify and classify various types of fruits. The application is built using Python and leverages libraries such as Streamlit for the user interface, OpenCV for image processing, and NumPy for numerical operations.

## Features
- Real-time fruit detection using a webcam or uploaded images.
- Visualization of detection results with bounding boxes around detected fruits.
- Display of confidence levels for each detection.
- Historical tracking of detections with statistical analysis.

## Project Structure
```
fruit-detector-app
├── src
│   └── SCRIPT.py          # Main application logic for fruit detection
├── requirements.txt       # Python dependencies for the project
├── Dockerfile              # Instructions to build the Docker image
└── README.md               # Documentation for the project
```

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- Docker (for containerization)

### Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd fruit-detector-app
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Application
To run the application locally, execute the following command:
```
streamlit run src/SCRIPT.py
```

### Docker Setup
To build and run the application using Docker, follow these steps:

1. Build the Docker image:
   ```
   docker build -t fruit-detector-app .
   ```

2. Run the Docker container:
   ```
   docker run -p 8501:8501 fruit-detector-app
   ```

3. Access the application in your web browser at `http://localhost:8501`.

## Usage
- Select the method of input (webcam, upload image, or example image).
- Adjust the confidence threshold as needed.
- Click on "Process" to detect fruits in the selected image.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.