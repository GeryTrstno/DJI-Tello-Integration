"""
Configuration settings for Tello Drone Control System
All constants and settings are defined here
"""

# Display settings
FPS = 120
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

# ROI parameters (Region of Interest)
ROI_X = WINDOW_WIDTH // 2
ROI_Y = 10
ROI_WIDTH = WINDOW_WIDTH // 2
ROI_HEIGHT = WINDOW_HEIGHT // 2

# Thread count
THREAD_COUNT = 5

# Drone control settings
SPEED = 50
ROTATE = 80

# Screenshot settings
SCREENSHOT_INTERVAL = 3
COUNTDOWN_DURATION = 3.0

# Recording settings
RECORDING_FPS = 30
RECORDING_BUFFER_SIZE = 100

# Directories
SCREENSHOTS_DIR = "screenshots"
RECORDINGS_DIR = "recordings"

# Detection settings
RED_DETECTION_THRESHOLD = 10000
YOLO_CONFIDENCE_THRESHOLD = 0.5
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.3
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.3

# Color detection ranges (HSV)
LOWER_RED1 = [0, 120, 130]
UPPER_RED1 = [30, 255, 255]
LOWER_RED2 = [170, 150, 130]
UPPER_RED2 = [255, 255, 255]

# Performance monitoring
FRAME_TIME_BUFFER_SIZE = 30
BATTERY_CHECK_INTERVAL = 10  # seconds

# Thread timing
VIDEO_THREAD_SLEEP = 0.01
DETECTION_THREAD_SLEEP = 0.03
CONTROL_THREAD_SLEEP = 1/30
RECORDING_THREAD_SLEEP = 0.01
AUTONOMOUS_THREAD_SLEEP = 0.1