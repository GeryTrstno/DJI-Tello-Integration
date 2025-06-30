"""
Tello Drone Control System Package
5-Thread Architecture with Smart Detection and Recording

Package structure:
- config.py: Configuration settings
- shared_state.py: Global state management  
- initialization.py: System initialization
- detection.py: AI detection algorithms
- recording.py: Recording and screenshot system
- input_controller.py: Input handling
- drone_threads.py: Thread management
- ui_controller.py: User interface
"""

__version__ = "1.0.0"
__author__ = "Tello Drone Control Team"
__description__ = "Multi-threaded Tello drone control system with AI detection"

# Package metadata
PACKAGE_NAME = "tello_drone_control"
THREAD_COUNT = 5
FEATURES = [
    "Multi-threaded architecture",
    "Human detection with YOLO",
    "Body part tracking with MediaPipe", 
    "Smart auto-screenshot system",
    "Time-based video recording",
    "Red color detection",
    "Autonomous behavior",
    "Joystick and keyboard control"
]