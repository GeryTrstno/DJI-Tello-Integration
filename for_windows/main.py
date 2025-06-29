#!/usr/bin/env python3
"""
Main entry point for Tello Drone Control System
5-Thread Architecture with Smart Features

Run this file to start the application:
python main.py
"""

import time
from src.initialization import initialize_all_systems
from src.drone_threads import start_all_threads, stop_all_threads
from src.ui_controller import main_loop
from src.config import THREAD_COUNT


def main():
    """Main function - Entry point of the application"""
    print(f"{THREAD_COUNT}-Thread Tello Drone Control System with Smooth Recording")
    print("Architecture:")
    print("- Thread 1: Video stream and human detection")
    print("- Thread 2: Drone control commands and battery monitoring") 
    print("- Thread 3: Red color detection")
    print("- Thread 4: Autonomous behavior")
    print("- Thread 5: Dedicated smooth recording")
    print("Features:")
    print("- Smart auto screenshot with 3-second countdown")
    print("- Manual screenshot with 'P' key or joystick buttons")
    print("- Time-based video recording (consistent speed)")
    print("- Human detection with pose and hand tracking")
    print("- Screenshots saved in 'screenshots' folder")
    print("- Recordings saved in 'recordings' folder")
    print("Controls:")
    print("- Keyboard: Arrow keys=move, W/S=up/down, A/D=rotate, T=takeoff, L=land")
    print("- P=screenshot, R=toggle recording, O=toggle detection, E=emergency")
    print("- Joystick: Move drone, A=takeoff, B=land, X/Y=screenshot, Y=recording")
    print("- Button 10=toggle detection, Button 11=emergency")
    print("- ESC or close window to quit")
    
    try:
        # Initialize all systems
        if not initialize_all_systems():
            print("Failed to initialize systems. Exiting...")
            return
        
        # Start all worker threads
        start_all_threads()
        
        # Wait a moment for threads to initialize
        time.sleep(2)
        
        # Run main UI loop
        main_loop()
        
    except KeyboardInterrupt:
        print("Keyboard interrupt detected")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print("Shutting down...")
        stop_all_threads()


if __name__ == '__main__':
    main()