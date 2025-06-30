"""
Thread management for all background processes - DEBUG VERSION
Contains all worker threads for video, control, detection, etc.
"""

import threading
import time
import cv2
import queue

from src.config import (
    VIDEO_THREAD_SLEEP, DETECTION_THREAD_SLEEP, CONTROL_THREAD_SLEEP,
    RECORDING_THREAD_SLEEP, AUTONOMOUS_THREAD_SLEEP, BATTERY_CHECK_INTERVAL,
    WINDOW_WIDTH, WINDOW_HEIGHT
)
from src.detection import process_human_detection, detect_red_color, detect_red_in_roi
from src.recording import handle_auto_screenshot, save_screenshot, process_recording_frame
import src.shared_state as state


def video_stream_thread():
    """Thread 1: Handle video capture, processing, screenshot operations, and recording"""
    print("Video stream thread started")
    
    try:
        frame_read = state.tello.get_frame_read()
        
        while state.running:
            try:
                if frame_read.stopped:
                    break
                    
                frame = frame_read.frame
                if frame is not None:
                    # Resize frame once here
                    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
                    
                    # Process human detection only if detection is enabled
                    if state.detection_enabled:
                        output_frame, detected, count = process_human_detection(frame)
                        # Handle auto screenshot logic
                        if state.auto_screenshot_enabled:
                            handle_auto_screenshot(output_frame, detected, count)
                    else:
                        # If detection disabled, use original frame without processing
                        output_frame = frame.copy()
                        detected = False
                        count = 0
                    
                    # Process any screenshot requests from queue
                    while not state.screenshot_queue.empty():
                        try:
                            screenshot_data = state.screenshot_queue.get_nowait()
                            frame_to_save, humans_count_param, source = screenshot_data
                            save_screenshot(frame_to_save, humans_count_param, source)
                        except queue.Empty:
                            break
                    
                    # Handle recording
                    process_recording_frame(output_frame)
                    
                    # Update shared data
                    with state.data_lock:
                        state.current_frame = frame.copy()
                        state.current_processed_frame = output_frame.copy()
                        state.human_detected = detected
                        state.humans_count = count
                        
                        # Calculate FPS
                        current_time = time.time()
                        state.frame_times.append(current_time)
                        if len(state.frame_times) > 1:
                            time_diff = state.frame_times[-1] - state.frame_times[0]
                            state.fps = len(state.frame_times) / time_diff if time_diff > 0 else 0
                
                # Control processing speed
                time.sleep(VIDEO_THREAD_SLEEP)
                
            except Exception as e:
                print(f"Video stream error: {e}")
                time.sleep(0.1)
        
    except Exception as e:
        print(f"Critical video stream error: {e}")
    
    print("Video stream thread ended")


def drone_control_thread():
    """Thread 2: Handle drone control commands, battery monitoring, and telemetry updates - DEBUG VERSION"""
    print("Drone control thread started")
    last_battery_check = time.time()
    last_telemetry_update = time.time()
    last_rc_debug = time.time()
    telemetry_update_interval = 1.0  # Update telemetry every second
    
    while state.running:
        try:
            # Process any queued commands
            while not state.command_queue.empty():
                try:
                    command = state.command_queue.get_nowait()
                    execute_drone_command(command)
                except queue.Empty:
                    break
            
            # Send RC control if active
            if state.send_rc_control:
                current_time = time.time()
                
                # Debug RC commands every 2 seconds
                if current_time - last_rc_debug >= 2.0:
                    if any([state.left_right_velocity, state.for_back_velocity, 
                           state.up_down_velocity, state.yaw_velocity]):
                        print(f"🚁 Sending RC: LR={state.left_right_velocity}, FB={state.for_back_velocity}, UD={state.up_down_velocity}, YAW={state.yaw_velocity}")
                    last_rc_debug = current_time
                
                try:
                    state.tello.send_rc_control(
                        state.left_right_velocity, 
                        state.for_back_velocity,
                        state.up_down_velocity, 
                        state.yaw_velocity
                    )
                except Exception as rc_error:
                    print(f"RC command error: {rc_error}")
            
            current_time = time.time()
            
            # Update battery level periodically
            if current_time - last_battery_check >= BATTERY_CHECK_INTERVAL:
                try:
                    with state.data_lock:
                        state.battery_level = state.tello.get_battery()
                    last_battery_check = current_time
                except Exception as e:
                    print(f"Battery check error: {e}")
            
            # Update telemetry data periodically
            if current_time - last_telemetry_update >= telemetry_update_interval:
                try:
                    update_telemetry_data()
                    last_telemetry_update = current_time
                except Exception as e:
                    print(f"Telemetry update error: {e}")
            
            time.sleep(CONTROL_THREAD_SLEEP)
            
        except Exception as e:
            print(f"Drone control error: {e}")
            time.sleep(0.1)
    
    print("Drone control thread ended")


def update_telemetry_data():
    """Update all telemetry variables from Tello state"""
    try:
        # Get state string from Tello
        state_string = state.tello.get_current_state()
        
        # Parse state string and update variables
        if state_string:
            state_data = {}
            for item in state_string.split(';'):
                if ':' in item:
                    key, value = item.split(':', 1)
                    try:
                        state_data[key] = float(value)
                    except ValueError:
                        state_data[key] = value
            
            # Update telemetry variables with thread safety
            with state.data_lock:
                # Environmental sensors
                state.temp = state_data.get('templ', state.temp)  # Temperature (low)
                if 'temph' in state_data:  # Use high temperature if available
                    state.temp = state_data.get('temph', state.temp)
                
                state.baro = state_data.get('baro', state.baro)    # Barometer
                state.height = state_data.get('h', state.height)   # Height
                state.tof = state_data.get('tof', state.tof)       # Time of flight
                
                # Attitude angles
                state.pitch = state_data.get('pitch', state.pitch)
                state.roll = state_data.get('roll', state.roll)
                state.yaw = state_data.get('yaw', state.yaw)
                
                # Velocities
                state.vgx = state_data.get('vgx', state.vgx)
                state.vgy = state_data.get('vgy', state.vgy)
                state.vgz = state_data.get('vgz', state.vgz)
                
                # Accelerations
                state.agx = state_data.get('agx', state.agx)
                state.agy = state_data.get('agy', state.agy)
                state.agz = state_data.get('agz', state.agz)
                
    except Exception as e:
        print(f"Error updating telemetry: {e}")


def detection_thread():
    """Thread 3: Handle red color detection"""
    print("Detection thread started")
    
    while state.running:
        try:
            # Only run detection if enabled
            # Get current frame safely
            with state.data_lock:
                if state.current_processed_frame is not None:
                    frame_copy = state.current_processed_frame.copy()
                else:
                    frame_copy = None
            
            if frame_copy is not None:
                frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                # Perform detection
                mask, result = detect_red_color(frame_copy)
                red_in_roi, roi_mask, full_roi_mask, pixel_count = detect_red_in_roi(frame_copy)
                
                # Update detection results safely
                with state.data_lock:
                    state.current_detection = {
                        'red_detected': red_in_roi,
                        'mask': mask,
                        'result': result,
                        'roi_mask': roi_mask,
                        'full_roi_mask': full_roi_mask,
                        'pixel_count': pixel_count
                    }
            else:
                # Clear detection results when disabled
                with state.data_lock:
                    state.current_detection = {
                        'red_detected': False,
                        'mask': None,
                        'result': None,
                        'roi_mask': None,
                        'full_roi_mask': None,
                        'pixel_count': 0
                    }

            time.sleep(DETECTION_THREAD_SLEEP)
            
        except Exception as e:
            print(f"Detection thread error: {e}")
            time.sleep(0.1)
    
    print("Detection thread ended")


def autonomous_behavior_thread():
    """Thread 4: Handle autonomous behavior based on detection results"""
    print("Autonomous behavior thread started")
    
    while state.running:
        try:
            # Only run autonomous behavior if detection is enabled and autonomous mode is on
            if state.set_autonomous_behavior and state.detection_enabled and state.current_detection is not None:
                # Get current detection results safely
                with state.data_lock:
                    if state.current_detection:
                        red_detected = state.current_detection.get('red_detected', False)
                        pixel_count = state.current_detection.get('pixel_count', 0)
                    else:
                        red_detected = False
                        pixel_count = 0
                
                # Autonomous behavior logic
                if red_detected:
                    print("🔴 Red detected in ROI! Moving towards target...")
                    state.tello.move_back(70)
                    time.sleep(2)
                    state.tello.rotate_clockwise(90)
                    time.sleep(2)
                else:
                    print("⚪ No red in ROI. Searching...")
                    state.tello.move_forward(20)
                    time.sleep(1)
            else:
                time.sleep(AUTONOMOUS_THREAD_SLEEP)
                
        except Exception as e:
            print(f"Autonomous behavior error: {e}")
            time.sleep(0.5)
    
    print("Autonomous behavior thread ended")


def recording_thread_func():
    """Thread 5: Dedicated thread for smooth recording"""
    print("Recording thread started")
    
    while state.running:
        try:
            if state.recording and state.video_writer and state.video_writer.isOpened():
                # Process buffered frames for smooth recording
                frames_written = 0
                while not state.recording_frame_buffer.empty() and frames_written < 5:
                    try:
                        frame_bgr = state.recording_frame_buffer.get_nowait()
                        state.video_writer.write(frame_bgr)
                        frames_written += 1
                    except queue.Empty:
                        break
                
                time.sleep(RECORDING_THREAD_SLEEP)
            else:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Recording thread error: {e}")
            time.sleep(0.1)
    
    print("Recording thread ended")


def execute_drone_command(command):
    """Execute drone commands safely"""
    try:
        print(f"🎯 Executing command: {command}")
        if command == "takeoff":
            state.tello.takeoff()
            state.send_rc_control = True
            print("✅ Takeoff completed, RC control enabled")
        elif command == "land":
            state.tello.land()
            state.send_rc_control = False
            print("✅ Landing completed, RC control disabled")
        elif command == "emergency":
            state.tello.emergency()
            state.send_rc_control = False
            print("🚨 Emergency landing executed")
        elif command == "flip f":
            state.tello.flip_forward()
        elif command == "flip b":
            state.tello.flip_back()
        elif command == "flip l":
            state.tello.flip_left()
        elif command == "flip r":
            state.tello.flip_right()
    except Exception as e:
        print(f"Command execution error: {e}")


def start_all_threads():
    """Start all worker threads"""
    print("Starting all threads...")
    
    # Create and start all threads
    threads_config = [
        ("Video Stream", video_stream_thread),
        ("Drone Control", drone_control_thread),
        ("Detection", detection_thread),
        ("Autonomous Behavior", autonomous_behavior_thread),
        ("Recording", recording_thread_func)
    ]
    
    for name, target_func in threads_config:
        thread = threading.Thread(target=target_func, daemon=True, name=name)
        thread.start()
        state.threads.append(thread)
    
    print(f"Started {len(state.threads)} worker threads")


def stop_all_threads():
    """Stop all worker threads"""
    print("Stopping threads...")
    state.running = False
    
    # Stop recording if active
    from src.recording import stop_recording
    if state.recording:
        stop_recording()
    
    # Wait for threads to finish
    for thread in state.threads:
        if thread.is_alive():
            thread.join(timeout=2)
    
    # Cleanup systems
    from src.initialization import cleanup_systems
    cleanup_systems()
    
    print("All threads stopped")