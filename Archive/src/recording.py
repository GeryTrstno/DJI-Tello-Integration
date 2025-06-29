"""
Recording and screenshot management system
Handles video recording and screenshot capture
"""

import cv2
import os
import time
from datetime import datetime

from src.config import WINDOW_WIDTH, WINDOW_HEIGHT, RECORDINGS_DIR, SCREENSHOTS_DIR
import src.shared_state as state


def start_recording():
    """Start video recording"""
    try:
        if not state.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            state.current_recording_file = f"tello_flight_{timestamp}.mp4"
            filepath = os.path.join(RECORDINGS_DIR, state.current_recording_file)
            
            # Use consistent 30 FPS with time-based control
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            state.video_writer = cv2.VideoWriter(filepath, fourcc, 30.0, (WINDOW_WIDTH, WINDOW_HEIGHT))
            
            if state.video_writer.isOpened():
                state.recording = True
                state.recording_start_time = time.time()
                state.last_recording_frame_time = time.time()
                print(f"🔴 Recording started: {state.current_recording_file}")
                print("📝 Using time-based frame control for consistent speed")
                return True
            else:
                print("❌ Failed to start recording - could not open video writer")
                return False
    except Exception as e:
        print(f"Error starting recording: {e}")
        return False


def stop_recording():
    """Stop video recording"""
    try:
        if state.recording and state.video_writer:
            state.recording = False
            state.video_writer.release()
            state.video_writer = None
            
            recording_duration = time.time() - state.recording_start_time
            print(f"⏹️ Recording stopped: {state.current_recording_file}")
            print(f"Duration: {recording_duration:.1f} seconds")
            return True
    except Exception as e:
        print(f"Error stopping recording: {e}")
        return False


def toggle_recording():
    """Toggle recording on/off"""
    if state.recording:
        stop_recording()
    else:
        start_recording()


def save_screenshot(frame, humans_count_param, source="auto"):
    """Save screenshot with timestamp and human count"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_prefix = "manual" if source in ["joystick", "keyboard"] else "auto"
        filename = f"{source_prefix}_human_detected_{timestamp}_{humans_count_param}persons_{state.screenshot_count:04d}.jpg"
        filepath = os.path.join(SCREENSHOTS_DIR, filename)
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for saving

        # Save the frame
        success = cv2.imwrite(filepath, frame_bgr)
        
        if success:
            state.screenshot_count += 1
            print(f"Screenshot saved ({source}): {filename}")
            return True
        else:
            print(f"Failed to save screenshot: {filename}")
            return False
    
    except Exception as e:
        print(f"Error saving screenshot: {e}")
        return False


def handle_auto_screenshot(output_frame, human_detected_now, humans_count_now):
    """Handle auto screenshot countdown logic"""
    try:
        current_time = time.time()
        
        if human_detected_now and humans_count_now >= 1:
            # Human detected
            if not state.last_human_detected and not state.countdown_active:
                # First time detecting human, start countdown
                state.countdown_active = True
                state.countdown_start_time = current_time
                print(f"Human detected! Starting 3-second countdown...")
            
            # If countdown is active
            if state.countdown_active:
                from src.config import COUNTDOWN_DURATION
                elapsed_time = current_time - state.countdown_start_time
                
                if elapsed_time >= COUNTDOWN_DURATION:
                    # Countdown finished, take screenshot
                    save_screenshot(output_frame.copy(), humans_count_now, "auto")
                    state.countdown_active = False
                    print("Countdown completed! Screenshot taken.")
        else:
            # No human detected
            if state.countdown_active:
                # Cancel countdown if human disappears
                state.countdown_active = False
                print("Human detection lost! Countdown cancelled.")
        
        # Update human detection status for next frame
        state.last_human_detected = human_detected_now
        
    except Exception as e:
        print(f"Error in auto screenshot handler: {e}")


def request_manual_screenshot(source):
    """Request a manual screenshot"""
    try:
        with state.data_lock:
            if state.current_processed_frame is not None:
                state.screenshot_queue.put((
                    state.current_processed_frame.copy(), 
                    state.humans_count, 
                    source
                ))
                print(f"Manual screenshot requested ({source})")
    except Exception as e:
        print(f"Screenshot request error: {e}")


def process_recording_frame(output_frame):
    """Process frame for recording with time-based control"""
    try:
        if state.recording and state.video_writer and state.video_writer.isOpened():
            current_time = time.time()
            time_since_last_record = current_time - state.last_recording_frame_time
            
            # Save frame exactly every 1/30 second (33.33ms) for consistent 30 FPS
            if time_since_last_record >= (1.0 / 30.0):
                frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                state.video_writer.write(frame_bgr)
                state.last_recording_frame_time = current_time
                
                # Add frame to buffer for recording thread
                try:
                    if not state.recording_frame_buffer.full():
                        state.recording_frame_buffer.put_nowait(frame_bgr.copy())
                except:
                    pass  # Skip if buffer is full
    except Exception as e:
        print(f"Recording frame error: {e}")