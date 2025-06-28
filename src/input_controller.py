"""
Input controller for keyboard and joystick input
Handles all user input for drone control and system commands
"""

import time
from src.config import JOYSTICK_SPEED, JOYSTICK_ROTATE
from src.recording import request_manual_screenshot, toggle_recording
import src.shared_state as state


def get_joystick_input():
    """Get joystick input and update drone control variables"""
    if not state.joystick:
        return

    try:
        # Read joystick input
        axis_lr = state.joystick.get_axis(0)  # Left-right movement
        axis_fb = state.joystick.get_axis(1)  # Forward-backward movement
        axis_yv = state.joystick.get_axis(2)  # Up-down movement  
        axis_ud = state.joystick.get_axis(3)  # Yaw rotation

        # Set velocities based on joystick input
        state.left_right_velocity = int(axis_lr * JOYSTICK_SPEED)
        state.for_back_velocity = int(-axis_fb * JOYSTICK_SPEED)
        state.up_down_velocity = int(-axis_ud * JOYSTICK_SPEED)
        state.yaw_velocity = int(axis_yv * JOYSTICK_ROTATE)

        # Handle takeoff/landing buttons
        if state.joystick.get_button(0):  # Button A - takeoff
            if not state.send_rc_control:
                state.command_queue.put("takeoff")
                time.sleep(0.5)

        if state.joystick.get_button(1):  # Button B - land
            if state.send_rc_control:
                state.command_queue.put("land")
                time.sleep(0.5)

        # Handle screenshot buttons
        current_screenshot_button_state = state.joystick.get_button(2)
        if current_screenshot_button_state and not state.last_joystick_screenshot_button_state:
            request_manual_screenshot("joystick")
        
        state.last_joystick_screenshot_button_state = current_screenshot_button_state

        if state.joystick.get_button(3):  # Alternative screenshot button
            request_manual_screenshot("joystick")
            time.sleep(0.2)

        # Handle recording button (Button 4 / Y button)
        current_recording_button_state = state.joystick.get_button(4)
        if current_recording_button_state and not state.last_joystick_recording_button_state:
            toggle_recording()
            time.sleep(0.3)  # Prevent multiple triggers
        
        state.last_joystick_recording_button_state = current_recording_button_state
        
        # Handle autonomous behavior buttons
        if state.joystick.get_button(6):
            state.set_autonomous_behavior = True
            print("Autonomous behavior enabled")

        elif state.joystick.get_button(7):
            state.set_autonomous_behavior = False
            print("Autonomous behavior disabled")
        
        # Handle RC control enable/disable
        if state.joystick.get_button(8):  # Left shoulder - enable RC control
            state.send_rc_control = True
            print("🎮 RC Control: ENABLED")

        if state.joystick.get_button(9):  # Right shoulder - disable RC control
            state.send_rc_control = False
            state.left_right_velocity = 0
            state.for_back_velocity = 0
            state.up_down_velocity = 0
            state.yaw_velocity = 0
            print("🎮 RC Control: DISABLED")

    except Exception as e:
        print(f"Joystick input error: {e}")


def handle_keyboard_input(key):
    """Handle keyboard input"""
    try:
        if key == 'p':  # Screenshot
            request_manual_screenshot("keyboard")
        elif key == 'r':  # Toggle recording
            toggle_recording()
        elif key == 't':  # Takeoff
            if not state.send_rc_control:
                state.command_queue.put("takeoff")
        elif key == 'l':  # Land
            if state.send_rc_control:
                state.command_queue.put("land")
        elif key == 'e':  # Emergency
            state.command_queue.put("emergency")
        elif key == 'a':  # Enable autonomous behavior
            state.set_autonomous_behavior = True
            print("Autonomous behavior enabled")
        elif key == 's':  # Disable autonomous behavior
            state.set_autonomous_behavior = False
            print("Autonomous behavior disabled")
        
    except Exception as e:
        print(f"Keyboard input error: {e}")


def handle_arrow_keys(keys_pressed):
    """Handle arrow key movement"""
    try:
        from src.config import DRONE_SPEED
        
        # Reset velocities
        state.left_right_velocity = 0
        state.for_back_velocity = 0
        state.up_down_velocity = 0
        state.yaw_velocity = 0
        
        # Set velocities based on pressed keys
        if keys_pressed['up']:
            state.for_back_velocity = DRONE_SPEED
        elif keys_pressed['down']:
            state.for_back_velocity = -DRONE_SPEED
            
        if keys_pressed['left']:
            state.left_right_velocity = -DRONE_SPEED
        elif keys_pressed['right']:
            state.left_right_velocity = DRONE_SPEED
            
        if keys_pressed['w']:
            state.up_down_velocity = DRONE_SPEED
        elif keys_pressed['s']:
            state.up_down_velocity = -DRONE_SPEED
            
        if keys_pressed['a']:
            state.yaw_velocity = -DRONE_SPEED
        elif keys_pressed['d']:
            state.yaw_velocity = DRONE_SPEED
            
    except Exception as e:
        print(f"Arrow key handling error: {e}")