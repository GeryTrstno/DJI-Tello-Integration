"""
Input controller for keyboard and joystick input - DEBUG VERSION
Handles all user input for drone control and system commands
"""

import time
from src.config import ROTATE
from src.recording import request_manual_screenshot, toggle_recording
import src.shared_state as state


def get_joystick_input():
    """Get joystick input and update drone control variables"""
    if not state.joystick:
        return

    if state.toggle_joystick == True:
        try:
            # Read joystick input
            
            axis_lr = state.joystick.get_axis(0)  # Left-right movement
            axis_fb = state.joystick.get_axis(1)  # Forward-backward movement
            axis_yv = state.joystick.get_axis(2)  # Up-down movement  
            axis_ud = state.joystick.get_axis(3)  # Yaw rotation

            # axis_side = state.joystick.get_axis(6)
            # axis_line = state.joystick.get_axis(7)

            # if axis_side > 0.1:
            #     state.send_rc_control = False
            #     state.command_queue.put("flip r")
            #     state.send_rc_control = True
            #     print("Flip Right")
            # elif axis_side < -0.1:
            #     state.send_rc_control = False
            #     state.command_queue.put("flip l")
            #     state.send_rc_control = True
            #     print("Flip Left")
            
            # if axis_line < 0.1:
            #     state.send_rc_control = False
            #     state.command_queue.put("flip f")
            #     state.send_rc_control = True
            #     print("Flip Forward")
            # elif axis_line > -0.1:
            #     state.send_rc_control = False
            #     state.command_queue.put("flip b")
            #     state.send_rc_control = True
            #     print("Flip Back")


            # Only update velocities if joystick has significant input (deadzone)
            deadzone = 0.1
            if abs(axis_lr) > deadzone or abs(axis_fb) > deadzone or abs(axis_ud) > deadzone or abs(axis_yv) > deadzone:
                # Set velocities based on joystick input
                if state.joystick_mode == True:
                    state.left_right_velocity = int(axis_lr * state.speed)
                    state.for_back_velocity = int(-axis_fb * state.speed)
                    state.up_down_velocity = int(-axis_ud * state.speed)
                    state.yaw_velocity = int(axis_yv * ROTATE)
                elif state.joystick_mode == False:
                    state.left_right_velocity = int(axis_yv * state.speed)
                    state.for_back_velocity = int(-axis_ud * state.speed)
                    state.up_down_velocity = int(-axis_fb * state.speed)
                    state.yaw_velocity = int(axis_lr * ROTATE)
                print(f"🎮 Joystick input detected - LR:{state.left_right_velocity}, FB:{state.for_back_velocity}")

            # Handle takeoff/landing buttons
            if state.joystick.get_button(0):  # Button A - takeoff
                if not state.send_rc_control:
                    state.command_queue.put("takeoff")
                    time.sleep(0.5)

            if state.joystick.get_button(1):  # Button B - land
                if state.send_rc_control:
                    state.command_queue.put("land")
                    time.sleep(0.5)

            if state.joystick.get_button(3):  # Alternative screenshot button
                request_manual_screenshot("joystick")
                time.sleep(0.2)

            # Handle recording button (Button 4 / Y button)
            current_recording_button_state = state.joystick.get_button(4)
            if current_recording_button_state and not state.last_joystick_recording_button_state:
                toggle_recording()
                time.sleep(0.3)  # Prevent multiple triggers
            
            state.last_joystick_recording_button_state = current_recording_button_state

            if state.joystick.get_button(6):  # Button X - toggle recording
                if state.auto_screenshot_enabled == True:
                    state.auto_screenshot_enabled = False
                    time.sleep(0.3)
                    print("Auto Screenshot Disabled")
                elif state.auto_screenshot_enabled == False:
                    state.auto_screenshot_enabled = True
                    print("Auto Screenshot Enabled")
                    time.sleep(0.3)

            if state.joystick.get_button(7):
                if state.set_autonomous_behavior == False:
                    state.send_rc_control = False
                    state.set_autonomous_behavior = True
                    time.sleep(0.3)
                    print("Autonomous behavior enabled")
                elif state.set_autonomous_behavior == True:
                    state.send_rc_control = True
                    state.set_autonomous_behavior = False
                    print("Autonomous behavior disabled")
                    time.sleep(0.3)

            # Handle detection toggle button (Button 10)
            current_detection_toggle_button_state = state.joystick.get_button(8)
            if current_detection_toggle_button_state and not state.last_joystick_detection_toggle_button_state:
                state.detection_enabled = not state.detection_enabled
                status = "ENABLED" if state.detection_enabled else "DISABLED"
                print(f"🔍 Detection & Auto-Screenshot: {status}")
                time.sleep(0.3)  # Prevent multiple triggers
            
            state.last_joystick_detection_toggle_button_state = current_detection_toggle_button_state

            # Handle joystick mode toggle (Button 9)
            if state.joystick.get_button(9):  # Button 9 - toggle joystick
                if state.toggle_keyboard == True:
                    state.toggle_keyboard = False
                    print("Control Keyboard Disabled")
                    time.sleep(0.3)
                elif state.toggle_keyboard == False:
                    state.toggle_keyboard = True
                    print("Control Keyboard Enabled")
                    time.sleep(0.3)

            if state.joystick.get_button(10): # Button 10 - toggle joystick control
                if state.joystick_mode == True:
                    state.joystick_mode = False
                    print("Joystick Mode 2")
                    time.sleep(0.3)
                elif state.joystick_mode == False:
                    state.joystick_mode = True
                    print("Joystick Mode 1")
                    time.sleep(0.3)

            # Handle emergency button (Button 11)
            current_emergency_button_state = state.joystick.get_button(11)
            if current_emergency_button_state and not state.last_joystick_emergency_button_state:
                state.command_queue.put("emergency")
                print("🚨 EMERGENCY LANDING ACTIVATED!")
                time.sleep(0.5)  # Prevent multiple triggers
            
            state.last_joystick_emergency_button_state = current_emergency_button_state

        except Exception as e:
            print(f"Joystick input error: {e}")


def handle_keyboard_input(key):
    """Handle keyboard input"""
    if state.toggle_keyboard == True:
        try:
            if key == 'o':  # Screenshot
                request_manual_screenshot("keyboard")
            elif key == 'p':  # Toggle recording
                toggle_recording()
            elif key == 't':  # Takeoff
                if not state.send_rc_control:
                    state.command_queue.put("takeoff")
                    print("🚁 Takeoff command sent")
            elif key == 'q':  # Land
                if state.send_rc_control:
                    state.command_queue.put("land")
                    print("🚁 Land command sent")
            elif key == 'e':  # Emergency
                state.command_queue.put("emergency")
                print("🚨 Emergency command sent")
            elif key == 'z':  # Toggle detection
                state.detection_enabled = not state.detection_enabled
                status = "ENABLED" if state.detection_enabled else "DISABLED"
                print(f"🔍 Detection & Auto-Screenshot: {status}")
            
            elif key == 'x':
                if state.auto_screenshot_enabled == True:
                    state.auto_screenshot_enabled = False
                    print("Auto Screenshot Disabled")
                elif state.auto_screenshot_enabled == False:
                    state.auto_screenshot_enabled = True
                    print("Auto Screenshot Enabled")

            elif key == 'v':
                if state.toggle_joystick == True:
                    state.toggle_joystick = False
                    print("Control Joystick Disabled")
                elif state.toggle_joystick == False:
                    state.toggle_joystick = True
                    print("Control Joystick Enabled")
                    
            elif key == 'c':
                if state.set_autonomous_behavior == False:
                    state.set_autonomous_behavior = True
                    print("Autonomous behavior enabled")
                elif state.set_autonomous_behavior == True:
                    state.set_autonomous_behavior = False
                    print("Autonomous behavior disabled")

            elif key == 'f':
                if state.Keyboard_Mode == True:
                    state.Keyboard_Mode = False
                    print("Keyboard Mode 2")
                elif state.Keyboard_Mode == False:
                    state.Keyboard_Mode = True
                    print("Keyboard Mode 1")
            
            elif key == 'i':
                state.send_rc_control = False
                state.command_queue.put("flip f")
                state.send_rc_control = True
                print("Flip Forward")
            elif key == 'j':
                state.send_rc_control = False
                state.command_queue.put("flip l")
                state.send_rc_control = True
                print("Flip Left")
            elif key == 'k':
                state.send_rc_control = False
                state.command_queue.put("flip b")
                state.send_rc_control = True
                print("Flip Back")
            elif key == 'l':
                state.send_rc_control = False
                state.command_queue.put("flip r")
                state.send_rc_control = True
                print("Flip Right")

            elif key == 'm':
                if state.speed < 100:
                    state.speed += 5
                print(f"Speed increased: {state.speed}")

            elif key == 'n':
                if state.speed > 5:  # Ensure the speed doesn't go below 5
                    state.speed -= 5
                print(f"Speed decreased: {state.speed}")



        except Exception as e:
            print(f"Keyboard input error: {e}")


def handle_arrow_keys(keys_pressed):
    """Handle arrow key movement - DEBUG VERSION"""
    if state.toggle_keyboard == True:
        try:
            # Check if we have joystick input first
            has_joystick_input = False
            if state.joystick:
                deadzone = 0.1
                axis_lr = state.joystick.get_axis(0)
                axis_fb = state.joystick.get_axis(1)
                axis_ud = state.joystick.get_axis(3)
                axis_yv = state.joystick.get_axis(2)
                
                if (abs(axis_lr) > deadzone or abs(axis_fb) > deadzone or 
                    abs(axis_ud) > deadzone or abs(axis_yv) > deadzone):
                    has_joystick_input = True

            # Only process keyboard if no joystick input
            if not has_joystick_input:
                # Reset all velocities first
                state.for_back_velocity = 0
                state.left_right_velocity = 0
                state.up_down_velocity = 0
                state.yaw_velocity = 0

                # Debug: Check what keys are actually pressed
                pressed_keys = []
                if keys_pressed['up']: pressed_keys.append('UP')
                if keys_pressed['down']: pressed_keys.append('DOWN')
                if keys_pressed['left']: pressed_keys.append('LEFT')
                if keys_pressed['right']: pressed_keys.append('RIGHT')
                if keys_pressed['w']: pressed_keys.append('W')
                if keys_pressed['s']: pressed_keys.append('S')
                if keys_pressed['a']: pressed_keys.append('A')
                if keys_pressed['d']: pressed_keys.append('D')

                # Set velocities based on pressed keys
                if state.Keyboard_Mode == True:
                    if keys_pressed['up']:
                        state.for_back_velocity = state.speed
                    elif keys_pressed['down']:
                        state.for_back_velocity = -state.speed
                        
                    if keys_pressed['left']:
                        state.left_right_velocity = -state.speed
                    elif keys_pressed['right']:
                        state.left_right_velocity = state.speed
                        
                    if keys_pressed['w']:
                        state.up_down_velocity = state.speed
                    elif keys_pressed['s']:
                        state.up_down_velocity = -state.speed
                        
                    if keys_pressed['a']:
                        state.yaw_velocity = -ROTATE
                    elif keys_pressed['d']:
                        state.yaw_velocity = ROTATE
                else:
                    if keys_pressed['w']:
                        state.for_back_velocity = state.speed
                    elif keys_pressed['s']:
                        state.for_back_velocity = -state.speed
                        
                    if keys_pressed['a']:
                        state.left_right_velocity = -state.speed
                    elif keys_pressed['d']:
                        state.left_right_velocity = state.speed
                        
                    if keys_pressed['up']:
                        state.up_down_velocity = state.speed
                    elif keys_pressed['down']:
                        state.up_down_velocity = -state.speed
                        
                    if keys_pressed['left']:
                        state.yaw_velocity = -ROTATE
                    elif keys_pressed['right']:
                        state.yaw_velocity = ROTATE

        except Exception as e:
            print(f"Arrow key handling error: {e}")


def reset_keyboard_velocities():
    """Reset all keyboard-controlled velocities to zero"""
    # Only reset if no joystick input
    has_joystick_input = False
    if state.joystick:
        deadzone = 0.1
        axis_lr = state.joystick.get_axis(0)
        axis_fb = state.joystick.get_axis(1)
        axis_ud = state.joystick.get_axis(3)
        axis_yv = state.joystick.get_axis(2)
        
        if (abs(axis_lr) > deadzone or abs(axis_fb) > deadzone or 
            abs(axis_ud) > deadzone or abs(axis_yv) > deadzone):
            has_joystick_input = True

    if not has_joystick_input:
        state.for_back_velocity = 0
        state.left_right_velocity = 0
        state.up_down_velocity = 0
        state.yaw_velocity = 0