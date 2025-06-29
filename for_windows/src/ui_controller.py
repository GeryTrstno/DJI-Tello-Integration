"""
UI Controller for the main game loop and display
Handles pygame events, rendering, and user interface
"""

import pygame
import time
import cv2
import numpy as np

from src.config import FPS, WINDOW_WIDTH, WINDOW_HEIGHT, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT, COUNTDOWN_DURATION
from src.input_controller import get_joystick_input, handle_keyboard_input, handle_arrow_keys, reset_keyboard_velocities
import src.shared_state as state


def draw_roi_rectangle(surface):
    """Draw ROI rectangle on pygame surface only if detection is enabled"""
    try:
        if state.detection_enabled:
            roi_color = (0, 255, 0)  # Green color for ROI
            roi_rect = pygame.Rect(ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT)
            pygame.draw.rect(surface, roi_color, roi_rect, 3)  # 3 pixel thick border
    except Exception as e:
        print(f"ROI drawing error: {e}")


def add_frame_overlays(frame):
    """Add information overlays to the frame"""
    try:
        with state.data_lock:
            current_fps = state.fps
            current_battery = state.battery_level
            current_humans = state.humans_count
            current_screenshots = state.screenshot_count
            detection_status = state.detection_enabled
        
        # Add info overlays
        cv2.putText(frame, f"Battery: {current_battery}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show detection status
        detection_text = "DETECTION: ON" if detection_status else "DETECTION: OFF"
        detection_color = (0, 255, 0) if detection_status else (0, 0, 255)
        cv2.putText(frame, detection_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_color, 2)
        
        # Only show human count if detection is enabled
        if detection_status and current_humans > 0:
            cv2.putText(frame, f"Humans: {current_humans}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Screenshots: {current_screenshots}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.putText(frame, "5-THREAD MODE", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Show recording status
        if state.recording:
            recording_duration = time.time() - state.recording_start_time
            cv2.putText(frame, f"🔴 REC {recording_duration:.1f}s", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # Add red circle indicator
            cv2.circle(frame, (WINDOW_WIDTH - 30, 30), 15, (0, 0, 255), -1)

        # Show countdown if active and detection enabled
        if detection_status and state.countdown_active:
            elapsed = time.time() - state.countdown_start_time
            remaining = max(0, COUNTDOWN_DURATION - elapsed)
            cv2.putText(frame, f"Screenshot in: {remaining:.1f}s", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show autonomous behavior status
        if state.set_autonomous_behavior:
            auto_status = "AUTO MODE: ON" if detection_status else "AUTO MODE: OFF (No Detection)"
            auto_color = (255, 0, 255) if detection_status else (128, 0, 128)
            cv2.putText(frame, auto_status, (10, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, auto_color, 2)

        # Show current velocities for debugging
        cv2.putText(frame, f"LR:{state.left_right_velocity} FB:{state.for_back_velocity}", (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"UD:{state.up_down_velocity} YAW:{state.yaw_velocity}", (10, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame
    except Exception as e:
        print(f"Overlay error: {e}")
        return frame


def handle_pygame_events():
    """Handle pygame events and return whether to continue"""
    for event in pygame.event.get():
        if event.type == pygame.USEREVENT + 1:
            pass  # Update timer
        elif event.type == pygame.QUIT:
            return False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            elif event.key == pygame.K_p:   # Record
                handle_keyboard_input('p')
            elif event.key == pygame.K_o:   # Manual ScreenCapture 
                handle_keyboard_input('o')
            elif event.key == pygame.K_t:   # Takeoff
                handle_keyboard_input('t')
            elif event.key == pygame.K_q:   # Land
                handle_keyboard_input('q')
            elif event.key == pygame.K_e:   # Emergency
                handle_keyboard_input('e')
            elif event.key == pygame.K_z:   # Toggle Detection & Auto Screenshot
                handle_keyboard_input('z')
            elif event.key == pygame.K_c:   # Toggle Autonomous
                handle_keyboard_input('c')
            elif event.key == pygame.K_f:   # Change Keyboard Mode
                handle_keyboard_input('f')
            elif event.key == pygame.K_i:   # Flip Forward
                handle_keyboard_input('i')
            elif event.key == pygame.K_j:   # Flip Left
                handle_keyboard_input('j')
            elif event.key == pygame.K_k:   # Flip Back
                handle_keyboard_input('k')
            elif event.key == pygame.K_l:   # Flip Right
                handle_keyboard_input('l')
            elif event.key == pygame.K_m:   # Speed Increased 5
                handle_keyboard_input('m')
            elif event.key == pygame.K_n:   # Speed Decreased 5
                handle_keyboard_input('n')
            elif event.key == pygame.K_x:   # Toggle Auto Screenshot
                handle_keyboard_input('x')  
            elif event.key == pygame.K_v:   # Toggle Joystick
                handle_keyboard_input('v')  
    return True


def handle_continuous_keys():
    """Handle continuously pressed keys (like arrow keys) - DEBUG VERSION"""
    keys = pygame.key.get_pressed()
    
    # Check if any movement keys are pressed
    any_movement_key_pressed = (
        keys[pygame.K_UP] or keys[pygame.K_DOWN] or 
        keys[pygame.K_LEFT] or keys[pygame.K_RIGHT] or
        keys[pygame.K_w] or keys[pygame.K_s] or 
        keys[pygame.K_a] or keys[pygame.K_d]
    )
    
    if any_movement_key_pressed:
        # Make sure RC control is enabled for keyboard movement
        if not state.send_rc_control:
            state.send_rc_control = True
            print("🎮 RC Control: AUTO-ENABLED for keyboard movement")
        
        keys_pressed = {
            'up': keys[pygame.K_UP],
            'down': keys[pygame.K_DOWN],
            'left': keys[pygame.K_LEFT],
            'right': keys[pygame.K_RIGHT],
            'w': keys[pygame.K_w],
            's': keys[pygame.K_s],
            'a': keys[pygame.K_a],
            'd': keys[pygame.K_d]
        }
        
        
        handle_arrow_keys(keys_pressed)
    else:
        # No movement keys pressed, reset velocities
        reset_keyboard_velocities()


def main_loop():
    """Main UI loop - handles display and input"""
    should_stop = False
    
    try:
        while not should_stop:
            # Handle pygame events
            if not handle_pygame_events():
                should_stop = True
                break
            
            # Handle continuous key presses FIRST (before joystick)
            handle_continuous_keys()
            
            # Get joystick input (this will override keyboard if joystick is active)
            get_joystick_input()

            # Clear screen
            state.screen.fill([0, 0, 0])

            # Get latest processed frame
            display_frame = None
            with state.data_lock:
                if state.current_processed_frame is not None:
                    display_frame = state.current_processed_frame.copy()
            
            if display_frame is not None:
                # Add overlays
                display_frame = add_frame_overlays(display_frame)

                # Convert frame for pygame display
                frame_rgb = np.rot90(display_frame)
                frame_rgb = np.flipud(frame_rgb)

                frame_surface = pygame.surfarray.make_surface(frame_rgb)
                state.screen.blit(frame_surface, (0, 0))

                # Draw ROI rectangle (only if detection enabled)
                draw_roi_rectangle(state.screen)

            pygame.display.update()
            time.sleep(1 / FPS)

    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    except Exception as e:
        print(f"Main loop error: {e}")
    finally:
        print(f"UI loop ended. Total screenshots taken: {state.screenshot_count}")