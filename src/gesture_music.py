import cv2
import numpy as np
import math
import os
import sounddevice as sd
from collections import deque

try:
    from mediapipe import Image, ImageFormat
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_TASKS_AVAILABLE = True
except (ImportError, AttributeError):
    MEDIAPIPE_TASKS_AVAILABLE = False


class MusicSynthesizer:
    """Generates and plays musical tones based on gestures"""
    
    def __init__(self):
        self.sample_rate = 44100
        self.current_note = None
        self.current_frequency = 0
        self.phase = 0
        self.stream = None
        
        # Musical note frequencies (4th octave)
        self.notes = {
            'C': 261.63,
            'D': 293.66,
            'E': 329.63,
            'F': 349.23,
            'G': 392.00,
            'A': 440.00,
            'B': 493.88,
            'SILENCE': 0
        }
        
        # Gesture to note mapping
        self.gesture_map = {
            '1️⃣ One Finger': 'C',
            '2️⃣ Two Fingers': 'D',
            '3️⃣ Three Fingers': 'E',
            '4️⃣ Four Fingers': 'F',
            '✋ Open Palm': 'A',
            '👍 Thumbs Up': 'B',
            '✊ Closed Fist': 'SILENCE'
        }
        
        # Start the audio stream
        self.start_stream()
    
    def audio_callback(self, outdata, frames, time, status):
        """Callback function for continuous audio generation"""
        if status:
            print(f"Audio status: {status}")
        
        if self.current_frequency == 0:
            # Silence
            outdata[:] = 0
        else:
            # Generate sine wave samples
            t = (np.arange(frames) + self.phase) / self.sample_rate
            outdata[:] = (0.3 * np.sin(2 * np.pi * self.current_frequency * t)).reshape(-1, 1).astype(np.float32)
            self.phase += frames
    
    def start_stream(self):
        """Start the audio output stream"""
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=2048
            )
            self.stream.start()
        except Exception as e:
            print(f"Error starting audio stream: {e}")
    
    def play_note(self, note_name):
        """Switch to playing a different note"""
        if note_name == self.current_note:
            return  # Already playing this note
        
        self.current_note = note_name
        
        if note_name in self.notes:
            self.current_frequency = self.notes[note_name]
        else:
            self.current_frequency = 0
    
    def get_note_for_gesture(self, gesture_name):
        """Get the musical note for a gesture"""
        for key in self.gesture_map:
            if key in gesture_name or gesture_name in key:
                return self.gesture_map[key]
        return None
    
    def cleanup(self):
        """Stop all playback and cleanup"""
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception as e:
            print(f"Cleanup error: {e}")


class HandGestureDetector:
    """Detects hand gestures using MediaPipe's hand landmark model"""
    
    def __init__(self):
        self.hand_detector_ready = False
        self.landmarker = None
        self.frame_index = 0
        
        if MEDIAPIPE_TASKS_AVAILABLE:
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_dir, 'hand_landmarker.task')
                
                if not os.path.exists(model_path):
                    print(f"Error: Model file not found at {model_path}")
                    return
                
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=1,  # Only track one hand for music
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                
                self.landmarker = vision.HandLandmarker.create_from_options(options)
                self.hand_detector_ready = True
                print("✓ Hand landmark model loaded!")
                
            except Exception as e:
                print(f"Error loading model: {e}")
    
    def distance(self, p1, p2):
        if p1 is None or p2 is None:
            return 0
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def is_finger_extended(self, landmarks, finger_tip, finger_pip):
        if landmarks[finger_tip] is None or landmarks[finger_pip] is None:
            return False
        return landmarks[finger_tip].y < landmarks[finger_pip].y
    
    def detect_gesture(self, landmarks):
        """Detect hand gesture from 21 landmark points"""
        if not landmarks or len(landmarks) != 21:
            return "No hand", 0
        
        lm = landmarks
        
        # Landmark indices
        THUMB_TIP, THUMB_PIP = 4, 3
        INDEX_TIP, INDEX_PIP = 8, 6
        MIDDLE_TIP, MIDDLE_PIP = 12, 10
        RING_TIP, RING_PIP = 16, 14
        PINKY_TIP, PINKY_PIP = 20, 18
        WRIST = 0
        
        # Check finger extensions
        thumb_extended = self.is_finger_extended(lm, THUMB_TIP, THUMB_PIP)
        index_extended = self.is_finger_extended(lm, INDEX_TIP, INDEX_PIP)
        middle_extended = self.is_finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP)
        ring_extended = self.is_finger_extended(lm, RING_TIP, RING_PIP)
        pinky_extended = self.is_finger_extended(lm, PINKY_TIP, PINKY_PIP)
        
        fingers_extended = [index_extended, middle_extended, ring_extended, pinky_extended]
        fingers_up_count = sum(fingers_extended)
        
        # Gesture recognition for music
        
        # Closed Fist (silence)
        if not any(fingers_extended) and not thumb_extended:
            return "✊ Closed Fist", 0.95
        
        # Open Palm (A)
        if all(fingers_extended) and thumb_extended:
            return "✋ Open Palm", 0.95
        
        # Thumbs Up (B)
        if thumb_extended and not any(fingers_extended):
            thumb_wrist_dist = self.distance(lm[THUMB_TIP], lm[WRIST])
            if thumb_wrist_dist > 0.2:
                return "👍 Thumbs Up", 0.9
        
        # Counting gestures (C, D, E, F)
        if fingers_up_count == 1 and not thumb_extended:
            return "1️⃣ One Finger", 0.85
        elif fingers_up_count == 2 and not thumb_extended:
            return "2️⃣ Two Fingers", 0.85
        elif fingers_up_count == 3 and not thumb_extended:
            return "3️⃣ Three Fingers", 0.85
        elif fingers_up_count == 4 and not thumb_extended:
            return "4️⃣ Four Fingers", 0.85
        
        return "❓ Unknown", 0.5
    
    def process_frame(self, frame):
        """Process frame and detect hand gestures"""
        gestures = []
        
        try:
            h, w, c = frame.shape
            frame = cv2.flip(frame, 1)
            
            if self.hand_detector_ready and self.landmarker is not None:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if rgb_frame.dtype != np.uint8:
                        rgb_frame = (rgb_frame * 255).astype(np.uint8)
                    
                    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
                    detection_result = self.landmarker.detect(mp_image)
                    
                    if (hasattr(detection_result, 'hand_landmarks') and 
                        detection_result.hand_landmarks and 
                        hasattr(detection_result, 'handedness') and
                        detection_result.handedness):
                        
                        for hand_landmarks, handedness in zip(
                            detection_result.hand_landmarks,
                            detection_result.handedness
                        ):
                            gesture_name, confidence = self.detect_gesture(hand_landmarks)
                            hand_label = handedness[0].category_name
                            gestures.append((gesture_name, confidence, hand_landmarks, hand_label))
                            self.draw_hand_landmarks(frame, hand_landmarks)
                            
                except Exception as e:
                    # Silently handle detection errors to prevent crashes
                    pass
        except Exception as e:
            print(f"Frame processing error: {e}")
        
        return frame, gestures
    
    def draw_hand_landmarks(self, frame, landmarks):
        """Draw hand landmarks on frame"""
        if not landmarks or len(landmarks) != 21:
            return
        
        h, w, c = frame.shape
        
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                start_pos = (int(start.x * w), int(start.y * h))
                end_pos = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)
        
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
    
    def draw_gestures(self, frame, gestures, current_note):
        """Draw detected gestures and current note on frame"""
        h, w, c = frame.shape
        
        for i, (gesture_name, confidence, landmarks, hand_label) in enumerate(gestures):
            if not landmarks or len(landmarks) != 21:
                continue
                
            x_coords = [pt.x for pt in landmarks]
            y_coords = [pt.y for pt in landmarks]
            
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
            
            x_min, x_max = max(0, x_min - 20), min(w, x_max + 20)
            y_min, y_max = max(0, y_min - 20), min(h, y_max + 20)
            
            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            
            # Draw gesture label
            label = f"{hand_label}: {gesture_name}"
            if current_note and current_note != 'SILENCE':
                label += f" → {current_note}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            
            label_y = max(30, y_min - 10)
            cv2.rectangle(frame, 
                         (x_min, label_y - text_size[1] - 5),
                         (x_min + text_size[0] + 5, label_y + 5),
                         (0, 255, 0), -1)
            cv2.putText(frame, label, (x_min + 2, label_y - 2),
                       font, font_scale, (0, 0, 0), thickness)
        
        return frame


def main():
    """Main function to run gesture-controlled music"""
    detector = HandGestureDetector()
    synthesizer = MusicSynthesizer()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    cv2.namedWindow('Gesture-Controlled Music', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gesture-Controlled Music', 1200, 800)
    
    print("\n" + "=" * 70)
    print(" Gesture-Controlled Musical Instrument")
    print("=" * 70)
    print("\nGesture → Note Mapping:")
    print("  1️⃣  One Finger   → C (261.63 Hz)")
    print("  2️⃣  Two Fingers  → D (293.66 Hz)")
    print("  3️⃣  Three Fingers → E (329.63 Hz)")
    print("  4️⃣  Four Fingers → F (349.23 Hz)")
    print("  ✋ Open Palm    → A (440.00 Hz)")
    print("  👍 Thumbs Up    → B (493.88 Hz)")
    print("  ✊ Closed Fist  → Silence")
    print("\n" + "=" * 70)
    print("Instructions:")
    print("  • Show gestures to play musical notes")
    print("  • Notes play continuously until gesture changes")
    print("  • Make a fist to stop sound")
    print("\nPress ESC to exit")
    print("=" * 70 + "\n")
    
    frame_count = 0
    last_gesture = None
    
    try:
        while True:
            try:
                success, frame = cap.read()
                
                if not success:
                    print("Error: Failed to capture frame")
                    break
                
                frame_count += 1
                
                # Process frame for gestures
                frame, gestures = detector.process_frame(frame)
                
                # Update music based on gesture
                current_gesture = None
                current_note = synthesizer.current_note
                
                if gestures:
                    gesture_name = gestures[0][0]
                    current_gesture = gesture_name
                    
                    # Check if gesture changed
                    if gesture_name != last_gesture:
                        note = synthesizer.get_note_for_gesture(gesture_name)
                        if note:
                            print(f"Gesture: {gesture_name} → Note: {note}")
                            synthesizer.play_note(note)
                            last_gesture = gesture_name
                else:
                    # No gesture detected - stop sound
                    if last_gesture is not None:
                        print("No gesture detected → Silence")
                        synthesizer.play_note('SILENCE')
                        last_gesture = None
                
                # Draw gestures with current note
                frame = detector.draw_gestures(frame, gestures, current_note)
                
                # Display info
                h, w, c = frame.shape
                cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, 'Press ESC to quit', (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                if detector.hand_detector_ready:
                    status_text = "✓ Music Mode Active"
                    color = (0, 255, 0)
                else:
                    status_text = "✗ Model Not Ready"
                    color = (0, 0, 255)
                cv2.putText(frame, status_text, (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Show current note
                if current_note and current_note != 'SILENCE':
                    note_text = f"Playing: {current_note}"
                    cv2.putText(frame, note_text, (10, h - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                
                cv2.imshow('Gesture-Controlled Music', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("\nExiting...")
                    break
                    
            except Exception as e:
                print(f"Frame processing error: {e}")
                continue
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        print("Cleaning up...")
        synthesizer.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")


if __name__ == "__main__":
    main()
