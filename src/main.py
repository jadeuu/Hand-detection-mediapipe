import cv2
import numpy as np
import math
import os
from collections import deque

try:
    from mediapipe import Image, ImageFormat
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_TASKS_AVAILABLE = True
except (ImportError, AttributeError):
    MEDIAPIPE_TASKS_AVAILABLE = False

class HandGestureDetector:
    """
    Detects hand gestures using MediaPipe's hand landmark model.
    Tracks 21 3D hand keypoints for precise gesture recognition.
    """
    
    def __init__(self):
        self.hand_detector_ready = False
        self.landmarker = None
        self.frame_index = 0
        
        if MEDIAPIPE_TASKS_AVAILABLE:
            try:
                # Get absolute path to model file
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_dir, 'hand_landmarker.task')
                
                print(f"Looking for model at: {model_path}")
                print(f"Model file exists: {os.path.exists(model_path)}")
                
                if not os.path.exists(model_path):
                    print(f"Error: Model file not found at {model_path}")
                    return
                
                # Create hand landmarker options
                base_options = python.BaseOptions(
                    model_asset_path=model_path
                )
                print("Creating HandLandmarkerOptions...")
                
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                
                print("Creating HandLandmarker...")
                self.landmarker = vision.HandLandmarker.create_from_options(options)
                self.hand_detector_ready = True
                print("✓ Hand landmark model loaded successfully!")
                print(f"✓ Model ready for gesture detection")
                
            except FileNotFoundError as e:
                print(f"Model file not found: {e}")
                print("Please download from: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker")
            except Exception as e:
                print(f"Error loading hand detection model: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
    
    def distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        if p1 is None or p2 is None:
            return 0
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def get_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        a = self.distance(p1, p2)
        b = self.distance(p2, p3)
        c = self.distance(p1, p3)
        
        if a == 0 or b == 0:
            return 0
        
        cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
        angle = math.degrees(math.acos(cos_angle))
        return angle
    
    def is_finger_extended(self, landmarks, finger_tip, finger_pip):
        """Check if a finger is extended (tip above PIP joint)"""
        if landmarks[finger_tip] is None or landmarks[finger_pip] is None:
            return False
        return landmarks[finger_tip].y < landmarks[finger_pip].y
    
    def detect_gesture(self, landmarks):
        """
        Detect hand gesture from 21 landmark points (as a list).
        Returns gesture name and confidence.
        """
        if not landmarks or len(landmarks) != 21:
            return "No hand", 0
        
        lm = landmarks
        
        # Landmark indices (MediaPipe Hand 21-point model)
        THUMB_TIP = 4
        THUMB_PIP = 3
        THUMB_IP = 2
        INDEX_TIP = 8
        INDEX_PIP = 6
        INDEX_MCP = 5
        MIDDLE_TIP = 12
        MIDDLE_PIP = 10
        MIDDLE_MCP = 9
        RING_TIP = 16
        RING_PIP = 14
        RING_MCP = 13
        PINKY_TIP = 20
        PINKY_PIP = 18
        PINKY_MCP = 17
        
        WRIST = 0
        PALM_CENTER = 9
        
        # Check finger extensions
        thumb_extended = self.is_finger_extended(lm, THUMB_TIP, THUMB_PIP)
        index_extended = self.is_finger_extended(lm, INDEX_TIP, INDEX_PIP)
        middle_extended = self.is_finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP)
        ring_extended = self.is_finger_extended(lm, RING_TIP, RING_PIP)
        pinky_extended = self.is_finger_extended(lm, PINKY_TIP, PINKY_PIP)
        
        fingers_extended = [index_extended, middle_extended, ring_extended, pinky_extended]
        fingers_up_count = sum(fingers_extended)
        
        # Calculate distances for gesture detection
        thumb_index_dist = self.distance(lm[THUMB_TIP], lm[INDEX_TIP])
        thumb_middle_dist = self.distance(lm[THUMB_TIP], lm[MIDDLE_TIP])
        index_middle_dist = self.distance(lm[INDEX_TIP], lm[MIDDLE_TIP])
        
        # Gesture recognition logic
        
        # Closed Fist
        if not any(fingers_extended) and not thumb_extended:
            return "✊ Closed Fist", 0.95
        
        # Open Palm (all fingers extended)
        if all(fingers_extended) and thumb_extended:
            return "✋ Open Palm", 0.95
        
        # Peace Sign (index and middle extended, others closed)
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            if index_middle_dist > 0.05:
                return "✌ Peace Sign", 0.9
        
        # Thumbs Up (thumb extended, others closed)
        if thumb_extended and not any(fingers_extended):
            thumb_wrist_dist = self.distance(lm[THUMB_TIP], lm[WRIST])
            if thumb_wrist_dist > 0.2:
                return "👍 Thumbs Up", 0.9
        
        # OK Sign (thumb and index touching, others extended)
        if thumb_index_dist < 0.05 and middle_extended and ring_extended and pinky_extended:
            return "👌 OK Sign", 0.85
        
        # Rock Sign (index and pinky extended, middle and ring closed)
        if index_extended and pinky_extended and not middle_extended and not ring_extended:
            return "🤘 Rock Sign", 0.9
        
        # Point (only index extended)
        if index_extended and not any([middle_extended, ring_extended, pinky_extended]):
            return "☝ Pointing", 0.85
        
        # Counting with fingers
        if fingers_up_count == 0:
            return "✊ Closed Fist", 0.8
        elif fingers_up_count == 1:
            return "1️⃣ One Finger", 0.8
        elif fingers_up_count == 2:
            return "2️⃣ Two Fingers", 0.8
        elif fingers_up_count == 3:
            return "3️⃣ Three Fingers", 0.8
        elif fingers_up_count == 4:
            return "4️⃣ Four Fingers", 0.8
        
        return "❓ Unknown", 0.5
    
    def process_frame(self, frame):
        """Process frame and detect hand gestures"""
        h, w, c = frame.shape
        
        # Flip for selfie view
        frame = cv2.flip(frame, 1)
        
        gestures = []
        
        if self.hand_detector_ready and self.landmarker is not None:
            try:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Ensure frame is uint8
                if rgb_frame.dtype != np.uint8:
                    rgb_frame = (rgb_frame * 255).astype(np.uint8)
                
                # Create MediaPipe Image
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
                
                # Detect hand landmarks
                detection_result = self.landmarker.detect(mp_image)
                
                # Process detections
                if (hasattr(detection_result, 'hand_landmarks') and 
                    detection_result.hand_landmarks and 
                    hasattr(detection_result, 'handedness') and
                    detection_result.handedness):
                    
                    for hand_landmarks, handedness in zip(
                        detection_result.hand_landmarks,
                        detection_result.handedness
                    ):
                        # Detect gesture
                        gesture_name, confidence = self.detect_gesture(hand_landmarks)
                        hand_label = handedness[0].category_name
                        gestures.append((gesture_name, confidence, hand_landmarks, hand_label))
                        
                        # Draw landmarks on frame
                        self.draw_hand_landmarks(frame, hand_landmarks)
            except Exception as e:
                print(f"Error processing hand landmarks: {type(e).__name__}: {e}")
        
        return frame, gestures
    
    def draw_hand_landmarks(self, frame, landmarks):
        """Draw hand landmarks and connections on frame"""
        if not landmarks or len(landmarks) != 21:
            return
        
        h, w, c = frame.shape
        
        # Landmark connections based on MediaPipe hand model
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                start_pos = (int(start.x * w), int(start.y * h))
                end_pos = (int(end.x * w), int(end.y * h))
                
                cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
    
    def draw_gestures(self, frame, gestures):
        """Draw detected gestures on frame"""
        h, w, c = frame.shape
        
        for i, (gesture_name, confidence, landmarks, hand_label) in enumerate(gestures):
            # Get hand bounding box
            if not landmarks or len(landmarks) != 21:
                continue
                
            x_coords = [pt.x for pt in landmarks]
            y_coords = [pt.y for pt in landmarks]
            
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
            
            # Add padding
            x_min, x_max = max(0, x_min - 20), min(w, x_max + 20)
            y_min, y_max = max(0, y_min - 20), min(h, y_max + 20)
            
            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            
            # Draw gesture label with background
            label = f"{hand_label}: {gesture_name}"
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
    """
    Main function to run hand gesture detection on webcam feed.
    """
    # Initialize gesture detector
    detector = HandGestureDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set window properties
    cv2.namedWindow('Hand Landmark Gesture Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hand Landmark Gesture Detection', 1200, 800)
    
    print("\n" + "=" * 70)
    print(" Hand Gesture Detection - 21 3D Landmark Keypoint Model")
    print("=" * 70)
    print("\nDetected Gestures:")
    print("  ✊ Closed Fist      - All fingers closed")
    print("  ✋ Open Palm        - All fingers extended")
    print("  ✌ Peace Sign       - Index and middle extended")
    print("  👍 Thumbs Up       - Thumb extended upward")
    print("  👌 OK Sign         - Thumb and index touching")
    print("  🤘 Rock Sign       - Index and pinky extended")
    print("  ☝ Pointing         - Only index finger extended")
    print("  1️⃣-4️⃣ Counting      - 1-4 fingers extended")
    print("\n" + "=" * 70)
    print("Features:")
    print("  • Tracks 21 3D hand keypoints")
    print("  • Real-time gesture recognition")
    print("  • Multi-hand detection (up to 2 hands)")
    print("  • Left/Right hand identification")
    print("\nPress ESC to exit")
    print("=" * 70 + "\n")
    
    frame_count = 0
    gesture_history = deque(maxlen=5)
    
    while True:
        success, frame = cap.read()
        
        if not success:
            print("Error: Failed to capture frame")
            break
        
        frame_count += 1
        
        # Process frame for gestures
        frame, gestures = detector.process_frame(frame)
        
        # Draw detected gestures
        frame = detector.draw_gestures(frame, gestures)
        
        # Store current gestures
        if gestures:
            gesture_history.append(gestures[0][0])
        
        # Display information
        h, w, c = frame.shape
        
        # FPS counter (simple)
        cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, 'Press ESC to quit', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if detector.hand_detector_ready:
            status_text = "✓ Hand Landmark Model Ready"
            cv2.putText(frame, status_text, (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            status_text = "✗ Model Not Ready - Download hand_landmarker.task"
            cv2.putText(frame, status_text, (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow('Hand Landmark Gesture Detection', frame)
        
        # Exit on ESC
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("\nExiting...")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()