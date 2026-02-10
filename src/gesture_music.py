import cv2
import numpy as np
import math
import os
import time
import sounddevice as sd
import threading
import queue
from collections import deque
from datetime import datetime
import subprocess

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
            '🤘 Rock Sign': 'G',
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


class AudioRecorder:
    """Records audio from microphone and saves video with audio"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.audio_frames = []
        self.desktop_audio_frames = []
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.desktop_queue = queue.Queue()
        self.stream = None
        self.desktop_stream = None
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input"""
        if status:
            print(f"Microphone status: {status}")
        # Collect frames while recording
        if self.is_recording:
            self.audio_frames.append(indata.copy())
        self.audio_queue.put(indata.copy())
    
    def desktop_callback(self, indata, frames, time_info, status):
        """Callback for desktop audio input"""
        if status:
            print(f"Desktop audio status: {status}")
        # Collect desktop audio frames while recording
        if self.is_recording:
            self.desktop_audio_frames.append(indata.copy())
        self.desktop_queue.put(indata.copy())
    
    def start_recording(self):
        """Start recording audio from microphone and desktop"""
        self.is_recording = True
        self.audio_frames = []
        self.desktop_audio_frames = []
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=2048
            )
            self.stream.start()
            print("Microphone recording started...")
            
            # Try to record desktop audio (stereo mix / loopback device)
            try:
                # Get list of available devices
                devices = sd.query_devices()
                stereo_mix_device = None
                
                # Look for stereo mix or loopback device
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        if 'stereo mix' in device['name'].lower() or 'loopback' in device['name'].lower():
                            stereo_mix_device = i
                            break
                
                if stereo_mix_device is not None:
                    self.desktop_stream = sd.InputStream(
                        device=stereo_mix_device,
                        samplerate=self.sample_rate,
                        channels=1,
                        callback=self.desktop_callback,
                        blocksize=2048
                    )
                    self.desktop_stream.start()
                    print(f"Desktop audio recording started from device: {stereo_mix_device}")
                else:
                    print("Stereo mix/loopback device not found. Desktop audio won't be recorded.")
                    print("To enable: Settings → Sound → Volume mix options → Turn on stereo mix")
            except Exception as e:
                print(f"Desktop audio recording skipped: {e}")
                
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
    
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop()
            time.sleep(0.1)
            self.stream.close()
        
        if self.desktop_stream:
            self.desktop_stream.stop()
            time.sleep(0.1)
            self.desktop_stream.close()
        
        # Collect microphone audio
        while not self.audio_queue.empty():
            try:
                frame = self.audio_queue.get_nowait()
                self.audio_frames.append(frame)
            except queue.Empty:
                break
        
        # Collect desktop audio
        while not self.desktop_queue.empty():
            try:
                frame = self.desktop_queue.get_nowait()
                self.desktop_audio_frames.append(frame)
            except queue.Empty:
                break
        
        print(f"Recording stopped. Microphone frames: {len(self.audio_frames)}, Desktop frames: {len(self.desktop_audio_frames)}")
        return self.get_audio_data()
    
    def get_audio_data(self):
        """Get recorded audio as numpy array"""
        if not self.audio_frames:
            return None
        return np.concatenate(self.audio_frames, axis=0)
    
    def get_desktop_audio_data(self):
        """Get recorded desktop audio as numpy array"""
        if not self.desktop_audio_frames:
            return None
        return np.concatenate(self.desktop_audio_frames, axis=0)
    

    
    def save_video_with_audio(self, video_frames, fps, output_filename=None):
        """Save video with recorded audio to desktop"""
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"hand_gesture_recording_{timestamp}.mp4"
        
        # Get desktop path
        desktop_path = os.path.expanduser("~/Desktop")
        output_path = os.path.join(desktop_path, output_filename)
        
        if not video_frames:
            print("No video frames to save")
            return None
        
        # Get video properties
        frame = video_frames[0]
        h, w, c = frame.shape
        
        # Create temporary video file without audio
        temp_video_path = os.path.join(desktop_path, f"temp_{output_filename}")
        
        # Write video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (w, h))
        
        for frame in video_frames:
            out.write(frame)
        out.release()
        print(f"Video written: {len(video_frames)} frames at {fps} fps")
        
        # If we have audio, combine video and audio using ffmpeg
        audio_data = self.get_audio_data()
        desktop_audio_data = self.get_desktop_audio_data()
        
        print(f"Microphone audio shape: {audio_data.shape if audio_data is not None else 'None'}")
        print(f"Desktop audio shape: {desktop_audio_data.shape if desktop_audio_data is not None else 'None'}")
        
        # Mix microphone and desktop audio if both exist
        if desktop_audio_data is not None and len(desktop_audio_data) > 0:
            print("Mixing microphone and desktop audio...")
            # Make sure both are same length
            min_len = min(len(audio_data), len(desktop_audio_data))
            audio_data = audio_data[:min_len]
            desktop_audio_data = desktop_audio_data[:min_len]
            
            # Mix with equal volume
            audio_data = (audio_data + desktop_audio_data) / 2
        
        if audio_data is not None and len(audio_data) > 0:
            # Save audio temporarily
            temp_audio_path = os.path.join(desktop_path, f"temp_audio_{output_filename}.wav")
            
            try:
                import scipy.io.wavfile as wavfile
                
                # Normalize audio
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_normalized = audio_data / max_val
                else:
                    audio_normalized = audio_data
                
                # Convert to int16
                audio_int16 = np.int16(audio_normalized * 32767)
                wavfile.write(temp_audio_path, self.sample_rate, audio_int16)
                print(f"Audio file saved: {temp_audio_path} ({len(audio_int16)} samples)")
                
                # Combine video and audio using ffmpeg
                try:
                    ffmpeg_path = r'c:\Users\janel\Documents\vscode\Hand-detection-mediapipe\ffmpeg-8.0.1\ffmpeg-2026-02-09-git-9bfa1635ae-full_build\ffmpeg-2026-02-09-git-9bfa1635ae-full_build\bin\ffmpeg.exe'
                    
                    print("Running FFmpeg to combine video and audio...")
                    result = subprocess.run([
                        ffmpeg_path, '-i', temp_video_path, '-i', temp_audio_path,
                        '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y',
                        output_path
                    ], capture_output=True, text=True, timeout=60)
                    
                    if result.returncode != 0:
                        print(f"FFmpeg error: {result.stderr}")
                        print("Saving video without audio instead...")
                        # If FFmpeg fails, just return the video without audio
                        print(f"Video saved to: {temp_video_path}")
                        return temp_video_path
                    
                    # Clean up temporary files
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
                    
                    print(f"\n✓ Video with audio saved to: {output_path}")
                    return output_path
                    
                except FileNotFoundError as e:
                    print(f"FFmpeg not found: {e}")
                    print("Saving video without audio.")
                    print(f"Video saved to: {temp_video_path}")
                    return temp_video_path
                except Exception as e:
                    print(f"FFmpeg error: {e}")
                    print("Saving video without audio.")
                    return temp_video_path
                    
            except ImportError:
                print("scipy not installed. Saving video without audio.")
                print(f"Video saved to: {temp_video_path}")
                return temp_video_path
            except Exception as e:
                print(f"Audio save error: {e}")
                print(f"Video saved to: {temp_video_path}")
                return temp_video_path
        else:
            print("No audio data recorded.")
            print(f"Video saved to: {temp_video_path}")
            return temp_video_path
            return temp_video_path


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
        
        # Rock Sign (G) - index and pinky extended, middle and ring closed
        if index_extended and pinky_extended and not middle_extended and not ring_extended:
            return "🤘 Rock Sign", 0.9
        
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
        
        # Draw hand connections in blue with thin lines
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                start_pos = (int(start.x * w), int(start.y * h))
                end_pos = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_pos, end_pos, (255, 0, 0), 1)  # Blue color, thin line
        
        # Draw hand landmarks as small blue circles
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Blue circles, smaller
    
    def draw_gestures(self, frame, gestures, current_note):
        """Draw detected gestures and current note on frame"""
        # Just return frame without drawing text or boxes
        # Hand landmarks are already drawn in process_frame
        return frame


def main():
    """Main function to run gesture-controlled music"""
    detector = HandGestureDetector()
    synthesizer = MusicSynthesizer()
    recorder = AudioRecorder()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default FPS
    
    cv2.namedWindow('Gesture-Controlled Music', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gesture-Controlled Music', 1200, 800)
    
    print("\n" + "=" * 70)
    print(" Gesture-Controlled Musical Instrument with Voice Recording")
    print("=" * 70)
    print("\nGesture → Note Mapping:")
    print("  1️⃣  One Finger   → C (261.63 Hz)")
    print("  2️⃣  Two Fingers  → D (293.66 Hz)")
    print("  3️⃣  Three Fingers → E (329.63 Hz)")
    print("  4️⃣  Four Fingers → F (349.23 Hz)")
    print("  🤘 Rock Sign    → G (392.00 Hz)")
    print("  ✋ Open Palm    → A (440.00 Hz)")
    print("  👍 Thumbs Up    → B (493.88 Hz)")
    print("  ✊ Closed Fist  → Silence")
    print("\n" + "=" * 70)
    print("Instructions:")
    print("  • Show gestures to control hand tracking")
    print("  • SPACE: Start/Stop microphone recording")
    print("  • 'R': Save recording to Desktop (microphone + desktop audio mixed)")
    print("  • ESC: Exit application")
    print("=" * 70)
    print("Recording Features:")
    print("  🎤 Microphone audio recording")
    print("  🎵 Desktop audio capture (if stereo mix enabled)")
    print("=" * 70 + "\n")
    
    frame_count = 0
    last_gesture = None
    video_frames = []
    is_recording = False
    
    try:
        while True:
            success, frame = cap.read()
            
            if not success:
                print("Error: Failed to capture frame")
                break
            
            frame_count += 1
            
            # Process frame for gestures
            frame, gestures = detector.process_frame(frame)
            
            # Store frame if recording
            if is_recording:
                video_frames.append(frame.copy())
            
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
                        last_gesture = gesture_name
            else:
                # No gesture detected - stop sound
                if last_gesture is not None:
                    last_gesture = None
            
            # Draw gestures with current note
            frame = detector.draw_gestures(frame, gestures, current_note)
            
            # Add recording status indicator
            status_text = "Recording" if is_recording else "Ready"
            status_color = (0, 255, 0) if is_recording else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Display the frame
            cv2.imshow('Gesture-Controlled Music', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE - Start/Stop recording
                if not is_recording:
                    recorder.start_recording()
                    video_frames = []
                    is_recording = True
                else:
                    recorder.stop_recording()
                    is_recording = False
                    print(f"Recorded {len(video_frames)} frames")
            elif key == ord('r') or key == ord('R'):  # R - Save recording
                if video_frames:
                    print("Saving recording...")
                    output_path = recorder.save_video_with_audio(video_frames, fps)
                    if output_path:
                        print(f"Successfully saved to: {output_path}")
                    video_frames = []
                else:
                    print("No frames to save. Start recording first (SPACE).")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        if is_recording:
            recorder.stop_recording()
        print("Cleaning up...")
        synthesizer.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")


if __name__ == "__main__":
    main()
