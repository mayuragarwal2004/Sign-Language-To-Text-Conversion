import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import time
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

# Configuration
MODEL_PATH = "gesture_transformer_model.keras"
LABEL_PATH = "gesture_labels.pkl"
SEQUENCE_LENGTH = 30  # Must match training sequence length

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

class GestureTesterApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Load model and label encoder
        self.model = load_model(MODEL_PATH)
        with open(LABEL_PATH, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Create GUI
        self.create_widgets()
        
        # Initialize prediction buffer
        self.sequence = []
        self.predictions = []
        self.current_gesture = "None"
        self.threshold = 0.7  # Confidence threshold
        self.cooldown = 0  # Frames to wait before next prediction
        
        self.update_video()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        # Main frame
        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video display
        self.video_frame = tk.Frame(self.main_frame)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.video_frame, width=960, height=720)
        self.canvas.pack()
        
        # Info panel
        self.info_frame = tk.Frame(self.main_frame)
        self.info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Current gesture display
        self.current_gesture_label = tk.Label(self.info_frame, text="Current Gesture:", font=('Helvetica', 14))
        self.current_gesture_label.pack(pady=10)
        
        self.gesture_display = tk.Label(self.info_frame, text="None", font=('Helvetica', 24, 'bold'))
        self.gesture_display.pack(pady=20)
        
        # Confidence threshold control
        self.threshold_frame = tk.Frame(self.info_frame)
        self.threshold_frame.pack(pady=20)
        
        tk.Label(self.threshold_frame, text="Confidence Threshold:").pack()
        self.threshold_slider = tk.Scale(
            self.threshold_frame, from_=0.5, to=0.95, resolution=0.05, 
            orient=tk.HORIZONTAL, command=self.update_threshold
        )
        # self.threshold_slider.set(self.threshold)
        # self.threshold_slider.pack()
        
        # Prediction history
        tk.Label(self.info_frame, text="Prediction History:", font=('Helvetica', 14)).pack(pady=10)
        
        self.history_tree = ttk.Treeview(self.info_frame, columns=('gesture', 'confidence'), show='headings')
        self.history_tree.heading('gesture', text='Gesture')
        self.history_tree.heading('confidence', text='Confidence')
        self.history_tree.column('gesture', width=150)
        self.history_tree.column('confidence', width=100)
        self.history_tree.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.history_tree, orient="vertical", command=self.history_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        # Stats
        self.stats_frame = tk.Frame(self.info_frame)
        self.stats_frame.pack(pady=10)
        
        self.fps_label = tk.Label(self.stats_frame, text="FPS: 0")
        self.fps_label.pack(side=tk.LEFT, padx=10)
        
        self.latency_label = tk.Label(self.stats_frame, text="Latency: 0ms")
        self.latency_label.pack(side=tk.LEFT, padx=10)
    
    def update_threshold(self, value):
        self.threshold = float(value)
    
    def extract_features(self, pose_landmarks, hands_results):
        """Extract normalized features in same format as training"""
        if not pose_landmarks:
            return None, False

        lm = pose_landmarks.landmark
        
        # Check visibility of key points
        if any(lm[i].visibility < 0.5 for i in [11, 12, 13, 14, 15, 16]):  # Key upper body points
            return None, False

        # Calculate normalization reference
        l_shoulder = lm[11]
        r_shoulder = lm[12]
        ref_x = (l_shoulder.x + r_shoulder.x) / 2
        ref_y = (l_shoulder.y + r_shoulder.y) / 2
        shoulder_dist = ((l_shoulder.x - r_shoulder.x) ** 2 + (l_shoulder.y - r_shoulder.y) ** 2) ** 0.5
        scale = shoulder_dist if shoulder_dist > 0 else 1

        # Extract normalized pose coordinates (landmarks 0-22)
        norm_data = []
        for i in range(23):  # Upper body landmarks only
            if i < len(lm):
                norm_x = (lm[i].x - ref_x) / scale
                norm_y = (lm[i].y - ref_y) / scale
                norm_z = lm[i].z / scale
                norm_data.extend([norm_x, norm_y, norm_z])
            else:
                norm_data.extend([0.0, 0.0, 0.0])

        # Extract normalized hand coordinates
        for h in range(2):
            if hands_results.multi_hand_landmarks and len(hands_results.multi_hand_landmarks) > h:
                hand_lm = hands_results.multi_hand_landmarks[h].landmark
                for i in range(21):
                    norm_x = (hand_lm[i].x - ref_x) / scale
                    norm_y = (hand_lm[i].y - ref_y) / scale
                    norm_z = hand_lm[i].z / scale
                    norm_data.extend([norm_x, norm_y, norm_z])
            else:
                # Append zeros if hand not detected
                norm_data.extend([0.0] * (21 * 3))

        return np.array(norm_data), True
    
    def update_video(self):
        start_time = time.time()
        ret, frame = self.cap.read()
        
        if ret:
            frame = cv2.flip(frame, 1)  # mirror image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with mediapipe
            pose_results = pose.process(frame_rgb)
            hands_results = hands.process(frame_rgb)
            
            # Draw landmarks
            if pose_results.pose_landmarks:
                # Draw upper body landmarks only
                for landmark_idx in range(23):  # 0-22 are upper body
                    if landmark_idx < len(pose_results.pose_landmarks.landmark):
                        landmark = pose_results.pose_landmarks.landmark[landmark_idx]
                        if landmark.visibility > 0.5:
                            x = int(landmark.x * frame.shape[1])
                            y = int(landmark.y * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            if hands_results.multi_hand_landmarks:
                for hand_lms in hands_results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0)),
                        mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0)))
            
            # Extract features
            features, all_visible = self.extract_features(pose_results.pose_landmarks, hands_results)
            
            if features is not None and all_visible:
                self.sequence.append(features)
                if len(self.sequence) > SEQUENCE_LENGTH:
                    self.sequence = self.sequence[-SEQUENCE_LENGTH:]
                
                # Make prediction when we have enough frames
                if len(self.sequence) == SEQUENCE_LENGTH and self.cooldown <= 0:
                    prediction_start = time.time()
                    input_data = np.expand_dims(self.sequence, axis=0)
                    prediction = self.model.predict(input_data)[0]
                    prediction_time = (time.time() - prediction_start) * 1000  # ms
                    
                    confidence = np.max(prediction)
                    predicted_class = np.argmax(prediction)
                    gesture_name = self.label_encoder.inverse_transform([predicted_class])[0]
                    
                    if confidence > self.threshold:
                        self.current_gesture = gesture_name
                        self.gesture_display.config(text=gesture_name)
                        
                        # Add to history
                        self.history_tree.insert('', 'end', values=(gesture_name, f"{confidence:.2f}"))
                        if self.history_tree.get_children():
                            self.history_tree.see(self.history_tree.get_children()[-1])
                        
                        # Set cooldown (1 second)
                        self.cooldown = int(self.cap.get(cv2.CAP_PROP_FPS))
                    
                    self.predictions.append((gesture_name, confidence))
                    if len(self.predictions) > 10:
                        self.predictions = self.predictions[-10:]
                    
                    self.latency_label.config(text=f"Latency: {prediction_time:.1f}ms")
            
            if self.cooldown > 0:
                self.cooldown -= 1
            
            # Display status
            status_text = "Body Visible" if all_visible else "Adjust Position"
            status_color = (0, 255, 0) if all_visible else (0, 0, 255)
            cv2.putText(frame, status_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # Display current gesture
            cv2.putText(frame, f"Gesture: {self.current_gesture}", (30, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            cv2.putText(frame, f"FPS: {fps:.1f}", (30, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
            
            # Convert to PIL image and display
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        
        self.window.after(10, self.update_video)
    
    def on_closing(self):
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.state("zoomed")  # Maximize window
    app = GestureTesterApp(root, "Gesture Recognition Tester")
    root.mainloop()