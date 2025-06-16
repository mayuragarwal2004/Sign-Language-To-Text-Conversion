import cv2
import mediapipe as mp
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import time
import numpy as np
import json
from datetime import datetime
import uuid
import threading

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# Globals for recording
output_dir = "gesture_data"
os.makedirs(output_dir, exist_ok=True)

# Landmark names for detailed labeling (only upper body)
UPPER_BODY_LANDMARKS = {
    0: "nose",
    1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear",
    9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb"
}

HAND_LANDMARKS = {
    "wrist": 0,
    "thumb_cmc": 1, "thumb_mcp": 2, "thumb_ip": 3, "thumb_tip": 4,
    "index_finger_mcp": 5, "index_finger_pip": 6, "index_finger_dip": 7, "index_finger_tip": 8,
    "middle_finger_mcp": 9, "middle_finger_pip": 10, "middle_finger_dip": 11, "middle_finger_tip": 12,
    "ring_finger_mcp": 13, "ring_finger_pip": 14, "ring_finger_dip": 15, "ring_finger_tip": 16,
    "pinky_mcp": 17, "pinky_pip": 18, "pinky_dip": 19, "pinky_tip": 20
}

def extract_upper_body_data(pose_landmarks, hands_results):
    if not pose_landmarks:
        return None, None, None, None, False

    lm = pose_landmarks.landmark

    # We'll use only upper body landmarks (0-22)
    indices = list(UPPER_BODY_LANDMARKS.keys())

    if any(lm[i].visibility < 0.5 for i in [11, 12, 13, 14, 15, 16]):  # Key upper body points
        return None, None, None, None, False

    l_shoulder = lm[11]
    r_shoulder = lm[12]
    ref_x = (l_shoulder.x + r_shoulder.x) / 2
    ref_y = (l_shoulder.y + r_shoulder.y) / 2

    shoulder_dist = ((l_shoulder.x - r_shoulder.x) ** 2 + (l_shoulder.y - r_shoulder.y) ** 2) ** 0.5
    scale = shoulder_dist if shoulder_dist > 0 else 1

    # Original coordinates with visibility
    original_pose_data = []
    for i in indices:
        original_pose_data.extend([lm[i].x, lm[i].y, lm[i].z, lm[i].visibility])

    # Normalized pose coordinates
    norm_pose_data = []
    for i in indices:
        norm_x = (lm[i].x - ref_x) / scale
        norm_y = (lm[i].y - ref_y) / scale
        norm_z = lm[i].z / scale
        norm_pose_data.extend([norm_x, norm_y, norm_z])

    # Hands data
    original_hands_data = []
    norm_hands_data = []
    hand_visibility = []
    
    for h in range(2):
        if hands_results.multi_hand_landmarks and len(hands_results.multi_hand_landmarks) > h:
            hand_lm = hands_results.multi_hand_landmarks[h].landmark
            for i in range(21):
                # Original hand coordinates
                original_hands_data.extend([hand_lm[i].x, hand_lm[i].y, hand_lm[i].z, 1.0])  # visibility=1 for hands
                # Normalized hand coordinates
                norm_x = (hand_lm[i].x - ref_x) / scale
                norm_y = (hand_lm[i].y - ref_y) / scale
                norm_z = hand_lm[i].z / scale
                norm_hands_data.extend([norm_x, norm_y, norm_z])
            hand_visibility.append(True)
        else:
            # Append zeros if hand not detected
            original_hands_data.extend([0.0] * (21 * 4))
            norm_hands_data.extend([0.0] * (21 * 3))
            hand_visibility.append(False)

    # Combine all data
    original_data = original_pose_data + original_hands_data
    norm_data = norm_pose_data + norm_hands_data
    
    # Body-only data (just pose landmarks)
    body_only_norm = norm_pose_data[:]
    body_only_original = original_pose_data[:]

    return original_data, norm_data, body_only_original, body_only_norm, True

def save_session_data(label, data, original_video_frames, labeled_video_frames, original_fps):
    # Create label directory if it doesn't exist
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    
    # Generate unique ID for this session
    session_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create session directory
    session_dir = os.path.join(label_dir, f"{timestamp}_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    
    try:
        # Save videos with original FPS
        if original_video_frames:
            height, width, _ = original_video_frames[0].shape
            original_video_path = os.path.join(session_dir, "original.avi")
            out = cv2.VideoWriter(original_video_path, cv2.VideoWriter_fourcc(*'XVID'), original_fps, (width, height))
            for frame in original_video_frames:
                out.write(frame)
            out.release()
        
        if labeled_video_frames:
            height, width, _ = labeled_video_frames[0].shape
            labeled_video_path = os.path.join(session_dir, "labeled.avi")
            out = cv2.VideoWriter(labeled_video_path, cv2.VideoWriter_fourcc(*'XVID'), original_fps, (width, height))
            for frame in labeled_video_frames:
                out.write(frame)
            out.release()
        
        # Prepare data for saving
        all_original = []
        all_normalized = []
        all_body_only_original = []
        all_body_only_normalized = []
        
        for frame_data in data:
            original, normalized, body_original, body_norm, _ = frame_data
            all_original.append(original)
            all_normalized.append(normalized)
            all_body_only_original.append(body_original)
            all_body_only_normalized.append(body_norm)
        
        # Save numpy arrays
        np.save(os.path.join(session_dir, "original_coords.npy"), np.array(all_original))
        np.save(os.path.join(session_dir, "normalized_coords.npy"), np.array(all_normalized))
        np.save(os.path.join(session_dir, "original_body_only.npy"), np.array(all_body_only_original))
        np.save(os.path.join(session_dir, "normalized_body_only.npy"), np.array(all_body_only_normalized))
        
        # Create stickman videos
        if data and len(data) > 0:
            # Get first frame dimensions from stickman image
            stickman_with_coords = draw_stickman_from_normalized(data[0][1], show_coords=True)
            height, width, _ = stickman_with_coords.shape
            
            # Video with coordinates
            stickman_coords_path = os.path.join(session_dir, "stickman_with_coords.avi")
            out_coords = cv2.VideoWriter(stickman_coords_path, cv2.VideoWriter_fourcc(*'XVID'), original_fps, (width, height))
            
            # Video without coordinates
            stickman_clean_path = os.path.join(session_dir, "stickman_clean.avi")
            out_clean = cv2.VideoWriter(stickman_clean_path, cv2.VideoWriter_fourcc(*'XVID'), original_fps, (width, height))
            
            for frame_data in data:
                _, norm_data, _, _, _ = frame_data
                stickman_coords = draw_stickman_from_normalized(norm_data, show_coords=True)
                stickman_clean = draw_stickman_from_normalized(norm_data, show_coords=False)
                
                out_coords.write(stickman_coords)
                out_clean.write(stickman_clean)
            
            out_coords.release()
            out_clean.release()
        
        return session_dir
    
    except Exception as e:
        raise Exception(f"Error saving session data: {str(e)}")

def draw_stickman_from_normalized(data, canvas_size=(600, 600), show_coords=True):
    img = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
    
    # Upper body connections (simplified)
    connections = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye/ear
        (0, 4), (4, 5), (5, 6), (6, 8),   # Right eye/ear
        (9, 10),  # Mouth
        
        # Shoulders to elbows to wrists
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        
        # Hands (simplified)
        (15, 17), (15, 19), (15, 21),  # Left hand connections
        (16, 18), (16, 20), (16, 22)   # Right hand connections
    ]

    # Extract points (only upper body landmarks 0-22)
    points = []
    for i in range(0, min(23*3, len(data)), 3):
        if i+2 < len(data):
            points.append((data[i], data[i+1]))

    if not points:
        return img

    xs, ys = zip(*points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    padding = 50

    scale_x = (canvas_size[0] - 2 * padding) / (max_x - min_x + 1e-5)
    scale_y = (canvas_size[1] - 2 * padding) / (max_y - min_y + 1e-5)
    scale = min(scale_x, scale_y)

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    cx, cy = canvas_size[0] // 2, canvas_size[1] // 2

    def to_canvas_coords(x, y):
        px = int(cx + (x - center_x) * scale)
        py = int(cy + (y - center_y) * scale)
        return px, py

    # Draw connections
    for a, b in connections:
        if a < len(points) and b < len(points):
            p1 = to_canvas_coords(*points[a])
            p2 = to_canvas_coords(*points[b])
            cv2.line(img, p1, p2, (0, 0, 255), 2)  # Red lines

    # Draw joints
    for i, (x, y) in enumerate(points):
        px, py = to_canvas_coords(x, y)
        cv2.circle(img, (px, py), 4, (0, 150, 0), -1)
        
        if show_coords and i in UPPER_BODY_LANDMARKS:
            # Show coordinates
            cv2.putText(img, f"{x:.2f},{y:.2f}", (px + 5, py - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            # Show landmark name
            cv2.putText(img, UPPER_BODY_LANDMARKS[i], (px + 5, py + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return img

class GestureRecorderApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get actual camera FPS
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.original_fps <= 0:
            self.original_fps = 30  # Default if cannot get FPS
            
        self.canvas_width = 960
        self.canvas_height = 720

        self.view_frame = tk.Frame(window)
        self.view_frame.pack()

        self.canvas = tk.Canvas(self.view_frame, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(side=tk.LEFT)

        self.stickman_panel = tk.Label(self.view_frame)
        self.stickman_panel.pack(side=tk.LEFT, padx=10)

        self.btn_frame = tk.Frame(window)
        self.btn_frame.pack(fill=tk.X, pady=10)

        self.record_btn = tk.Button(self.btn_frame, text="Start Recording", command=self.start_recording)
        self.record_btn.pack(side=tk.LEFT, padx=10)

        self.stop_btn = tk.Button(self.btn_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10)

        self.save_btn = tk.Button(self.btn_frame, text="Save Dataset", command=self.save_data)
        self.save_btn.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(window, text="Status: Not Recording", fg="green")
        self.status_label.pack()

        self.recording = False
        self.data = []
        self.label = ""
        self.original_frames = []
        self.labeled_frames = []
        self.last_frame_time = time.time()
        self.frame_times = []  # To track actual frame rate

        self.update_video()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_recording(self):
        self.recording = True
        self.data.clear()
        self.original_frames.clear()
        self.labeled_frames.clear()
        self.frame_times.clear()
        self.label = ""
        self.record_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Recording...", fg="red")
        self.last_frame_time = time.time()

    def stop_recording(self):
        self.recording = False
        self.record_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Recording stopped")
        
        # Calculate actual FPS during recording
        if len(self.frame_times) > 1:
            durations = np.diff(self.frame_times)
            actual_fps = 1.0 / np.mean(durations)
            print(f"Actual recording FPS: {actual_fps:.2f}")
        
        # Ask for label
        self.label = simpledialog.askstring("Input", "Enter label for recorded gesture:", parent=self.window)
        if self.label:
            messagebox.showinfo("Info", f"Gesture label set to: {self.label}")
        else:
            messagebox.showwarning("Warning", "No label entered, data will not be saved until label is provided.")

    def save_data(self):
        if not self.data:
            messagebox.showwarning("Warning", "No data to save!")
            return
        if not self.label:
            self.label = simpledialog.askstring("Input", "Enter label for recorded gesture:", parent=self.window)
            if not self.label:
                messagebox.showwarning("Warning", "No label entered. Save cancelled.")
                return

        # Calculate actual FPS from recorded frames
        if len(self.frame_times) > 1:
            durations = np.diff(self.frame_times)
            actual_fps = 1.0 / np.mean(durations)
        else:
            actual_fps = self.original_fps

        # Disable buttons during save
        self.record_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Saving data...", fg="blue")

        # Use threading to prevent UI freeze during save
        def save_thread():
            try:
                session_dir = save_session_data(
                    self.label, 
                    self.data, 
                    self.original_frames, 
                    self.labeled_frames,
                    actual_fps
                )
                self.window.after(0, lambda: messagebox.showinfo("Success", f"Saved data to {session_dir}"))
            except Exception as e:
                self.window.after(0, lambda: messagebox.showerror("Error", f"Failed to save data: {str(e)}"))
            finally:
                self.window.after(0, self.reset_after_save)

        threading.Thread(target=save_thread, daemon=True).start()

    def reset_after_save(self):
        self.data.clear()
        self.original_frames.clear()
        self.labeled_frames.clear()
        self.frame_times.clear()
        self.label = ""
        self.record_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Not Recording", fg="green")

    def update_video(self):
        current_time = time.time()
        ret, frame = self.cap.read()
        
        if ret:
            frame = cv2.flip(frame, 1)  # mirror image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_results = pose.process(frame_rgb)
            hands_results = hands.process(frame_rgb)

            # Create labeled frame
            labeled_frame = frame.copy()
            
            if pose_results.pose_landmarks:
                # Only draw upper body landmarks
                for landmark_idx in UPPER_BODY_LANDMARKS:
                    if landmark_idx < len(pose_results.pose_landmarks.landmark):
                        landmark = pose_results.pose_landmarks.landmark[landmark_idx]
                        if landmark.visibility > 0.5:
                            x = int(landmark.x * frame.shape[1])
                            y = int(landmark.y * frame.shape[0])
                            cv2.circle(labeled_frame, (x, y), 5, (0, 255, 0), -1)

            if hands_results.multi_hand_landmarks:
                for hand_lms in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        labeled_frame, 
                        hand_lms, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 0, 0)),  # Blue for hands
                        mp_drawing.DrawingSpec(color=(255, 0, 0)))
                    
            try:
                # Extract upper body data only
                original_data, norm_data, body_original, body_norm, all_visible = extract_upper_body_data(
                    pose_results.pose_landmarks, 
                    hands_results
                )
            except Exception as e:
                original_data, norm_data, body_original, body_norm, all_visible = None, None, None, None, False
                print(f"Error extracting data: {e}")

            if original_data is None:
                all_visible = False
                self.status_label.config(text="Status: Body Occluded", fg="red")
            else:
                self.status_label.config(text="Status: Body Visible", fg="green")
            
            # Update status
            status_text = "Body Visible" if all_visible else "Occluded – Adjust Position"
            status_color = (0, 255, 0) if all_visible else (0, 0, 255)
            cv2.putText(labeled_frame, status_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            # Recording logic
            if self.recording:
                if all_visible:
                    self.data.append((original_data, norm_data, body_original, body_norm, all_visible))
                    self.original_frames.append(frame.copy())
                    self.labeled_frames.append(labeled_frame.copy())
                    self.frame_times.append(current_time)
                else:
                    # Clear data if occlusion occurs
                    self.data.clear()
                    self.original_frames.clear()
                    self.labeled_frames.clear()
                    self.frame_times.clear()
                    self.status_label.config(text="Status: Restarting due to occlusion", fg="orange")

            # Display stickman if we have normalized data
            if norm_data:
                stickman = draw_stickman_from_normalized(norm_data, show_coords=True)
                stickman_img = Image.fromarray(stickman)
                stickman_imgtk = ImageTk.PhotoImage(image=stickman_img)
                self.stickman_panel.configure(image=stickman_imgtk)
                self.stickman_panel.imgtk = stickman_imgtk

            # Calculate and display FPS
            elapsed = current_time - self.last_frame_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            cv2.putText(labeled_frame, f"FPS: {fps:.1f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
            
            # Display recording status
            if self.recording:
                height, width, _ = labeled_frame.shape
                cv2.putText(labeled_frame, "REC", (width - 100, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(labeled_frame, f"Frames: {len(self.data)}", (30, height - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Convert to PIL image and display
            img = cv2.cvtColor(labeled_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            
            self.last_frame_time = current_time
        
        self.window.after(10, self.update_video)

    def on_closing(self):
        if self.recording and self.data:
            if messagebox.askyesno("Quit", "Recording in progress. Save before quitting?"):
                self.save_data()
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.state("zoomed")  # Maximizes the window (Windows only)
    app = GestureRecorderApp(root, "Upper Body Gesture Recorder")
    root.mainloop()