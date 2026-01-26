from ultralytics import YOLO
import cv2
import numpy as np
import streamlit as st
import time
from datetime import datetime

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

class TrafficSignalManager:
    def __init__(self):
        self.min_green_duration = 5  # Minimum green light duration
        self.max_green_duration = 30  # Maximum green light duration
        self.vehicles_per_second = 2  # Average vehicles that can pass per second
        self.current_green_lane = None
        self.green_start_time = None
        self.calculated_green_time = 0
        
    def calculate_green_duration(self, vehicle_count):
        # Calculate required time based on vehicle count
        required_time = max(
            self.min_green_duration,
            min(self.max_green_duration, vehicle_count / self.vehicles_per_second)
        )
        return int(required_time)
    
    def should_switch_signal(self, current_time):
        if not self.green_start_time:
            return True
        
        time_elapsed = current_time - self.green_start_time
        return time_elapsed >= self.calculated_green_time
    
    def get_time_remaining(self, current_time):
        if not self.green_start_time:
            return 0
        
        elapsed = current_time - self.green_start_time
        remaining = max(0, self.calculated_green_time - elapsed)
        return int(remaining)

def detect_vehicles(frame):
    results = model(frame)
    # Count only vehicles (car, motorbike, bus, truck)
    vehicle_count = sum(1 for obj in results[0].boxes if obj.cls in [2, 3, 5, 7])
    return vehicle_count or 0, results[0].plot()

def main():
    st.title("AI Based Traffic Management System")
    
    # Create layout
    video_cols = st.columns(4)
    info_cols = st.columns(2)
    
    # Initialize video captures
    lane_videos = ["lane1.mp4", "lane2.mp4", "lane3.mp4", "lane4.mp4"]
    caps = [cv2.VideoCapture(video) for video in lane_videos]
    
    # Initialize displays
    frame_displays = [col.empty() for col in video_cols]
    signal_display = info_cols[0].empty()
    count_display = info_cols[1].empty()
    
    # Initialize traffic manager
    traffic_manager = TrafficSignalManager()
    
    while True:
        current_time = time.time()
        lane_counts = {}
        processed_frames = []
        
        # Process each lane
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            count, processed_frame = detect_vehicles(frame)
            lane_counts[f"Lane {i+1}"] = count
            processed_frames.append(processed_frame)
        
        # Determine which lane should get green signal
        if traffic_manager.should_switch_signal(current_time):
            max_lane = max(lane_counts.items(), key=lambda x: x[1])
            traffic_manager.current_green_lane = max_lane[0]
            traffic_manager.green_start_time = current_time
            traffic_manager.calculated_green_time = traffic_manager.calculate_green_duration(max_lane[1])
        
        # Update displays
        for i, display in enumerate(frame_displays):
            if i < len(processed_frames):
                lane_num = f"Lane {i+1}"
                is_green = lane_num == traffic_manager.current_green_lane
                signal = "🟢" if is_green else "🔴"
                
                if is_green:
                    time_left = traffic_manager.get_time_remaining(current_time)
                    caption = f"{lane_num} {signal} - Vehicles: {lane_counts[lane_num]} (⏱️ {time_left}s)"
                else:
                    caption = f"{lane_num} {signal} - Vehicles: {lane_counts[lane_num]}"
                    
                display.image(processed_frames[i], caption=caption)
        
        # Update signal status
        status_text = "### Traffic Signal Status\n"
        for lane_num in lane_counts.keys():
            is_green = lane_num == traffic_manager.current_green_lane
            signal = "🟢" if is_green else "🔴"
            count = lane_counts[lane_num]
            
            if is_green:
                time_left = traffic_manager.get_time_remaining(current_time)
                status_text += f"{lane_num}: {signal} ({count} vehicles) ⏱️ {time_left}s remaining\n"
            else:
                status_text += f"{lane_num}: {signal} ({count} vehicles)\n"
                
        signal_display.write(status_text)
        
        # Update vehicle counts
        count_text = "### Vehicle Counts\n"
        for lane, count in lane_counts.items():
            count_text += f"{lane}: {count} vehicles\n"
        count_display.write(count_text)
        
        time.sleep(1/30)  # Control frame rate

if __name__ == "__main__":
    main()