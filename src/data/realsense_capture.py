# src/data/realsense_capture.py

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime
import json

class RealSenseCapture:
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        # Align object to align depth to color frame
        self.align = rs.align(rs.stream.color)

    def start(self):
        """Start the RealSense pipeline"""
        self.profile = self.pipeline.start(self.config)
        
        # Get depth scale for depth -> distance conversion
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Get camera intrinsics
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        return self

    def capture_frame(self):
        """Capture a single aligned RGB-D frame"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None
            
        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image, depth_image

    def save_intrinsics(self, save_path):
        """Save camera intrinsics to JSON"""
        intrinsics_dict = {
            'width': self.intrinsics.width,
            'height': self.intrinsics.height,
            'ppx': self.intrinsics.ppx,
            'ppy': self.intrinsics.ppy,
            'fx': self.intrinsics.fx,
            'fy': self.intrinsics.fy,
            'model': str(self.intrinsics.model),
            'coeffs': self.intrinsics.coeffs,
            'depth_scale': self.depth_scale
        }
        
        with open(save_path, 'w') as f:
            json.dump(intrinsics_dict, f, indent=4)

    def stop(self):
        """Stop the RealSense pipeline"""
        self.pipeline.stop()