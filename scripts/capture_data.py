# scripts/capture_data.py

import os
import sys
import numpy as np
import cv2
from datetime import datetime
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.data.realsense_capture import RealSenseCapture

def create_capture_session():
    """Create a new capture session directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(project_root, 'data', 'raw', f'session_{timestamp}')
    
    os.makedirs(os.path.join(session_dir, 'rgb'))
    os.makedirs(os.path.join(session_dir, 'depth'))
    
    return session_dir

def main(args):
    session_dir = create_capture_session()

    camera = RealSenseCapture(
        width=args.width,
        height=args.height,
        fps=args.fps
    ).start()

    camera.save_intrinsics(os.path.join(session_dir, 'intrinsics.json'))
    
    frame_count = 0
    try:
        while True:
            color_image, depth_image = camera.capture_frame()
            
            if color_image is None or depth_image is None:
                continue
            
            cv2.imshow('RGB-D Capture', np.hstack((
                color_image,
                cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            )))
            
            key = cv2.waitKey(1)
            
            if key == 32:  
                cv2.imwrite(os.path.join(session_dir, 'rgb', f'{frame_count:06d}.png'), color_image)
                np.save(os.path.join(session_dir, 'depth', f'{frame_count:06d}.npy'), depth_image)
                print(f'Captured frame {frame_count}')
                frame_count += 1
            
            elif key == 27:  # ESC
                break
                
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture RGB-D data from RealSense camera')
    parser.add_argument('--width', type=int, default=1280, help='Camera width')
    parser.add_argument('--height', type=int, default=720, help='Camera height')
    parser.add_argument('--fps', type=int, default=30, help='Camera FPS')
    
    args = parser.parse_args()
    main(args)