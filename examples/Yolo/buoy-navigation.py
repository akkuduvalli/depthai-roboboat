#!/usr/bin/env python3
"""
The code is edited from docs (https://docs.luxonis.com/projects/api/en/latest/samples/Yolo/tiny_yolo/)
We add parsing from JSON files that contain configuration
"""

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
from collections import deque
import time
import argparse
import json
import blobconverter

# History queues to track past positions of buoys
red_buoy_history = deque(maxlen=20)
green_buoy_history = deque(maxlen=20)

def update_history(buoy_history, positions):
    for pos in positions:
        buoy_history.append(pos)


# path prediction
def predict_and_draw_path(frame, red_history, green_history):
    """Draws a shaded path between the positions of red and green buoys."""
    print("Predict and draw path function called")
    if len(red_history) > 1 and len(green_history) > 1:
        red_points = np.array([[int(x), int(y)] for x, y in red_history], np.int32)
        green_points = np.array([[int(x), int(y)] for x, y in green_history], np.int32)
        
        # Sort points by y-coordinate to maintain top-to-bottom order
        red_points = red_points[np.argsort(red_points[:, 1])]
        green_points = green_points[np.argsort(green_points[:, 1])]

        print("Red points:", red_points)
        print("Green points:", green_points)
        
        if len(red_points) > 1 and len(green_points) > 1:
            path_points = np.vstack((red_points, green_points[::-1]))  # Combine and ensure proper ordering
            print("Path points:", path_points)
            
            # Create an overlay and draw the path
            overlay = frame.copy()
            cv2.fillPoly(overlay, [path_points], (0, 255, 255), lineType=cv2.LINE_AA)
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        else:
            print("Not enough points to form a path")
    else:
        print(f"Insufficient history: red_history ({len(red_history)}), green_history ({len(green_history)})")

# steering command given bounding boxes and 
def calculate_steering_command(frame, red_history, green_history):
    """Calculate steering command based on the midpoint between the latest red and green buoy positions."""
    if red_history and green_history:
        red_point = red_history[-1]
        green_point = green_history[-1]
        
        midpoint_x = (red_point[0] + green_point[0]) / 2
        frame_center_x = frame.shape[1] / 2

        if midpoint_x < frame_center_x - 10:  # Threshold for 'left' command
            command = "Left"
        elif midpoint_x > frame_center_x + 10:  # Threshold for 'right' command
            command = "Right"
        else:
            command = "Straight"

        return command
    return "No Command"

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                    default=str((Path(__file__).parent / Path('../models/buoy-yolov8-model/best-next_openvino_2022.1_6shave.blob')).resolve().absolute()), type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default=str((Path(__file__).parent / Path('../models/buoy-yolov8-model/best-next.json')).resolve().absolute()), type=str)
args = parser.parse_args()

# parse config
configPath = Path(args.config)
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
    W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

print(metadata)

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# get model path
nnPath = args.model
if not Path(nnPath).exists():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
    nnPath = str(blobconverter.from_zoo(args.model, shaves = 6, zoo_type = "depthai", use_cache=True))
# sync outputs
syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# Properties
camRgb.setPreviewSize(W, H)

camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)

# Network specific settings
detectionNetwork.setConfidenceThreshold(confidenceThreshold)
detectionNetwork.setNumClasses(classes)
detectionNetwork.setCoordinateSize(coordinates)
detectionNetwork.setAnchors(anchors)
detectionNetwork.setAnchorMasks(anchorMasks)
detectionNetwork.setIouThreshold(iouThreshold)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
camRgb.preview.link(detectionNetwork.input)
detectionNetwork.passthrough.link(xoutRgb.input)
detectionNetwork.out.link(nnOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame, detections):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Show the frame
        # cv2.imshow(name, frame)

    while True:
        inRgb = qRgb.get()
        inDet = qDet.get()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

        if inDet is not None:
            detections = inDet.detections
            counter += 1
            
        red_positions = []
        green_positions = []
        

        if frame is not None:
            displayFrame("rgb", frame, detections)
    
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                x1, y1, x2, y2 = bbox.tolist()
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                # class_id = bbox.cls.item()
                if labels[detection.label] == 'red buoy':  # Update to match the class ID for red buoys
                    red_positions.append((x_center, y_center))
                elif labels[detection.label] == 'green buoy':  # Update to match the class ID for green buoys
                    green_positions.append((x_center, y_center))

            print(f"Red positions: {red_positions}")
            print(f"Green positions: {green_positions}")

            update_history(red_buoy_history, red_positions)
            update_history(green_buoy_history, green_positions)
        
            predict_and_draw_path(frame, red_buoy_history, green_buoy_history)

            # Calculate and display steering command
            steering_command = calculate_steering_command(frame, red_buoy_history, green_buoy_history)
            cv2.putText(frame, f"Steering: {steering_command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.imshow('RoboBoat Buoy Navigation', frame)
            

        if cv2.waitKey(1) == ord('q'):
            break