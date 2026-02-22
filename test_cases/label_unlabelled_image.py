"""
Label the Unlabelled Image: picture-unlabel.webp
"""
import os
import json
import cv2
import numpy as np
from datetime import datetime

print("Labelling Unlabelled Image: picture-unlabel.webp")
print("=" * 50)

# Path to the unlabelled image
image_path = "picture-unlabel.webp"

try:
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        exit()
    
    print(f"Loading image: {image_path}")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image file")
        exit()
    
    print(f"Image loaded successfully: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to separate objects from background
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Invert if needed (assuming dark objects on light background)
    # Check if most pixels are white or black
    white_pixels = np.sum(thresh == 255)
    black_pixels = np.sum(thresh == 0)
    
    if white_pixels > black_pixels:
        # Most pixels are white, invert to detect dark objects
        thresh = cv2.bitwise_not(thresh)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} potential objects")
    
    detected_objects = []
    
    for i, contour in enumerate(contours):
        # Calculate area
        area = cv2.contourArea(contour)
        
        # Skip very small contours (noise)
        if area < 100:
            continue
        
        # Calculate perimeter and circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Determine shape
        shape = "unknown"
        confidence = 0.0
        
        if circularity > 0.8:
            shape = "circle"
            confidence = min(0.95, circularity)
        else:
            # Approximate contour for polygon analysis
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices = len(approx)
            
            if vertices == 3:
                shape = "triangle"
                confidence = 0.85
            elif vertices == 4:
                aspect_ratio = w / h
                if 0.8 <= aspect_ratio <= 1.2:
                    shape = "square"
                    confidence = 0.90
                else:
                    shape = "rectangle"
                    confidence = 0.85
            elif vertices > 8:
                shape = "circle"  # High-vertex shapes as circles
                confidence = 0.75
            else:
                shape = "polygon"
                confidence = 0.70
        
        # Create object data
        obj_data = {
            "id": i + 1,
            "label": shape,
            "confidence": round(confidence, 3),
            "circularity": round(circularity, 3),
            "vertices": vertices if 'vertices' in locals() else 0,
            "bbox": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "center_x": int(x + w/2),
                "center_y": int(y + h/2)
            },
            "area": float(area),
            "perimeter": float(perimeter) if perimeter > 0 else 0.0
        }
        
        detected_objects.append(obj_data)
        
        # Draw on image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label_text = f"{shape} {confidence:.2f}"
        cv2.putText(image, label_text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        print(f"Object {i+1}: {shape}")
        print(f"  Position: ({x}, {y}) Size: {w}x{h}")
        print(f"  Area: {area:.0f} pixels")
        print(f"  Circularity: {circularity:.3f}")
        print(f"  Confidence: {confidence:.3f}")
        print()
    
    # Create labelled data structure
    labelled_data = {
        "image_path": image_path,
        "filename": os.path.basename(image_path),
        "timestamp": datetime.now().isoformat(),
        "image_dimensions": {
            "width": image.shape[1],
            "height": image.shape[0],
            "channels": image.shape[2]
        },
        "detection_method": "opencv_contour_analysis",
        "num_detections": len(detected_objects),
        "objects": detected_objects
    }
    
    # Save results to JSON
    output_json = "picture-unlabel-labelled.json"
    with open(output_json, 'w') as f:
        json.dump(labelled_data, f, indent=2)
    
    print(f"Results saved to: {output_json}")
    
    # Save annotated image
    output_image = "picture-unlabel-labelled.jpg"
    cv2.imwrite(output_image, image)
    
    print(f"Annotated image saved to: {output_image}")
    
    # Summary
    print("\n" + "=" * 50)
    print("IMAGE LABELLING COMPLETE!")
    print(f"Summary:")
    print(f"  - Image: {os.path.basename(image_path)}")
    print(f"  - Objects detected: {len(detected_objects)}")
    print(f"  - Processing method: OpenCV contour analysis")
    print(f"  - Output files: {output_json}, {output_image}")
    
    # Object breakdown
    if detected_objects:
        shape_counts = {}
        for obj in detected_objects:
            shape = obj['label']
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        
        print(f"\nDetected shapes:")
        for shape, count in shape_counts.items():
            print(f"  - {shape}: {count}")
        
        avg_confidence = sum(obj['confidence'] for obj in detected_objects) / len(detected_objects)
        print(f"\nAverage confidence: {avg_confidence:.3f}")
    else:
        print("\nNo objects detected. The image may be too uniform or require different processing.")

except Exception as e:
    print(f"Error processing image: {e}")
    import traceback
    traceback.print_exc()
