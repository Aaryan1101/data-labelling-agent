"""
Correct Circle Detection - Fix the Shape Recognition
"""
import os
import json
import cv2
import numpy as np

print("Correct Circle Detection")
print("=" * 40)

# Test image
test_image = "vision data/auto_label_dataset/circle_01.png"

try:
    print(f"📸 Image: {test_image}")
    
    # Load image
    image = cv2.imread(test_image)
    if image is None:
        print("❌ Failed to load image")
        exit()
    
    print(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to separate black shape from white background
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Invert so black shape becomes white (for contour detection)
    thresh = cv2.bitwise_not(thresh)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"🔍 Found {len(contours)} objects")
    
    detected_objects = []
    
    for i, contour in enumerate(contours):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate area
        area = cv2.contourArea(contour)
        
        # Skip very small contours (noise)
        if area < 500:
            continue
        
        # Calculate circularity (how close to a perfect circle)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Determine shape with better logic
        shape = "unknown"
        confidence = 0.0
        
        # Check for circle (circularity close to 1.0)
        if circularity > 0.8:
            shape = "circle"
            confidence = min(0.95, circularity)
        else:
            # Approximate contour for other shapes
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
                shape = "circle"  # Fallback for high-vertex shapes
                confidence = 0.75
            else:
                shape = "polygon"
                confidence = 0.70
        
        # Create object data
        obj_data = {
            "id": i + 1,
            "label": shape,
            "confidence": round(confidence, 3),
            "bbox": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "center_x": int(x + w/2),
                "center_y": int(y + h/2)
            },
            "area": float(area),
            "circularity": round(circularity, 3),
            "perimeter": float(perimeter)
        }
        
        detected_objects.append(obj_data)
        
        # Draw on image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{shape} {confidence:.2f}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print(f"   Object {i+1}: {shape}")
        print(f"     Position: ({x}, {y}) Size: {w}x{h}")
        print(f"     Area: {area:.0f} pixels")
        print(f"     Circularity: {circularity:.3f} (closer to 1.0 = more circular)")
        print(f"     Confidence: {confidence:.3f}")
    
    # Create labeled data
    labeled_data = {
        "image_path": test_image,
        "detection_method": "opencv_circularity_analysis",
        "image_dimensions": {
            "width": image.shape[1],
            "height": image.shape[0],
            "channels": image.shape[2]
        },
        "num_detections": len(detected_objects),
        "objects": detected_objects
    }
    
    # Save results
    output_file = "circle_detection_corrected.json"
    with open(output_file, 'w') as f:
        json.dump(labeled_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Save annotated image
    annotated_file = "circle_detection_corrected.jpg"
    cv2.imwrite(annotated_file, image)
    print(f"🖼️  Annotated image saved to: {annotated_file}")
    
    # Summary
    print("\n" + "=" * 40)
    print("🎉 CORRECTED CIRCLE DETECTION!")
    print(f"📊 Summary:")
    print(f"   - Image processed: {os.path.basename(test_image)}")
    print(f"   - Objects detected: {len(detected_objects)}")
    print(f"   - Detection method: Circularity analysis")
    print(f"   - Output files: {output_file}, {annotated_file}")
    
    # Show detected shapes
    shape_counts = {}
    for obj in detected_objects:
        shape = obj['label']
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    
    print(f"\n🔢 Shape breakdown:")
    for shape, count in shape_counts.items():
        print(f"   - {shape}: {count}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
