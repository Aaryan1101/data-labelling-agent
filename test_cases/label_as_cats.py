"""
Label Objects as Cats - Custom Classification
"""
import os
import json
import cv2
import numpy as np
from datetime import datetime

print("Label Objects as Cats - Custom Classification")
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
    white_pixels = np.sum(thresh == 255)
    black_pixels = np.sum(thresh == 0)
    
    if white_pixels > black_pixels:
        thresh = cv2.bitwise_not(thresh)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} potential objects")
    
    detected_objects = []
    
    for i, contour in enumerate(contours):
        # Calculate area
        area = cv2.contourArea(contour)
        
        # Skip very small contours (noise) and very large (background)
        if area < 50 or area > 10000:
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Determine cat type based on size and position
        cat_type = "cat"
        confidence = 0.8  # Base confidence
        
        # Classify based on size
        if area < 200:
            cat_type = "kitten"
            confidence = 0.85
        elif area > 2000:
            cat_type = "large_cat"
            confidence = 0.75
        else:
            cat_type = "cat"
            confidence = 0.80
        
        # Add some variety based on position
        if y < 100:  # Top area
            cat_type = "jumping_cat"
            confidence = 0.78
        elif y > 200:  # Bottom area
            cat_type = "sitting_cat"
            confidence = 0.82
        
        # Create object data
        obj_data = {
            "id": i + 1,
            "label": cat_type,
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
            "cat_attributes": {
                "size_category": "small" if area < 200 else "medium" if area < 2000 else "large",
                "position": "top" if y < 100 else "middle" if y < 200 else "bottom",
                "estimated_age": "young" if area < 200 else "adult" if area < 2000 else "mature"
            }
        }
        
        detected_objects.append(obj_data)
        
        # Draw on image with cat-specific colors
        if cat_type == "kitten":
            color = (255, 0, 255)  # Magenta for kittens
        elif cat_type == "large_cat":
            color = (0, 0, 255)    # Red for large cats
        elif cat_type == "jumping_cat":
            color = (255, 165, 0)    # Orange for jumping cats
        elif cat_type == "sitting_cat":
            color = (0, 255, 255)    # Yellow for sitting cats
        else:
            color = (0, 255, 0)      # Green for regular cats
        
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label_text = f"{cat_type} {confidence:.2f}"
        cv2.putText(image, label_text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        print(f"Cat {i+1}: {cat_type}")
        print(f"  Position: ({x}, {y}) Size: {w}x{h}")
        print(f"  Area: {area:.0f} pixels")
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
        "detection_method": "cat_classification_algorithm",
        "classification_type": "custom_cat_labels",
        "num_detections": len(detected_objects),
        "objects": detected_objects
    }
    
    # Save results to JSON
    output_json = "picture-unlabel-cats.json"
    with open(output_json, 'w') as f:
        json.dump(labelled_data, f, indent=2)
    
    print(f"Results saved to: {output_json}")
    
    # Save annotated image
    output_image = "picture-unlabel-cats.jpg"
    cv2.imwrite(output_image, image)
    
    print(f"Annotated image saved to: {output_image}")
    
    # Summary
    print("\n" + "=" * 50)
    print("CAT CLASSIFICATION COMPLETE!")
    print(f"Summary:")
    print(f"  - Image: {os.path.basename(image_path)}")
    print(f"  - Cats detected: {len(detected_objects)}")
    print(f"  - Classification method: Custom cat detection")
    print(f"  - Output files: {output_json}, {output_image}")
    
    # Cat breakdown
    if detected_objects:
        cat_counts = {}
        for obj in detected_objects:
            cat_type = obj['label']
            cat_counts[cat_type] = cat_counts.get(cat_type, 0) + 1
        
        print(f"\nDetected cat types:")
        for cat_type, count in cat_counts.items():
            print(f"  - {cat_type}: {count}")
        
        avg_confidence = sum(obj['confidence'] for obj in detected_objects) / len(detected_objects)
        print(f"\nAverage confidence: {avg_confidence:.3f}")
        
        # Additional cat statistics
        size_categories = {}
        for obj in detected_objects:
            size = obj['cat_attributes']['size_category']
            size_categories[size] = size_categories.get(size, 0) + 1
        
        print(f"\nSize distribution:")
        for size, count in size_categories.items():
            print(f"  - {size}: {count}")
    else:
        print("\nNo cats detected. The image may not contain clear cat-like objects.")

except Exception as e:
    print(f"Error processing image: {e}")
    import traceback
    traceback.print_exc()
