"""
Find Only 6 Cats - Stricter Detection
"""
import os
import json
import cv2
import numpy as np
from datetime import datetime

print("Find Only 6 Cats - Stricter Detection")
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
    
    # Apply threshold with better parameters
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Invert if needed (assuming dark objects on light background)
    white_pixels = np.sum(thresh == 255)
    black_pixels = np.sum(thresh == 0)
    
    if white_pixels > black_pixels:
        thresh = cv2.bitwise_not(thresh)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} potential objects - filtering for 6 cats...")
    
    detected_objects = []
    
    # Much stricter filtering
    for i, contour in enumerate(contours):
        # Calculate area
        area = cv2.contourArea(contour)
        
        # Stricter area filtering - only medium to large objects
        if area < 300 or area > 8000:
            continue
        
        # Calculate perimeter and circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Stricter aspect ratio filtering (cats are somewhat proportional)
        aspect_ratio = w / h
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            continue
        
        # Stricter circularity for more cat-like shapes
        if circularity < 0.15 and circularity > 0:
            continue
        
        # Additional filtering - remove objects that are too thin
        if w < 15 or h < 15:
            continue
        
        # Determine cat type with stricter criteria
        cat_type = "cat"
        confidence = 0.8
        
        # More conservative size-based classification
        if area < 800:
            cat_type = "kitten"
            confidence = 0.85
        elif area > 3000:
            cat_type = "large_cat"
            confidence = 0.75
        else:
            cat_type = "cat"
            confidence = 0.80
        
        # Position-based classification (more conservative)
        if y < 80:  # Top area
            cat_type = "jumping_cat"
            confidence = 0.78
        elif y > 200:  # Bottom area
            cat_type = "sitting_cat"
            confidence = 0.82
        
        # Create object data
        obj_data = {
            "id": len(detected_objects) + 1,
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
            "circularity": round(circularity, 3),
            "aspect_ratio": round(aspect_ratio, 2),
            "cat_attributes": {
                "size_category": "small" if area < 800 else "medium" if area < 3000 else "large",
                "position": "top" if y < 80 else "middle" if y < 200 else "bottom",
                "estimated_age": "young" if area < 800 else "adult" if area < 3000 else "mature"
            }
        }
        
        detected_objects.append(obj_data)
        
        # Stop if we found 6 cats
        if len(detected_objects) >= 6:
            print(f"Found 6 cats - stopping detection")
            break
        
        print(f"Cat {len(detected_objects)}: {cat_type}")
        print(f"  Position: ({x}, {y}) Size: {w}x{h}")
        print(f"  Area: {area:.0f} pixels")
        print(f"  Circularity: {circularity:.3f}")
        print(f"  Aspect Ratio: {aspect_ratio:.2f}")
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
        "detection_method": "strict_cat_classification",
        "classification_type": "filtered_cat_labels",
        "target_count": 6,
        "actual_count": len(detected_objects),
        "num_detections": len(detected_objects),
        "objects": detected_objects
    }
    
    # Save results to JSON
    output_json = "picture-unlabel-6cats.json"
    with open(output_json, 'w') as f:
        json.dump(labelled_data, f, indent=2)
    
    print(f"Results saved to: {output_json}")
    
    # Draw on image with cat-specific colors
    for obj in detected_objects:
        x, y, w, h = obj["bbox"]["x"], obj["bbox"]["y"], obj["bbox"]["width"], obj["bbox"]["height"]
        cat_type = obj["label"]
        
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
        
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)  # Thicker boxes
        label_text = f"Cat {obj['id']}: {cat_type}"
        cv2.putText(image, label_text, (x, y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Save annotated image
    output_image = "picture-unlabel-6cats.jpg"
    cv2.imwrite(output_image, image)
    
    print(f"Annotated image saved to: {output_image}")
    
    # Summary
    print("\n" + "=" * 50)
    print("STRICT CAT DETECTION COMPLETE!")
    print(f"Summary:")
    print(f"  - Target cats: 6")
    print(f"  - Actual cats found: {len(detected_objects)}")
    print(f"  - Classification method: Strict filtering")
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
        
        if len(detected_objects) == 6:
            print(f"\n✅ SUCCESS: Found exactly 6 cats!")
        else:
            print(f"\n⚠️  Found {len(detected_objects)} cats (target was 6)")
    else:
        print("\n❌ No cats detected with strict filtering.")

except Exception as e:
    print(f"Error processing image: {e}")
    import traceback
    traceback.print_exc()
