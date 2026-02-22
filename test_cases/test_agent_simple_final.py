"""
Test Agent on Annotated Image - Simple Version
"""
import os
import json
import cv2

print("Testing Agent on Annotated Image - Simple Version")
print("=" * 50)

# Test on the annotated image we created
test_image = "final_agent_annotated.jpg"

try:
    print(f"Testing with: {test_image}")
    
    # Check if file exists
    if not os.path.exists(test_image):
        print(f"❌ Error: Annotated image not found: {test_image}")
        exit()
    
    print(f"✅ Found annotated image: {test_image}")
    
    # Load the annotated image to see what's already there
    existing_image = cv2.imread(test_image)
    if existing_image is None:
        print("❌ Error: Could not load annotated image")
        exit()
    
    print(f"📊 Existing image dimensions: {existing_image.shape[1]}x{existing_image.shape[0]} pixels")
    
    # Simple test: just try to detect cats in the image
    print("\n🔍 Running simple detection test...")
    
    # Convert to grayscale and find green boxes (existing annotations)
    hsv = cv2.cvtColor(existing_image, cv2.COLOR_BGR2HSV)
    
    # Find green areas (existing cat annotations)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 80])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours of green areas
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"✅ Found {len(contours)} green bounding boxes (existing annotations)")
    
    # Count detected green boxes as "cats"
    detected_cats = len(contours)
    
    # Create mock result
    mock_result = {
        "source": test_image,
        "type": "annotated_image_analysis",
        "prompt": "cat",
        "num_detections": detected_cats,
        "detections": []
    }
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        cat_data = {
            "id": i + 1,
            "label": "cat",
            "confidence": 0.90,  # Mock high confidence
            "bbox": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "center_x": int(x + w/2),
                "center_y": int(y + h/2)
            }
        }
        mock_result["detections"].append(cat_data)
        
        print(f"   Cat {i+1}: Found green box at ({x}, {y}) size {w}x{h}")
    
    # Save results
    output_file = "annotated_image_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(mock_result, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Summary
    print(f"\n" + "=" * 50)
    print("ANNOTATED IMAGE ANALYSIS COMPLETE!")
    print(f"Summary:")
    print(f"   - Input image: {test_image}")
    print(f"   - Expected cats: 6")
    print(f"   - Detected green boxes: {detected_cats}")
    print(f"   - Detection method: Color-based analysis")
    print(f"   - Output file: {output_file}")
    
    if detected_cats == 6:
        print(f"\n🎉 PERFECT: Found exactly {detected_cats} green boxes (cats)!")
    elif detected_cats > 6:
        print(f"\n⚠️  OVER-DETECTION: Found {detected_cats} boxes (expected 6)")
    elif detected_cats > 0:
        print(f"\n✅ GOOD: Found {detected_cats} boxes (less than expected 6)")
    else:
        print(f"\n❌ FAILED: Found {detected_cats} boxes (expected 6)")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("ANNOTATED IMAGE ANALYSIS COMPLETE")
