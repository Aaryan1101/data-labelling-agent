"""
Single Image Autodistill Test - Label One Image
"""
import os
import json
import cv2
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import supervision as sv

# Set API key
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCzA8Uf_xfghL2ypS6EhuCHIy1eI2zzdt8'

print("Single Image Autodistill Test")
print("=" * 40)

# Test image and label
test_image = "vision data/auto_label_dataset/circle_01.png"
label_name = "circle"

try:
    print(f"📸 Image: {test_image}")
    print(f"🏷️  Label: {label_name}")
    print()
    
    # Create ontology (mapping text prompt to class name)
    ontology = CaptionOntology({label_name: label_name})
    
    # Initialize Grounding DINO model
    print("🔄 Loading Grounding DINO model...")
    base_model = GroundingDINO(ontology=ontology)
    
    # Run detection
    print("🔍 Detecting objects...")
    results = base_model.predict(test_image)
    
    # Process results
    print(f"✅ Detection complete!")
    print(f"   Objects found: {len(results.xyxy)}")
    
    # Create labeled output
    labeled_data = {
        "image_path": test_image,
        "detection_method": "grounding_dino",
        "label": label_name,
        "num_detections": len(results.xyxy),
        "objects": []
    }
    
    # Process each detection
    for i in range(len(results.xyxy)):
        x1, y1, x2, y2 = results.xyxy[i]
        confidence = results.confidence[i]
        class_name = results.class_names[i]
        
        obj_data = {
            "id": i + 1,
            "label": class_name,
            "confidence": float(confidence),
            "bbox": {
                "x1": float(x1),
                "y1": float(y1), 
                "x2": float(x2),
                "y2": float(y2),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
                "center_x": float((x1 + x2) / 2),
                "center_y": float((y1 + y2) / 2)
            }
        }
        
        labeled_data["objects"].append(obj_data)
        
        print(f"   Object {i+1}: {class_name}")
        print(f"     Confidence: {confidence:.3f}")
        print(f"     Bounding Box: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
        print(f"     Size: {int(x2-x1)} x {int(y2-y1)} pixels")
    
    # Save results
    output_file = "single_image_labelled.json"
    with open(output_file, 'w') as f:
        json.dump(labeled_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Create annotated image
    print("🎨 Creating annotated image...")
    
    # Load original image
    image = cv2.imread(test_image)
    
    # Draw bounding boxes
    for i in range(len(results.xyxy)):
        x1, y1, x2, y2 = map(int, results.xyxy[i])
        confidence = results.confidence[i]
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label_text = f"{label_name} {confidence:.2f}"
        cv2.putText(image, label_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save annotated image
    annotated_file = "single_image_annotated.jpg"
    cv2.imwrite(annotated_file, image)
    print(f"🖼️  Annotated image saved to: {annotated_file}")
    
    # Summary
    print("\n" + "=" * 40)
    print("🎉 SINGLE IMAGE LABELLING COMPLETE!")
    print(f"📊 Summary:")
    print(f"   - Image processed: {os.path.basename(test_image)}")
    print(f"   - Objects detected: {len(results.xyxy)}")
    print(f"   - Label assigned: {label_name}")
    print(f"   - Average confidence: {sum(results.confidence)/len(results.confidence):.3f}" if len(results.confidence) > 0 else "   - Average confidence: N/A")
    print(f"   - Output files: {output_file}, {annotated_file}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
