"""
Test Labelling Agent on Annotated Image
"""
import os
import json
import cv2
from labelling_agent import grounding_dino_detect

print("Testing Labelling Agent on Annotated Image")
print("=" * 50)

# Test on the annotated image we created
test_image = "final_agent_annotated.jpg"
text_prompt = "cat"

try:
    print(f"Testing with: {test_image}")
    print(f"Text prompt: {text_prompt}")
    
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
    
    # Call the agent function on the annotated image
    print("\n🔍 Running agent detection on annotated image...")
    result = grounding_dino_detect(
        file_path=test_image,
        text_prompt=text_prompt,
        box_threshold=0.2,  # Lower threshold to see if it finds existing boxes
        text_threshold=0.25
    )
    
    # Parse and display results
    detection_result = json.loads(result)
    
    if "error" in detection_result:
        print(f"❌ Error: {detection_result['error']}")
    else:
        print(f"✅ Detection successful!")
        print(f"   Method: {detection_result['type']}")
        print(f"   Objects found: {detection_result['num_detections']}")
        print(f"   Model method: {detection_result.get('model_info', {}).get('method', 'unknown')}")
        
        # Compare with what we expect
        expected_cats = 6
        found_cats = detection_result['num_detections']
        
        print(f"\n📊 COMPARISON:")
        print(f"   Expected cats: {expected_cats}")
        print(f"   Found cats: {found_cats}")
        
        if found_cats == expected_cats:
            print(f"   🎯 PERFECT: Found exactly {expected_cats} cats!")
        elif found_cats > expected_cats:
            print(f"   ⚠️  OVER-DETECTION: Found {found_cats} cats (expected {expected_cats})")
        else:
            print(f"   ⚠️  UNDER-DETECTION: Found {found_cats} cats (expected {expected_cats})")
        
        # Show detection details
        for i, obj in enumerate(detection_result['detections']):
            print(f"\n   Cat {i+1}:")
            print(f"     Label: {obj['label']}")
            print(f"     Confidence: {obj['confidence']:.3f}")
            print(f"     Bounding Box: ({obj['bbox']['x']}, {obj['bbox']['y']}) to ({obj['bbox']['x'] + obj['bbox']['width']}, {obj['bbox']['y'] + obj['bbox']['height']})")
        
        # Save results
        output_file = "agent_on_annotated_results.json"
        with open(output_file, 'w') as f:
            json.dump(detection_result, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Create new annotated image (different color)
        new_annotated = existing_image.copy()
        for obj in detection_result['detections']:
            x, y, w, h = int(obj['bbox']['x']), int(obj['bbox']['y']), int(obj['bbox']['width']), int(obj['bbox']['height'])
            cv2.rectangle(new_annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red boxes
            label_text = f"Detected {obj['label']} {obj['confidence']:.2f}"
            cv2.putText(new_annotated, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        new_output_file = "agent_on_annotated_red.jpg"
        cv2.imwrite(new_output_file, new_annotated)
        print(f"✅ New annotated image saved to: {new_output_file}")
        
        # Summary
        print(f"\n" + "=" * 50)
        print("AGENT ON ANNOTATED IMAGE TEST COMPLETE!")
        print(f"Summary:")
        print(f"   - Input image: {test_image}")
        print(f"   - Expected cats: {expected_cats}")
        print(f"   - Detected cats: {found_cats}")
        print(f"   - Detection method: {detection_result['type']}")
        print(f"   - Output files: {output_file}, {new_output_file}")
        
        if found_cats == expected_cats:
            print(f"\n🎉 SUCCESS: Agent correctly identified all {expected_cats} cats!")
        elif found_cats > 0:
            print(f"\n✅ Agent working but found {found_cats} vs expected {expected_cats}")
        else:
            print(f"\n❌ Agent failed to detect any cats")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("AGENT ON ANNOTATED IMAGE TEST COMPLETE")
