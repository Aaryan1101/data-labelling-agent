"""
Test Final Fixed Labelling Agent
"""
import os
import json
import cv2

print("Testing Final Fixed Labelling Agent")
print("=" * 50)

# Test image
test_image = "picture-unlabel.webp"
text_prompt = "cat"

try:
    print(f"Testing with: {test_image}")
    print(f"Text prompt: {text_prompt}")
    
    # Import the fixed agent function
    from labelling_agent import grounding_dino_detect
    
    # Call the fixed function
    result = grounding_dino_detect(
        file_path=test_image,
        text_prompt=text_prompt,
        box_threshold=0.2,  # Lower threshold for more detections
        text_threshold=0.25
    )
    
    # Parse and display results
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Detection successful!")
        print(f"   Method: {result['type']}")
        print(f"   Objects found: {result['num_detections']}")
        print(f"   Model method: {result['model_info']['method']}")
        
        for i, obj in enumerate(result['detections']):
            print(f"   Object {i+1}: {obj['label']} (confidence: {obj['confidence']:.3f})")
            print(f"     Bounding Box: ({obj['bbox']['x']}, {obj['bbox']['y']}) to ({obj['bbox']['x'] + obj['bbox']['width']}, {obj['bbox']['y'] + obj['bbox']['height']})")
            print(f"     Size: {obj['bbox']['width']}x{obj['bbox']['height']} pixels")
        
        # Save results
        with open('final_agent_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✅ Results saved to: final_agent_results.json")
        
        # Create annotated image
        image = cv2.imread(test_image)
        for obj in result['detections']:
            x, y, w, h = int(obj['bbox']['x']), int(obj['bbox']['y']), int(obj['bbox']['width']), int(obj['bbox']['height'])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Thicker green box
            label_text = f"Cat {obj['label']} {obj['confidence']:.2f}"
            cv2.putText(image, label_text, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite('final_agent_annotated.jpg', image)
        print(f"✅ Annotated image saved to: final_agent_annotated.jpg")
        
        # Summary
        print(f"\n🎯 FINAL AGENT RESULTS:")
        print(f"   - Detection method: {result['type']}")
        print(f"   - Objects detected: {result['num_detections']}")
        print(f"   - Model: Autodistill Grounding DINO")
        print(f"   - Confidence threshold: {result['model_info']['confidence_threshold']}")
        
        if result['num_detections'] >= 6:
            print(f"   🎉 SUCCESS: Agent is working perfectly!")
        elif result['num_detections'] >= 4:
            print(f"   ✅ GOOD: Agent working well!")
        else:
            print(f"   ⚠️  Agent working but detecting fewer objects")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("FINAL AGENT TEST COMPLETE")
