"""
Test the Fixed Labelling Agent with Autodistill
"""
import os
import json
from labelling_agent import grounding_dino_detect

print("Testing Fixed Labelling Agent")
print("=" * 40)

# Test image
test_image = "picture-unlabel.webp"
text_prompt = "cat"

try:
    print(f"Testing with: {test_image}")
    print(f"Text prompt: {text_prompt}")
    
    # Call the fixed agent function
    result = grounding_dino_detect(
        file_path=test_image,
        text_prompt=text_prompt,
        box_threshold=0.2,  # Lower threshold for testing
        text_threshold=0.25
    )
    
    # Parse and display results
    detection_result = json.loads(result)
    
    if "error" in detection_result:
        print(f"❌ Error: {detection_result['error']}")
    else:
        print(f"✅ Detection successful!")
        print(f"   Method: {detection_result.get('type', 'unknown')}")
        print(f"   Objects found: {detection_result['num_detections']}")
        
        for i, obj in enumerate(detection_result['detections']):
            print(f"   Object {i+1}: {obj['label']} (confidence: {obj['confidence']:.3f})")
            print(f"     Bounding Box: ({obj['bbox']['x']}, {obj['bbox']['y']}) to ({obj['bbox']['x'] + obj['bbox']['width']}, {obj['bbox']['y'] + obj['bbox']['height']})")
            print(f"     Size: {obj['bbox']['width']}x{obj['bbox']['height']} pixels")
        
        # Save results
        with open('agent_test_results.json', 'w') as f:
            json.dump(detection_result, f, indent=2)
        
        print(f"\n✅ Results saved to: agent_test_results.json")
        
        # Summary
        print(f"\n🎯 AGENT TEST RESULTS:")
        print(f"   - Detection method: {detection_result.get('type', 'unknown')}")
        print(f"   - Objects detected: {detection_result['num_detections']}")
        print(f"   - Model info: {detection_result.get('model_info', 'N/A')}")
        
        if detection_result['num_detections'] >= 4:
            print(f"   ✅ Agent is working well!")
        elif detection_result['num_detections'] >= 2:
            print(f"   ⚠️  Agent working but missing some objects")
        else:
            print(f"   ❌ Agent not detecting properly")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 40)
print("AGENT TEST COMPLETE")
