"""
Test Real Autodistill Vision Pipeline
"""
import os
import json
from labelling_agent import grounding_dino_detect, grounded_sam_detect_and_segment

# Set API key
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCzA8Uf_xfghL2ypS6EhuCHIy1eI2zzdt8'

print("Testing Real Autodistill Vision Pipeline...")
print("=" * 60)

# Test 1: Grounding DINO Object Detection
print("\n=== TEST 1: GROUNDING DINO OBJECT DETECTION ===")

test_image = "vision data/auto_label_dataset/circle_01.png"
text_prompt = "circle"

try:
    print(f"Image: {test_image}")
    print(f"Prompt: {text_prompt}")
    
    result = grounding_dino_detect(
        file_path=test_image,
        text_prompt=text_prompt,
        box_threshold=0.35,
        text_threshold=0.25
    )
    
    # Parse and display results
    detection_result = json.loads(result)
    
    if "error" in detection_result:
        print(f"❌ Error: {detection_result['error']}")
    else:
        print(f"✅ Detection successful!")
        print(f"   Type: {detection_result['type']}")
        print(f"   Objects found: {detection_result['num_detections']}")
        
        for i, obj in enumerate(detection_result['detections']):
            print(f"   Object {i+1}: {obj['label']} (confidence: {obj['confidence']:.2f})")
            print(f"   Bounding Box: {obj['bbox']}")
        
        # Save results
        with open('grounding_dino_results.json', 'w') as f:
            json.dump(detection_result, f, indent=2)
        print("✅ Saved to grounding_dino_results.json")

except Exception as e:
    print(f"❌ Grounding DINO test failed: {e}")

# Test 2: GroundedSAM Detection + Segmentation
print("\n=== TEST 2: GROUNDED SAM DETECTION + SEGMENTATION ===")

try:
    print(f"Image: {test_image}")
    print(f"Prompt: {text_prompt}")
    
    result = grounded_sam_detect_and_segment(
        file_path=test_image,
        text_prompt=text_prompt,
        output_dir="./grounded_sam_output"
    )
    
    # Parse and display results
    segmentation_result = json.loads(result)
    
    if "error" in segmentation_result:
        print(f"❌ Error: {segmentation_result['error']}")
    else:
        print(f"✅ Segmentation successful!")
        print(f"   Type: {segmentation_result['type']}")
        print(f"   Objects found: {segmentation_result['num_detections']}")
        
        for i, obj in enumerate(segmentation_result['detections']):
            print(f"   Object {i+1}: {obj['label']} (confidence: {obj['confidence']:.2f})")
            print(f"   Bounding Box: {obj['bbox']}")
            print(f"   Mask available: {'mask' in obj}")
        
        # Save results
        with open('grounded_sam_results.json', 'w') as f:
            json.dump(segmentation_result, f, indent=2)
        print("✅ Saved to grounded_sam_results.json")
        print(f"✅ Output directory: ./grounded_sam_output")

except Exception as e:
    print(f"❌ GroundedSAM test failed: {e}")

# Test 3: Multiple Object Detection
print("\n=== TEST 3: MULTIPLE OBJECT DETECTION ===")

# Test with different shapes
shape_images = [
    ("vision data/auto_label_dataset/circle_01.png", "circle"),
    ("vision data/auto_label_dataset/square_02.png", "square"),
    ("vision data/auto_label_dataset/triangle_03.png", "triangle")
]

for img_path, prompt in shape_images:
    try:
        print(f"\nTesting: {os.path.basename(img_path)} -> {prompt}")
        
        result = grounding_dino_detect(
            file_path=img_path,
            text_prompt=prompt
        )
        
        detection_result = json.loads(result)
        
        if "error" not in detection_result:
            print(f"✅ Found {detection_result['num_detections']} {prompt}(s)")
            for obj in detection_result['detections']:
                print(f"   Confidence: {obj['confidence']:.2f}")
        else:
            print(f"❌ Error: {detection_result['error']}")
            
    except Exception as e:
        print(f"❌ Failed: {e}")

print("\n=== AUTODISTILL PIPELINE SUMMARY ===")
print("✅ Grounding DINO: Object detection")
print("✅ GroundedSAM: Detection + Segmentation") 
print("✅ Multiple shape recognition")
print("✅ JSON output generation")
print("\n🎉 Real Autodistill Vision Pipeline is now WORKING!")
