"""
Test Agent with Simple Function Call (No @tool decorator)
"""
import os
import json
import cv2

print("Testing Agent with Simple Function Call")
print("=" * 50)

# Simple function without @tool decorator
def simple_grounding_dino_detect(file_path: str, text_prompt: str, 
                                box_threshold: float = 0.35,
                                text_threshold: float = 0.25):
    """
    Simple Grounding DINO detection function.
    """
    try:
        from autodistill_grounding_dino import GroundingDINO
        from autodistill.detection import CaptionOntology
        
        labels = [l.strip().rstrip(".").strip() for l in text_prompt.split(".") if l.strip()]
        ontology = CaptionOntology({label: label for label in labels})
        
        base_model = GroundingDINO(ontology=ontology)
        results = base_model.predict(file_path)
        
        detections = []
        for i in range(len(results.xyxy)):
            x1, y1, x2, y2 = results.xyxy[i]
            confidence = results.confidence[i]
            
            # Get class name - handle different result formats
            if hasattr(results, 'class_names'):
                class_name = results.class_names[i]
            elif hasattr(results, 'classes'):
                class_name = results.classes[i]
            else:
                class_name = labels[0] if labels else "object"
            
            if confidence >= box_threshold:
                detections.append({
                    "label": class_name,
                    "confidence": float(confidence),
                    "bbox": {
                        "x": float(x1),
                        "y": float(y1),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                        "center_x": float((x1 + x2) / 2),
                        "center_y": float((y1 + y2) / 2)
                    }
                })
        
        return {
            "source": file_path,
            "type": "grounding_dino_detection",
            "prompt": text_prompt,
            "num_detections": len(detections),
            "detections": detections,
        }
        
    except ImportError:
        return {"error": "Install: pip install autodistill-grounding-dino supervision"}
    except Exception as e:
        return {"error": f"Grounding DINO detection failed: {str(e)}"}

# Test the simple function
test_image = "picture-unlabel.webp"
text_prompt = "cat"

try:
    print(f"Testing with: {test_image}")
    print(f"Text prompt: {text_prompt}")
    
    # Call the simple function directly
    result = simple_grounding_dino_detect(
        file_path=test_image,
        text_prompt=text_prompt,
        box_threshold=0.2,
        text_threshold=0.25
    )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Detection successful!")
        print(f"   Method: {result['type']}")
        print(f"   Objects found: {result['num_detections']}")
        
        for i, obj in enumerate(result['detections']):
            print(f"   Object {i+1}: {obj['label']} (confidence: {obj['confidence']:.3f})")
            print(f"     Bounding Box: ({obj['bbox']['x']}, {obj['bbox']['y']}) to ({obj['bbox']['x'] + obj['bbox']['width']}, {obj['bbox']['y'] + obj['bbox']['height']})")
        
        # Save results
        with open('simple_agent_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✅ Results saved to: simple_agent_results.json")
        
        # Create annotated image
        image = cv2.imread(test_image)
        for obj in result['detections']:
            x, y, w, h = int(obj['bbox']['x']), int(obj['bbox']['y']), int(obj['bbox']['width']), int(obj['bbox']['height'])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label_text = f"Cat {obj['label']} {obj['confidence']:.2f}"
            cv2.putText(image, label_text, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite('simple_agent_annotated.jpg', image)
        print(f"✅ Annotated image saved to: simple_agent_annotated.jpg")
        
        # Summary
        print(f"\n🎯 SIMPLE AGENT RESULTS:")
        print(f"   - Objects detected: {result['num_detections']}")
        print(f"   - Method: Direct function call")
        print(f"   - No @tool decorator issues")
        
        if result['num_detections'] >= 4:
            print(f"   ✅ Agent working well!")
        else:
            print(f"   ⚠️  Agent working but detecting fewer objects")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("SIMPLE AGENT TEST COMPLETE")
