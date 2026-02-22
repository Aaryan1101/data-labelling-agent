"""
Test Autodistill After Version Fix
"""
import os
import json
import cv2

print("Testing Autodistill After Version Fix")
print("=" * 50)

# Check versions
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
except ImportError:
    print("PyTorch: Not installed")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError:
    print("Transformers: Not installed")

print("\nTesting Autodistill Grounding DINO...")
print("-" * 40)

try:
    from autodistill_grounding_dino import GroundingDINO
    from autodistill.detection import CaptionOntology
    
    print("✅ Autodistill Grounding DINO imported successfully!")
    
    # Test with the cat image
    test_image = "picture-unlabel.webp"
    if os.path.exists(test_image):
        print(f"Testing with: {test_image}")
        
        # Create ontology for cat detection
        ontology = CaptionOntology({"cat": "cat"})
        
        print("Initializing Grounding DINO model...")
        print("(This may take a moment to download models...)")
        
        base_model = GroundingDINO(ontology=ontology)
        
        print("Running cat detection...")
        results = base_model.predict(test_image)
        
        print(f"✅ Detection successful!")
        print(f"   Objects found: {len(results.xyxy)}")
        
        # Process results
        detected_cats = []
        for i in range(len(results.xyxy)):
            x1, y1, x2, y2 = results.xyxy[i]
            confidence = results.confidence[i]
            class_name = results.class_names[i]
            
            # Only keep high-confidence detections
            if confidence > 0.3:  # Confidence threshold
                cat_data = {
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
                detected_cats.append(cat_data)
                
                print(f"   Cat {i+1}: {class_name} (confidence: {confidence:.3f})")
                print(f"     Bounding box: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
        
        # Save results
        output_data = {
            "image_path": test_image,
            "detection_method": "autodistill_grounding_dino_fixed",
            "model_info": {
                "transformers_version": transformers.__version__,
                "torch_version": torch.__version__,
                "confidence_threshold": 0.3
            },
            "num_detections": len(detected_cats),
            "objects": detected_cats
        }
        
        with open('autodistill_fixed_cats.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Results saved to: autodistill_fixed_cats.json")
        
        # Create annotated image
        image = cv2.imread(test_image)
        for cat in detected_cats:
            x1, y1, x2, y2 = int(cat['bbox']['x1']), int(cat['bbox']['y1']), int(cat['bbox']['x2']), int(cat['bbox']['y2'])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Thicker green box
            label_text = f"Cat {cat['id']}: {cat['confidence']:.2f}"
            cv2.putText(image, label_text, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite('autodistill_fixed_cats.jpg', image)
        print(f"✅ Annotated image saved to: autodistill_fixed_cats.jpg")
        
        # Summary
        print(f"\n🎯 AUTODISTILL RESULTS:")
        print(f"   - Cats detected: {len(detected_cats)}")
        print(f"   - Method: AI-powered object detection")
        print(f"   - Model: Grounding DINO")
        print(f"   - Confidence threshold: 0.3")
        
        if len(detected_cats) >= 4:
            print(f"   ✅ Much better than manual detection!")
        elif len(detected_cats) >= 2:
            print(f"   ⚠️  Some improvement, but still missing cats")
        else:
            print(f"   ❌ Still not detecting well")
        
    else:
        print(f"❌ Test image not found: {test_image}")
        
except Exception as e:
    print(f"❌ Autodistill still not working: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n🔧 TROUBLESHOOTING:")
    print("If still failing, try:")
    print("1. Restart Python environment")
    print("2. Clear __pycache__ folders")
    print("3. Use different Transformers version (4.30.0)")
    print("4. Check CUDA/GPU compatibility")

print("\n" + "=" * 50)
print("AUTODISTILL DIAGNOSIS COMPLETE")
