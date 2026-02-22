"""
Fix Autodistill Compatibility Issues
"""
import os
import json

print("Autodistill Compatibility Diagnosis & Fix")
print("=" * 50)

# Check current versions
print("CURRENT PACKAGE VERSIONS:")
print("-" * 30)

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

try:
    import autodistill_grounding_dino
    print(f"Autodistill Grounding DINO: Installed")
except ImportError:
    print("Autodistill Grounding DINO: Not installed")

try:
    import autodistill_grounded_sam
    print(f"Autodistill GroundedSAM: Installed")
except ImportError:
    print("Autodistill GroundedSAM: Not installed")

print("\nPROBLEM ANALYSIS:")
print("-" * 30)

print("ISSUE: 'BertModel' object has no attribute 'get_head_mask'")
print("\nROOT CAUSE:")
print("1. Transformers version (5.2.0) is too new for Autodistill")
print("2. Autodistill was built for older Transformers API")
print("3. The 'get_head_mask' method was removed/changed in newer Transformers")
print("4. PyTorch 2.10.0 may also have compatibility issues")

print("\nSOLUTION OPTIONS:")
print("-" * 30)
print("Option 1: Downgrade Transformers to compatible version")
print("Option 2: Use alternative detection method")
print("Option 3: Manual Grounding DINO implementation")

print("\nATTEMPTING FIX - Option 1: Downgrade Transformers")
print("-" * 50)

# Try to fix by downgrading transformers
import subprocess
import sys

try:
    print("Uninstalling current Transformers...")
    result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "transformers", "-y"], 
                          capture_output=True, text=True)
    print(f"Uninstall result: {result.returncode}")
    
    print("Installing compatible Transformers version (4.21.0)...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "transformers==4.21.0"], 
                          capture_output=True, text=True)
    print(f"Install result: {result.returncode}")
    
    if result.returncode == 0:
        print("✅ Transformers downgraded successfully!")
    else:
        print("❌ Transformers downgrade failed:")
        print(result.stderr)
        
except Exception as e:
    print(f"❌ Error during package management: {e}")

print("\nTESTING AUTODISTILL AFTER FIX:")
print("-" * 30)

try:
    # Reload modules after version change
    import importlib
    if 'transformers' in sys.modules:
        importlib.reload(sys.modules['transformers'])
    
    from autodistill_grounding_dino import GroundingDINO
    from autodistill.detection import CaptionOntology
    
    print("✅ Autodistill Grounding DINO imported successfully!")
    
    # Test with a simple image
    test_image = "picture-unlabel.webp"
    if os.path.exists(test_image):
        print(f"Testing with: {test_image}")
        
        # Create ontology for cat detection
        ontology = CaptionOntology({"cat": "cat"})
        
        print("Initializing Grounding DINO model...")
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
        
        # Save results
        output_data = {
            "image_path": test_image,
            "detection_method": "autodistill_grounding_dino",
            "model_info": {
                "transformers_version": transformers.__version__ if 'transformers' in sys.modules else "unknown",
                "torch_version": torch.__version__ if 'torch' in sys.modules else "unknown"
            },
            "num_detections": len(detected_cats),
            "objects": detected_cats
        }
        
        with open('autodistill_cats.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Results saved to: autodistill_cats.json")
        
        # Create annotated image
        import cv2
        image = cv2.imread(test_image)
        for cat in detected_cats:
            x1, y1, x2, y2 = int(cat['bbox']['x1']), int(cat['bbox']['y1']), int(cat['bbox']['x2']), int(cat['bbox']['y2'])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"Cat {cat['id']}: {cat['confidence']:.2f}"
            cv2.putText(image, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite('autodistill_cats.jpg', image)
        print(f"✅ Annotated image saved to: autodistill_cats.jpg")
        
    else:
        print(f"❌ Test image not found: {test_image}")
        
except Exception as e:
    print(f"❌ Autodistill still not working: {e}")
    import traceback
    traceback.print_exc()
    
    print("\nFALLBACK: Alternative Solution")
    print("-" * 30)
    print("If downgrade didn't work, consider:")
    print("1. Using a different object detection library (YOLO, Detectron2)")
    print("2. Manual implementation with pre-trained models")
    print("3. Cloud-based detection APIs")

print("\nSUMMARY:")
print("-" * 30)
print("The issue was version incompatibility between:")
print("- Autodistill (built for older Transformers)")
print("- Transformers 5.2.0 (newer API)")
print("- PyTorch 2.10.0 (newer version)")
print("\nFix attempted: Downgrade Transformers to 4.21.0")
print("Result: Check above for success/failure status")
