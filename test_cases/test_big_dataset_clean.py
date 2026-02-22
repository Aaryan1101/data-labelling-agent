"""
Big Dataset Processing Test - Clean Output (No Emojis)
"""
import os
import json
import csv
import time
from datetime import datetime
import glob

print("Big Dataset Processing Test - Clean Output")
print("=" * 50)

# Test 1: Process all vision data images
print("\n=== TEST 1: BATCH VISION PROCESSING ===")

vision_folder = "vision data/auto_label_dataset"
image_files = glob.glob(os.path.join(vision_folder, "*.png"))

print(f"Found {len(image_files)} images in {vision_folder}")

vision_results = []
start_time = time.time()

for i, image_path in enumerate(image_files):
    try:
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Simple shape detection (using our corrected method)
        import cv2
        import numpy as np
        
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_not(thresh)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        for j, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 500:
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # Determine shape
            if circularity > 0.8:
                shape = "circle"
                confidence = min(0.95, circularity)
            else:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                vertices = len(approx)
                
                if vertices == 3:
                    shape = "triangle"
                    confidence = 0.85
                elif vertices == 4:
                    aspect_ratio = cv2.boundingRect(contour)[2] / cv2.boundingRect(contour)[3]
                    if 0.8 <= aspect_ratio <= 1.2:
                        shape = "square"
                        confidence = 0.90
                    else:
                        shape = "rectangle"
                        confidence = 0.85
                else:
                    shape = "polygon"
                    confidence = 0.70
            
            x, y, w, h = cv2.boundingRect(contour)
            detected_objects.append({
                "id": j + 1,
                "label": shape,
                "confidence": round(confidence, 3),
                "bbox": {"x": x, "y": y, "width": w, "height": h},
                "area": float(area)
            })
        
        # Create annotated image with bounding boxes
        annotated_image = image.copy()
        for obj in detected_objects:
            x, y, w, h = obj["bbox"]["x"], obj["bbox"]["y"], obj["bbox"]["width"], obj["bbox"]["height"]
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_image, f"{obj['label']} {obj['confidence']:.2f}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save annotated image
        annotated_filename = f"annotated_{os.path.basename(image_path)}"
        annotated_path = os.path.join("annotated_images", annotated_filename)
        os.makedirs("annotated_images", exist_ok=True)
        cv2.imwrite(annotated_path, annotated_image)
        
        result = {
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "annotated_image_path": annotated_path,
            "num_detections": len(detected_objects),
            "objects": detected_objects,
            "processing_time": time.time() - start_time
        }
        vision_results.append(result)
        
        print(f"   Found {len(detected_objects)} objects -> saved to {annotated_filename}")
        
    except Exception as e:
        print(f"   Error: {e}")

vision_time = time.time() - start_time
print(f"\nVision batch processing complete in {vision_time:.2f} seconds")

# Save vision results
vision_output = {
    "batch_info": {
        "total_images": len(image_files),
        "processing_time_seconds": round(vision_time, 2),
        "avg_time_per_image": round(vision_time / len(image_files), 3),
        "timestamp": datetime.now().isoformat()
    },
    "results": vision_results
}

with open('big_dataset_vision_clean.json', 'w') as f:
    json.dump(vision_output, f, indent=2)

print("Vision results saved to big_dataset_vision_clean.json")
print(f"Annotated images saved to annotated_images/ folder ({len(vision_results)} files)")

# Test 2: Process all text data
print("\n=== TEST 2: BATCH TEXT PROCESSING ===")

# Combine all text data sources
text_data = []

# Load reviews
try:
    with open('reviews_unlabelled.csv', 'r', encoding='utf-8') as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            text_data.append({
                "source": "reviews_unlabelled.csv",
                "id": row['review_id'],
                "text": row['review_text'],
                "type": "review"
            })
    print(f"Loaded {len(text_data)} reviews")
except:
    print("Could not load reviews.csv")

# Load headlines
try:
    with open('headlines.json', 'r', encoding='utf-8') as f:
        headlines = json.load(f)
        for item in headlines:
            text_data.append({
                "source": "headlines.json",
                "id": item['id'],
                "text": item['headline'],
                "type": "headline"
            })
    print(f"Loaded {len([d for d in text_data if d['type'] == 'headline'])} headlines")
except:
    print("Could not load headlines.json")

# Load simple reviews
try:
    with open('reviews.txt', 'r', encoding='utf-8') as f:
        simple_reviews = f.readlines()
        for i, line in enumerate(simple_reviews):
            if line.strip():
                text_data.append({
                    "source": "reviews.txt",
                    "id": i + 1,
                    "text": line.strip(),
                    "type": "simple_review"
                })
    print(f"Loaded {len([d for d in text_data if d['type'] == 'simple_review'])} simple reviews")
except:
    print("Could not load reviews.txt")

print(f"\nTotal text items to process: {len(text_data)}")

# Process text data with AI (sample for performance)
sample_size = min(20, len(text_data))  # Process sample to avoid API limits
sample_data = text_data[:sample_size]

print(f"Processing {sample_size} items with AI...")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatGoogleGenerativeAI(model="gemma-3-4b-it", temperature=0)

text_results = []
start_time = time.time()

for i, item in enumerate(sample_data):
    try:
        print(f"Processing {i+1}/{sample_size}: {item['type']} from {item['source']}")
        
        # Determine task based on text type
        if item['type'] in ['review', 'simple_review']:
            prompt = ChatPromptTemplate.from_template("""
Classify sentiment as positive, negative, or neutral.
Text: {text}
Output JSON: {{"sentiment": "positive/negative/neutral", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}""")
        else:  # headline
            prompt = ChatPromptTemplate.from_template("""
Classify topic: politics, sports, business, technology, entertainment, or other.
Text: {text}
Output JSON: {{"topic": "category", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}""")
        
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({"text": item['text']})
        
        text_results.append({
            "source": item['source'],
            "id": item['id'],
            "type": item['type'],
            "original_text": item['text'],
            "classification": result,
            "processing_time": time.time() - start_time
        })
        
        print(f"   Classified: {list(result.keys())[0]} = {list(result.values())[0]}")
        
    except Exception as e:
        print(f"   Error: {e}")

text_time = time.time() - start_time
print(f"\nText batch processing complete in {text_time:.2f} seconds")

# Save text results
text_output = {
    "batch_info": {
        "total_items": len(text_data),
        "processed_items": len(text_results),
        "processing_time_seconds": round(text_time, 2),
        "avg_time_per_item": round(text_time / len(text_results), 3),
        "timestamp": datetime.now().isoformat()
    },
    "results": text_results
}

with open('big_dataset_text_clean.json', 'w') as f:
    json.dump(text_output, f, indent=2)

print("Text results saved to big_dataset_text_clean.json")

# Summary
print("\n" + "=" * 50)
print("BIG DATASET PROCESSING COMPLETE - CLEAN OUTPUT")
print(f"SUMMARY:")
print(f"   Vision: {len(image_files)} images in {vision_time:.2f}s")
print(f"   Text: {len(text_results)} items in {text_time:.2f}s")
print(f"   Total processing time: {vision_time + text_time:.2f}s")
print(f"   Output files: big_dataset_vision_clean.json, big_dataset_text_clean.json")
print(f"   Annotated images: annotated_images/ folder ({len(vision_results)} files)")

# Performance metrics
print(f"\nPERFORMANCE:")
print(f"   Vision: {vision_time/len(image_files):.3f}s per image")
print(f"   Text: {text_time/len(text_results):.3f}s per text item")
print(f"   Overall: {(vision_time + text_time)/(len(image_files) + len(text_results)):.3f}s per item")

# Vision image generation explanation
print(f"\nVISION IMAGE GENERATION:")
print(f"   - Annotated images are now generated for each processed image")
print(f"   - Saved to 'annotated_images/' folder with bounding boxes")
print(f"   - Each image shows detected objects with labels and confidence scores")
print(f"   - Previous tests only saved JSON data, not visual results")
