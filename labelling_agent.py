"""
Data Labelling Agent using LangChain + ZyndAI (Multi-Input + Vision Models)

A comprehensive data labelling agent that handles BOTH text and vision data:

TEXT PIPELINE (LangChain + Gemini):
  - Plain text, CSV, JSON, TXT files
  - PDF text extraction → labelling
  - Audio transcription (Whisper) → labelling
  - Excel spreadsheet rows → labelling
  - URL web scraping → labelling
  - Structured tabular data → row-level labelling

VISION PIPELINE (Autodistill + Grounding DINO + SAM):
  - Object detection via Grounding DINO (zero-shot, text-prompted)
  - Instance segmentation via Segment Anything Model (SAM)
  - GroundedSAM = Grounding DINO + SAM combined pipeline
  - Autodistill for auto-labelling image folders → train target models (YOLOv8)
  - Gemini Vision for general image classification

Install dependencies:
    # Core
    pip install langchain langchain-google-genai langchain-core langchain-classic
    pip install zyndai-agent python-dotenv

    # Text pipeline
    pip install Pillow pytesseract PyPDF2 openpyxl openai-whisper
    pip install requests beautifulsoup4 pandas

    # Vision pipeline (Autodistill + Grounding DINO + SAM)
    pip install autodistill autodistill-grounded-sam autodistill-grounding-dino
    pip install autodistill-yolov8 supervision
    pip install torch torchvision  # required for vision models
"""

from zyndai_agent.agent import AgentConfig, ZyndAIAgent, AgentFramework
from zyndai_agent.message import AgentMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv
import os
import json
import csv
import tempfile

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════
# TOOLS — TEXT INPUT
# ══════════════════════════════════════════════════════════════════════════

@tool
def read_csv_data(file_path: str) -> str:
    """Read unlabelled data from a CSV file. Returns JSON string of all rows."""
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return json.dumps(rows, indent=2)


@tool
def read_json_data(file_path: str) -> str:
    """Read unlabelled data from a JSON file. Returns JSON string."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, indent=2)


@tool
def read_text_data(file_path: str) -> str:
    """Read unlabelled data from a text file (one item per line). Returns JSON array."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return json.dumps(lines, indent=2)


# ══════════════════════════════════════════════════════════════════════════
# TOOLS — IMAGE: OCR + GEMINI VISION
# ══════════════════════════════════════════════════════════════════════════

@tool
def extract_text_from_image(file_path: str) -> str:
    """
    Extract text from an image using Tesseract OCR.
    Use for: receipts, scanned documents, screenshots, handwritten text.
    Supports JPG, PNG, WEBP, BMP, TIFF.
    """
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return json.dumps({"source": file_path, "type": "image_ocr",
                           "extracted_lines": lines, "full_text": text.strip()}, indent=2)
    except ImportError:
        return json.dumps({"error": "Install: pip install Pillow pytesseract"})
    except Exception as e:
        return json.dumps({"error": f"OCR failed: {str(e)}"})


@tool
def classify_image_with_gemini(file_path: str, task_description: str) -> str:
    """
    Use Gemini Vision to classify an image based on visual content.
    Use for: general photo classification, scene recognition, quality inspection.
    Does NOT detect individual objects — use grounding_dino_detect for that.
    """
    try:
        import google.generativeai as genai
        from PIL import Image
        import cv2
        img = Image.open(file_path)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""Task: {task_description}
Analyze this image and respond with ONLY valid JSON:
{{"source": "{file_path}", "type": "image_vision", "label": "assigned label",
  "confidence": 0.0, "reasoning": "explanation", "description": "image contents"}}"""
        response = model.generate_content([prompt, img])
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        
        # Create annotated image
        annotated_path = file_path.rsplit('.', 1)[0] + '_classified.' + file_path.rsplit('.', 1)[1]
        cv2.imwrite(annotated_path, cv2.imread(file_path))
        
        return text
    except ImportError:
        return json.dumps({"error": "Install: pip install google-generativeai Pillow"})
    except Exception as e:
        return json.dumps({"error": f"Vision classification failed: {str(e)}"})


# ══════════════════════════════════════════════════════════════════════════
# TOOLS — VISION: GROUNDING DINO (Zero-Shot Object Detection)
# ══════════════════════════════════════════════════════════════════════════

@tool
def grounding_dino_detect(file_path: str, text_prompt: str,
                          box_threshold: float = 0.25,
                          text_threshold: float = 0.25) -> str:
    """
    Detect objects in an image using Autodistill Grounding DINO (zero-shot, text-prompted).
    Autodistill Grounding DINO finds objects matching natural language descriptions.

    Args:
        file_path: Path to image file (JPG, PNG, etc.)
        text_prompt: Text description of objects to find. Use periods to separate
                     multiple objects, e.g. "cat . dog . person ."
        box_threshold: Confidence threshold for box predictions (default 0.25)
        text_threshold: Confidence threshold for text matching (default 0.25)

    Returns JSON with detected objects: bounding boxes, labels, confidence scores.
    Use this for: finding specific objects, counting items, locating components.
    """
    try:
        # Primary method: Use Autodistill Grounding DINO (working approach)
        from autodistill_grounding_dino import GroundingDINO
        from autodistill.detection import CaptionOntology
        import supervision as sv
        import cv2

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

            # Apply confidence threshold
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

        # Create annotated image
        import cv2
        image = cv2.imread(file_path)
        if image is not None:
            for i, det in enumerate(detections):
                x, y, w, h = int(det["bbox"]["x"]), int(det["bbox"]["y"]), int(det["bbox"]["width"]), int(det["bbox"]["height"])
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label_text = f"{det['label']} {det['confidence']:.2f}"
                cv2.putText(image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save annotated image
            annotated_path = file_path.rsplit('.', 1)[0] + '_annotated.' + file_path.rsplit('.', 1)[1]
            cv2.imwrite(annotated_path, image)
        else:
            annotated_path = None

        return {
            "source": file_path,
            "type": "autodistill_grounding_dino_detection",
            "prompt": text_prompt,
            "num_detections": len(detections),
            "detections": detections,
            "annotated_image": annotated_path,
            "model_info": {
                "method": "autodistill_wrapper",
                "confidence_threshold": box_threshold
            }
        }

    except ImportError:
        return {"error": "Install: pip install autodistill-grounding-dino supervision"}
    except Exception as e:
        return {"error": f"Autodistill Grounding DINO detection failed: {str(e)}"}


# ══════════════════════════════════════════════════════════════════════════
# TOOLS — VISION: GROUNDED SAM (Detection + Segmentation)
# ══════════════════════════════════════════════════════════════════════════

@tool
def grounded_sam_detect_and_segment(file_path: str, text_prompt: str,
                                     output_dir: str = "./grounded_sam_output") -> str:
    """
    Detect AND segment objects using GroundedSAM (Grounding DINO + Segment Anything).
    Combines zero-shot detection with pixel-level segmentation masks.

    Args:
        file_path: Path to image file
        text_prompt: Objects to detect, separated by periods. e.g. "cat . dog ."
        output_dir: Directory to save annotated images and masks

    Returns JSON with detections including bounding boxes, masks, and labels.
    Use this for: auto-labelling images for training, instance segmentation,
    generating YOLO/COCO annotations, pixel-level object isolation.
    """
    try:
        from autodistill_grounded_sam import GroundedSAM
        from autodistill.detection import CaptionOntology
        import supervision as sv
        import cv2
        import numpy as np

        labels = [l.strip().rstrip(".").strip() for l in text_prompt.split(".") if l.strip()]
        ontology = CaptionOntology({label: label for label in labels})

        base_model = GroundedSAM(ontology=ontology)
        results = base_model.predict(file_path)

        os.makedirs(output_dir, exist_ok=True)

        detections = []
        for i in range(len(results.xyxy)):
            box = results.xyxy[i].tolist()
            conf = results.confidence[i] if results.confidence is not None else 0.0
            cls_id = results.class_id[i] if results.class_id is not None else 0
            label = labels[cls_id] if cls_id < len(labels) else "unknown"

            has_mask = results.mask is not None and i < len(results.mask)
            mask_area = 0
            if has_mask:
                mask = results.mask[i]
                mask_area = int(np.sum(mask))
                mask_path = os.path.join(output_dir, f"mask_{i}_{label}.png")
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

            detections.append({
                "label": label,
                "confidence": round(float(conf), 4),
                "bbox": {"x1": round(box[0], 2), "y1": round(box[1], 2),
                         "x2": round(box[2], 2), "y2": round(box[3], 2)},
                "has_mask": has_mask,
                "mask_area_pixels": mask_area,
            })

        # Save annotated image
        image = cv2.imread(file_path)
        annotated = sv.BoundingBoxAnnotator().annotate(scene=image.copy(), detections=results)
        annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=results,
                                                  labels=[d["label"] for d in detections])
        annotated_path = os.path.join(output_dir, "annotated_output.jpg")
        cv2.imwrite(annotated_path, annotated)

        return json.dumps({
            "source": file_path,
            "type": "grounded_sam_segmentation",
            "prompt": text_prompt,
            "num_detections": len(detections),
            "detections": detections,
            "annotated_image": annotated_path,
            "output_dir": output_dir,
        }, indent=2)

    except ImportError:
        return json.dumps({"error": "Install: pip install autodistill-grounded-sam supervision opencv-python"})
    except Exception as e:
        return json.dumps({"error": f"GroundedSAM failed: {str(e)}"})


# ══════════════════════════════════════════════════════════════════════════
# TOOLS — VISION: AUTODISTILL (Auto-Label Folder + Train Target Model)
# ══════════════════════════════════════════════════════════════════════════

@tool
def autodistill_label_folder(input_folder: str, output_folder: str,
                              ontology_json: str) -> str:
    """
    Auto-label an entire folder of images using Autodistill + GroundedSAM.
    Generates YOLO-format annotations for every image in the folder.

    Args:
        input_folder: Path to folder containing unlabelled images
        output_folder: Path to save labelled dataset (YOLO format)
        ontology_json: JSON string mapping prompts to class names.
            Example: '{"milk bottle": "bottle", "bottle cap": "cap"}'
            Keys are text prompts for Grounding DINO, values are class names.

    Returns JSON with labelling statistics and output paths.
    Use this for: bulk auto-labelling image datasets, preparing training data.
    """
    try:
        from autodistill_grounded_sam import GroundedSAM
        from autodistill.detection import CaptionOntology

        ontology_dict = json.loads(ontology_json)
        ontology = CaptionOntology(ontology_dict)
        base_model = GroundedSAM(ontology=ontology)

        # Count input images
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        input_images = [f for f in os.listdir(input_folder)
                        if os.path.splitext(f)[1].lower() in image_extensions]

        # Label the folder (generates YOLO annotations)
        base_model.label(
            input_folder=input_folder,
            output_folder=output_folder,
        )

        # Count generated labels
        labels_dir = os.path.join(output_folder, "train", "labels")
        label_files = []
        if os.path.isdir(labels_dir):
            label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

        return json.dumps({
            "type": "autodistill_folder_labelling",
            "input_folder": input_folder,
            "output_folder": output_folder,
            "ontology": ontology_dict,
            "input_images": len(input_images),
            "labels_generated": len(label_files),
            "format": "YOLO",
            "class_names": list(ontology_dict.values()),
            "data_yaml": os.path.join(output_folder, "data.yaml"),
        }, indent=2)

    except ImportError:
        return json.dumps({"error": "Install: pip install autodistill autodistill-grounded-sam"})
    except Exception as e:
        return json.dumps({"error": f"Autodistill labelling failed: {str(e)}"})


@tool
def autodistill_train_target_model(data_yaml_path: str, model_type: str = "yolov8n.pt",
                                    epochs: int = 50) -> str:
    """
    Train a target model (YOLOv8) on an Autodistill-labelled dataset.
    This distils the knowledge from GroundedSAM into a fast, deployable model.

    Args:
        data_yaml_path: Path to data.yaml from autodistill_label_folder output
        model_type: YOLOv8 model variant (yolov8n.pt, yolov8s.pt, yolov8m.pt)
        epochs: Number of training epochs (default: 50)

    Returns JSON with training results and model weights path.
    Use this for: training fast edge-deployable models from auto-labelled data.
    """
    try:
        from autodistill_yolov8 import YOLOv8

        target_model = YOLOv8(model_type)
        target_model.train(data_yaml_path, epochs=epochs)

        weights_path = os.path.join("runs", "detect", "train", "weights", "best.pt")

        return json.dumps({
            "type": "autodistill_training",
            "data_yaml": data_yaml_path,
            "model_type": model_type,
            "epochs": epochs,
            "weights_path": weights_path,
            "status": "training_complete",
        }, indent=2)

    except ImportError:
        return json.dumps({"error": "Install: pip install autodistill-yolov8 ultralytics"})
    except Exception as e:
        return json.dumps({"error": f"Training failed: {str(e)}"})


# ══════════════════════════════════════════════════════════════════════════
# TOOLS — PDF, AUDIO, EXCEL, URL
# ══════════════════════════════════════════════════════════════════════════

@tool
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text page-by-page from a PDF document."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        pages = [{"page_number": i + 1, "text": (p.extract_text() or "").strip()}
                 for i, p in enumerate(reader.pages) if (p.extract_text() or "").strip()]
        return json.dumps({"source": file_path, "type": "pdf",
                           "total_pages": len(reader.pages), "pages": pages}, indent=2)
    except ImportError:
        return json.dumps({"error": "Install: pip install PyPDF2"})
    except Exception as e:
        return json.dumps({"error": f"PDF extraction failed: {str(e)}"})


@tool
def transcribe_audio(file_path: str) -> str:
    """Transcribe audio to text with timestamps using Whisper. Supports MP3, WAV, OGG, FLAC."""
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        segments = [{"start": round(s["start"], 2), "end": round(s["end"], 2),
                     "text": s["text"].strip()} for s in result.get("segments", [])]
        return json.dumps({"source": file_path, "type": "audio",
                           "language": result.get("language", "unknown"),
                           "full_text": result["text"].strip(), "segments": segments}, indent=2)
    except ImportError:
        return json.dumps({"error": "Install: pip install openai-whisper"})
    except Exception as e:
        return json.dumps({"error": f"Transcription failed: {str(e)}"})


@tool
def read_excel_data(file_path: str, sheet_name: str = "") -> str:
    """Read data from an Excel file (.xlsx, .xls). Returns JSON with columns and rows."""
    try:
        import pandas as pd
        kwargs = {"engine": "openpyxl"}
        if sheet_name:
            kwargs["sheet_name"] = sheet_name
        df = pd.read_excel(file_path, **kwargs).fillna("")
        rows = df.to_dict(orient="records")
        return json.dumps({"source": file_path, "type": "excel",
                           "columns": list(df.columns), "row_count": len(rows), "rows": rows}, indent=2)
    except ImportError:
        return json.dumps({"error": "Install: pip install pandas openpyxl"})
    except Exception as e:
        return json.dumps({"error": f"Excel read failed: {str(e)}"})


@tool
def extract_text_from_url(url: str) -> str:
    """Scrape text from a web page. Extracts title, headings, paragraphs, list items."""
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, headers={"User-Agent": "DataLabellingAgent/1.0"}, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 20]
        return json.dumps({"source": url, "type": "web_page", "title": title,
                           "paragraphs": paragraphs[:50], "total_paragraphs": len(paragraphs)}, indent=2)
    except ImportError:
        return json.dumps({"error": "Install: pip install requests beautifulsoup4"})
    except Exception as e:
        return json.dumps({"error": f"URL extraction failed: {str(e)}"})


# ══════════════════════════════════════════════════════════════════════════
# TOOLS — LABELLING ENGINES
# ══════════════════════════════════════════════════════════════════════════

@tool
def label_data_batch(data_and_config: str) -> str:
    """
    Label a batch of text items using Gemini. Input: JSON with keys:
    - "items": list of strings  - "labels": optional allowed labels
    - "task": task description  - "few_shot": optional examples
    Returns JSON array with original_text, label, confidence, reasoning.
    """
    config = json.loads(data_and_config)
    items, labels = config.get("items", []), config.get("labels")
    task, few_shot = config.get("task", "Classify the text."), config.get("few_shot")

    label_inst = f"\nAssign one of: {json.dumps(labels)}\n" if labels else ""
    fs = ""
    if few_shot:
        fs = "\n## Examples:\n" + "".join(f'- "{e["text"]}" -> "{e["label"]}"\n' for e in few_shot)

    system = f"""You are a data labelling assistant.
## Task
{task}{label_inst}{fs}
## Output
JSON array. Each: {{"original_text": "...", "label": "...", "confidence": 0.0, "reasoning": "..."}}
ONLY valid JSON. No markdown."""

    numbered = "\n".join(f"{j+1}. {item}" for j, item in enumerate(items))
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "Label:\n\n{data_items}")])
    result = (prompt | llm | JsonOutputParser()).invoke({"data_items": numbered})
    return json.dumps(result, indent=2)


@tool
def label_tabular_rows(data_and_config: str) -> str:
    """
    Label structured tabular rows considering all columns. Input: JSON with keys:
    - "rows": list of row dicts  - "labels": optional allowed labels
    - "task": task description   - "label_column": column name for label (default: "label")
    Returns JSON array of enriched rows with label, confidence, reasoning columns.
    """
    config = json.loads(data_and_config)
    rows, labels = config.get("rows", []), config.get("labels")
    task = config.get("task", "Classify each row.")
    label_col = config.get("label_column", "label")

    label_inst = f"\nAssign one of: {json.dumps(labels)}\n" if labels else ""
    row_texts = [f"Row {i+1}: {json.dumps(r)}" for i, r in enumerate(rows)]

    system = f"""You are a tabular data labelling assistant.
## Task
{task}{label_inst}
## Output
JSON array. Each: {{"row_index": N, "label": "...", "confidence": 0.0, "reasoning": "..."}}
ONLY valid JSON."""

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "Label:\n\n{data_items}")])
    result = (prompt | llm | JsonOutputParser()).invoke({"data_items": "\n".join(row_texts), "row_index": list(range(len(rows)))})

    label_map = {item.get("row_index", 0) - 1: item for item in (result if isinstance(result, list) else [])}
    enriched = []
    for i, row in enumerate(rows):
        r = dict(row)
        m = label_map.get(i, {})
        r[label_col] = m.get("label", "unknown")
        r[f"{label_col}_confidence"] = m.get("confidence", 0.0)
        r[f"{label_col}_reasoning"] = m.get("reasoning", "")
        enriched.append(r)
    return json.dumps(enriched, indent=2)


# ══════════════════════════════════════════════════════════════════════════
# TOOLS — SAVE OUTPUT
# ══════════════════════════════════════════════════════════════════════════

@tool
def save_labelled_csv(file_path: str, labelled_json: str) -> str:
    """Save labelled data to CSV."""
    data = json.loads(labelled_json)
    if not data: return "Error: No data."
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=data[0].keys())
        w.writeheader()
        w.writerows(data)
    return f"Saved {len(data)} items to {file_path}"


@tool
def save_labelled_json(file_path: str, labelled_json: str) -> str:
    """Save labelled data to JSON."""
    data = json.loads(labelled_json)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return f"Saved to {file_path}"


@tool
def save_labelled_excel(file_path: str, labelled_json: str) -> str:
    """Save labelled data to Excel (.xlsx)."""
    try:
        import pandas as pd
        data = json.loads(labelled_json)
        if not data: return "Error: No data."
        pd.DataFrame(data).to_excel(file_path, index=False, engine="openpyxl")
        return f"Saved {len(data)} items to {file_path}"
    except ImportError:
        return "Error: pip install pandas openpyxl"


# ══════════════════════════════════════════════════════════════════════════
# AGENT CREATION
# ══════════════════════════════════════════════════════════════════════════

def create_labelling_agent():
    """Create the multi-input labelling agent with text + vision pipelines."""

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    tools = [
        # Text input tools
        read_csv_data, read_json_data, read_text_data,
        # Image tools
        extract_text_from_image, classify_image_with_gemini,
        # Vision pipeline tools
        grounding_dino_detect, grounded_sam_detect_and_segment,
        autodistill_label_folder, autodistill_train_target_model,
        # PDF, Audio, Excel, URL
        extract_text_from_pdf, transcribe_audio, read_excel_data, extract_text_from_url,
        # Labelling engines
        label_data_batch, label_tabular_rows,
        # Save output
        save_labelled_csv, save_labelled_json, save_labelled_excel,
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional multi-input data labelling agent with TWO pipelines:

=== TEXT PIPELINE (Gemini LLM) ===
For text classification, sentiment analysis, topic labelling, intent detection:
- TEXT files: read_csv_data / read_json_data / read_text_data -> label_data_batch -> save
- PDF: extract_text_from_pdf -> label_data_batch -> save
- AUDIO: transcribe_audio -> label_data_batch -> save
- EXCEL: read_excel_data -> label_tabular_rows or label_data_batch -> save
- URL: extract_text_from_url -> label_data_batch -> save
- INLINE TEXT: label_data_batch directly

=== VISION PIPELINE (Grounding DINO + SAM + Autodistill) ===
For object detection, instance segmentation, image auto-labelling:

1. grounding_dino_detect: Zero-shot object detection with text prompts.
   Use for: "Find all X in this image", counting objects, locating items.
   Input: image path + text prompt (e.g. "cat . dog . person .")

2. grounded_sam_detect_and_segment: Detection + pixel-level segmentation masks.
   Combines Grounding DINO + Segment Anything Model.
   Use for: generating training annotations, instance segmentation, object isolation.
   Output includes bounding boxes, segmentation masks, and annotated image.

3. autodistill_label_folder: Auto-label an ENTIRE FOLDER of images.
   Uses GroundedSAM as base model. Generates YOLO-format annotations.
   Use for: bulk dataset labelling, preparing training data at scale.
   Input: folder path + ontology mapping prompts to class names.

4. autodistill_train_target_model: Train YOLOv8 on auto-labelled data.
   Distils GroundedSAM knowledge into a fast, deployable model.
   Use after autodistill_label_folder to get a production-ready model.

5. classify_image_with_gemini: General image classification via Gemini Vision.
   Use for: scene classification, quality inspection, general categorisation.

6. extract_text_from_image: OCR text extraction from images.
   Use for: reading text from receipts, documents, screenshots.

=== DECISION RULES ===
- "detect objects" / "find X in image" / "count items" -> grounding_dino_detect
- "segment objects" / "generate masks" / "annotate image" -> grounded_sam_detect_and_segment
- "label all images in folder" / "auto-label dataset" -> autodistill_label_folder
- "train a model" / "distil" / "YOLOv8" -> autodistill_train_target_model
- "classify this image" / "what is this" -> classify_image_with_gemini
- "read text from image" / "OCR" -> extract_text_from_image
- "label [image]" -> classify_image_with_gemini
- Text/CSV/JSON/PDF/Audio/Excel/URL -> Text pipeline tools

=== SMART DEFAULTS ===
- If user says "detect objects in [image]" without specifying objects, use "object" as default
- If user says "detect [specific objects]" use those objects directly
- If user says "label [image]" without specifying labels, use "general classification" as default
- Always try to infer intent and provide reasonable defaults
- Don't ask for clarification unless absolutely necessary

Always provide a summary of results after labelling."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=20)


# ══════════════════════════════════════════════════════════════════════════
# STANDALONE CONVENIENCE FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def label_data_inline(data, labels=None, few_shot_examples=None,
                      task_description="Classify the text.", model_name="gemini-2.5-flash",
                      api_key=None):
    """Label a list of strings directly (no ZyndAI). Returns list of dicts."""
    llm_kwargs = {"model": model_name, "temperature": 0}
    if api_key: llm_kwargs["google_api_key"] = api_key
    llm = ChatGoogleGenerativeAI(**llm_kwargs)

    li = f"\nAssign one of: {json.dumps(labels)}\n" if labels else ""
    fs = ""
    if few_shot_examples:
        fs = "\n## Examples:\n" + "".join(f'- "{e["text"]}" -> "{e["label"]}"\n' for e in few_shot_examples)

    sys = f"""Data labelling assistant.\n## Task\n{task_description}{li}{fs}
## Output\nJSON array. Each: {{"original_text":"..","label":"..","confidence":0.0,"reasoning":".."}}
ONLY JSON."""

    prompt = ChatPromptTemplate.from_messages([("system", sys), ("human", "Label:\n\n{data_items}")])
    chain = prompt | llm | JsonOutputParser()

    results = []
    for i in range(0, len(data), 20):
        batch = data[i:i+20]
        numbered = "\n".join(f"{j+1}. {item}" for j, item in enumerate(batch))
        r = chain.invoke({"data_items": numbered})
        results.extend(r if isinstance(r, list) else [r])
    return results


# ══════════════════════════════════════════════════════════════════════════
# MAIN: ZyndAI Integration
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    agent_config = AgentConfig(
        name="Data Labelling Agent (Multi-Input + Vision Models)",
        description=(
            "Multi-modal data labelling agent with TWO pipelines: "
            "(1) Text pipeline using Gemini for CSV/JSON/PDF/Excel/URL classification. "
            "(2) Vision pipeline using Grounding DINO + Segment Anything + Autodistill "
            "for zero-shot object detection, instance segmentation, folder auto-labelling, "
            "and YOLOv8 model distillation."
        ),
        capabilities={
            "ai": ["nlp", "classification", "langchain", "gemini", "grounding_dino",
                    "segment_anything", "autodistill", "vision", "ocr", "transcription"],
            "protocols": ["http"],
            "services": [
                "data_labelling", "sentiment_analysis", "topic_classification",
                "intent_detection", "content_moderation", "object_detection",
                "instance_segmentation", "image_auto_labelling", "model_distillation",
                "document_classification", "audio_classification", "web_content_classification",
            ],
            "domains": ["machine_learning", "data_science", "nlp", "computer_vision"],
        },
        webhook_host="0.0.0.0",
        webhook_port=5003,
        registry_url="https://registry.zynd.ai",
        price="$0.000",
        api_key=os.environ["ZYND_API_KEY"],
        config_dir=".agent-labelling",
        use_ngrok=True,
        ngrok_auth_token=os.environ.get(
            "2qnJ7BnoYYdxkmOTtKLNkCkIgIr_4icmFZy9xv7vNLPjDKG6Z"
        ),
    )

    zynd_agent = ZyndAIAgent(agent_config=agent_config)
    agent_executor = create_labelling_agent()
    zynd_agent.set_langchain_agent(agent_executor)

    def message_handler(message: AgentMessage, topic: str):
        import traceback
        print(f"\n{'='*60}\n[Agent] Received: {message.content}\n{'='*60}\n")
        try:
            response = zynd_agent.invoke(message.content, chat_history=[])
            print(f"\nResponse: {response}\n")
            zynd_agent.set_response(message.message_id, response)
        except Exception as e:
            print(f"ERROR: {e}\n{traceback.format_exc()}")
            zynd_agent.set_response(message.message_id, f"Error: {str(e)}")

    zynd_agent.add_message_handler(message_handler)

    print("\n" + "=" * 60)
    print("Systum is running")
    print(f"Price: 0.000 USDC per request")
    print(f"Webhook: {zynd_agent.webhook_url}")
    print("=" * 60)
    print("\n--- TEXT PIPELINE ---")
    print(" CSV / JSON / TXT / PDF ")
    print("\n--- VISION PIPELINE ---")
    print("  Grounding DINO    : Zero-shot object detection")
    print("  GroundedSAM       : Detection + segmentation masks")
    print("  Autodistill       : Folder auto-labelling (YOLO format)")
    print("  Gemini Vision     : General image classification")
    print("  OCR (Tesseract)   : Text extraction from images")
    print("\nExample messages:")
    print('  "Detect all bottles and caps in warehouse.jpg"')
    print('  "Auto-label all images in ./dataset/ with ontology: bottle, cap"')
    print('  "Classify reviews.csv by sentiment as positive/negative/neutral"')
    print('  "segment dogs and cats in [img]"')
    print("\nType 'exit' to quit\n")

    while True:
        cmd = input("Command: ")
        if cmd.lower() == "exit":
            break
        elif cmd.strip():
            try:
                response = zynd_agent.invoke(cmd, chat_history=[])
                print(f"\nResponse: {response}\n")
            except Exception as e:
                print(f"ERROR: {e}\n")
