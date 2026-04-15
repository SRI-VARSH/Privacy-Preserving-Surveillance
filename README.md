# 🎭 Face Anonymization Pipeline using Deep Learning

A privacy-preserving computer vision system that detects, tracks, and anonymizes faces in video footage using deep learning. The system identifies a **target person** (the one to protect/keep visible) and replaces all **non-target faces** in every frame with a donor face using AI-powered face swapping — ensuring bystanders and other individuals are anonymized automatically.

---

## 🎯 Project Overview

This project is designed to understand and implement a full face anonymization pipeline using:

- **YOLOv8-based face detection** for locating faces in each frame
- **ByteTrack multi-object tracking** for assigning consistent IDs across frames
- **FaceNet (InceptionResnetV1)** for generating face embeddings and identifying the target person
- **InsightFace + SimSwap (inswapper_128)** for realistic face-swap anonymization of non-target faces

The pipeline processes a sequence of frames extracted from a video, preserves the identity of a designated target person, and outputs anonymized frames where all other visible faces are replaced with a donor face.

---

## 🎯 Learning Objectives

This project helps in understanding:

- YOLOv8 object detection applied to face localization
- Multi-object tracking with ByteTrack (`bytetrack.yaml`)
- Face embedding generation and cosine similarity for identity matching
- Face swapping with InsightFace's `buffalo_l` model and `inswapper_128.onnx`
- Modular pipeline design across multiple Jupyter notebooks
- YAML-based configuration management for experiment reproducibility

---

## 📂 Project Structure

```
├── config.yaml               # Central configuration file for all pipeline parameters
├── detections.ipynb          # Stage 1: YOLOv8 face detection on all frames
├── tracking.ipynb            # Stage 2: ByteTrack multi-object face tracking
├── face_embeddings.ipynb     # Stage 3: FaceNet embedding + target identity filtering
└── Anonymization.ipynb       # Stage 4: InsightFace face swap for anonymization
```

---

## 📋 Dataset / Input

| Component | Description |
|---|---|
| `base_raw_path` | Directory of sequential input frames (`.jpg` images, e.g. `00000000.jpg`, `00000001.jpg`, ...) |
| `donor` | Path to a single image of the **donor** whose face replaces non-target faces |
| `output_path` | Directory where anonymized output frames are saved |
| `target.image_path` | Path to a clear image of the **target person** to preserve (not anonymize) |

The pipeline was tested on a dataset of **2,292 video frames** extracted from surveillance/camera footage (stored in a `.zip` archive and extracted to `/content/cam/`).

---

## 🧩 Pipeline Stages

### Stage 1 — Face Detection (`detections.ipynb`)

Uses a **YOLOv8 model fine-tuned for face detection**, downloaded from Hugging Face (`arnabdhar/YOLOv8-Face-Detection`), to locate all faces in every frame.

**Key functions:**
- `_load_model()` — Downloads and loads the YOLOv8 face model from Hugging Face Hub
- `detect_faces(frame_id, image)` — Runs inference and returns bounding boxes with confidence scores per frame
- `prepare_detections_for_bytetracker(frame_data)` — Filters detections by confidence threshold and minimum bounding box area, converting them into the `[x1, y1, x2, y2, score]` format required by ByteTrack
- `run_detection_pipeline(dataset_path)` — Main pipeline runner; iterates over all frames and returns prepared detections

**Output format per frame:**
```python
{
  "frame_id": int,
  "detections": np.array([[x1, y1, x2, y2, score], ...])
}
```

---

### Stage 2 — Face Tracking (`tracking.ipynb`)

Uses **ByteTrack** (via Ultralytics' built-in tracker with `bytetrack.yaml`) to assign consistent track IDs to detected faces across frames, enabling temporal consistency in anonymization.

**Key functions:**
- `track_faces(frame_id, image)` — Runs ByteTrack on a single frame using the pre-loaded YOLO model from Stage 1 (shared via `detections.model`)
- `run_tracking_pipeline(dataset_path)` — Calls the detection pipeline first, then applies tracking on each frame

**Output format per frame:**
```python
{
  "frame_id": int,
  "tracks": [
    {"track_id": int, "bbox": [x1, y1, x2, y2], "confidence": float},
    ...
  ]
}
```

---

### Stage 3 — Face Embedding & Target Filtering (`face_embeddings.ipynb`)

Uses **FaceNet's InceptionResnetV1** model (pretrained on `vggface2`) to generate 512-dimensional L2-normalized face embeddings. Each tracked face is compared against the target person's embedding using **cosine similarity**. Faces that match the target above a similarity threshold are excluded from anonymization.

**Key functions:**
- `get_embedding(crop)` — Preprocesses a face crop (resize to 160×160, normalize to `[-1, 1]`) and returns its embedding vector
- `get_target_embedding()` — Loads the target image and computes its reference embedding
- `is_target(embedding, target_embedding)` — Computes dot product similarity and compares against `similarity_threshold`
- `get_face_crop(frame, bbox)` — Safely crops a face region from a frame with boundary clamping
- `run_embedding_pipeline(dataset_path)` — Full pipeline: runs tracking, then filters out the target person from each frame's tracks

**Output format per frame:**
```python
{
  "frame_id": int,
  "tracks": [
    {
      "track_id": int,
      "bbox": [x1, y1, x2, y2],
      "confidence": float,
      "embedding": list[float],  # 512-dim vector
      "crop": np.ndarray         # face crop image
    },
    ...
  ]
}
```

Only **non-target** faces are included in the output — the target person is silently excluded from anonymization.

---

### Stage 4 — Face Swap Anonymization (`Anonymization.ipynb`)

Uses **InsightFace** (`buffalo_l` model) for landmark detection and **inswapper_128.onnx** (SimSwap-style face swapper) to replace non-target faces with the donor face.

**Key steps:**
1. **Setup** — Mounts Google Drive, extracts dataset `.zip`, installs dependencies (`ultralytics`, `facenet-pytorch`, `insightface`, `ipynb`)
2. **InsightFace Initialization** — Loads `buffalo_l` (includes landmark detection, recognition, gender/age models) and `inswapper_128.onnx` for face swapping
3. **Donor Face Loading** — Reads and resizes the donor image to 512×512, detects its face using InsightFace, and stores the reference donor face object
4. **`is_same_face(b1, b2, threshold=60)`** — Matches a tracker bounding box to an InsightFace-detected face by comparing bounding box center coordinates (within 60-pixel threshold)
5. **`swap_face(frame, bbox, donor_face)`** — Detects all faces in the full frame with InsightFace, finds the face matching the tracker's bounding box, and applies the face swap using `swapper.get(frame, face, donor_face, paste_back=True)`
6. **`run_anonymization_pipeline()`** — Calls the embedding pipeline, then for each frame and each non-target track, applies `swap_face()`. Saves the output frame as `.jpg` to the output directory. Processes up to `MAX_FRAMES = 300` frames.

**Bounding box format handling:** The pipeline includes a safety check to detect and convert `(x, y, w, h)` format bounding boxes to `(x1, y1, x2, y2)` format if needed.

---

## ⚙️ Configuration (`config.yaml`)

All parameters are centralized in `config.yaml`:

```yaml
project:
  base_raw_path: "Add your input_path"      # Path to input frames directory
  donor: "Add your donor_path"              # Path to donor face image
  output_path: "Add your output_path"       # Path to save output frames

face_detection:
  source: "huggingface"
  repo_id: "arnabdhar/YOLOv8-Face-Detection"
  filename: "model.pt"
  conf_threshold: 0.25                      # Minimum detection confidence
  iou_threshold: 0.5                        # NMS IoU threshold
  max_det: 10                               # Max detections per frame
  device: "cpu"                             # "cpu" or "cuda"
  min_box_area: 150                         # Minimum bounding box area (px²)

tracking:
  tracker_type: "bytetrack"
  track_thresh: 0.5                         # Track confidence threshold
  match_thresh: 0.8                         # IoU threshold for track matching
  track_buffer: 30                          # Frames to keep lost tracks alive
  frame_rate: 30                            # Input video frame rate

face_embedding:
  model_name: "vggface2"                    # FaceNet pretrained weights
  image_size: 160                           # Input size for FaceNet
  similarity_threshold: 0.6                # Cosine similarity to identify target
  min_face_size: 40                         # Minimum face crop size (px)
  device: "cpu"

target:
  image_path: "Add your target_path"       # Path to target person's reference image
```

---

## 🔢 Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `conf_threshold` | 0.25 | Minimum YOLO confidence to keep a detection |
| `iou_threshold` | 0.5 | NMS overlap threshold for face detection |
| `min_box_area` | 150 | Discard tiny face detections below this area |
| `similarity_threshold` | 0.6 | Cosine similarity above which a face is the target |
| `min_face_size` | 40 | Skip faces smaller than 40×40 pixels for embedding |
| `track_buffer` | 30 | Frames a track is kept alive when the face disappears |
| `MAX_FRAMES` | 300 | Maximum frames processed in the anonymization stage |

---

## 🤖 Models Used

| Model | Purpose | Source |
|---|---|---|
| YOLOv8 Face (`model.pt`) | Face detection | Hugging Face: `arnabdhar/YOLOv8-Face-Detection` |
| ByteTrack | Multi-object tracking | Ultralytics built-in (`bytetrack.yaml`) |
| InceptionResnetV1 (`vggface2`) | Face embedding generation | `facenet-pytorch` |
| InsightFace `buffalo_l` | Facial landmark + recognition | `insightface` pip package |
| `inswapper_128.onnx` | Face swap / anonymization | Auto-downloaded via InsightFace model zoo |

---

## ⚙️ Full Pipeline Workflow

```
Input Frames (sequential .jpg files)
         │
         ▼
┌─────────────────────────┐
│  Stage 1: Detection      │  YOLOv8 → bounding boxes + confidence scores
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Stage 2: Tracking       │  ByteTrack → track_id per face across frames
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Stage 3: Embedding      │  FaceNet → embeddings → filter out target person
└──────────┬──────────────┘
           │  (only non-target faces remain)
           ▼
┌─────────────────────────┐
│  Stage 4: Anonymization  │  InsightFace + inswapper_128 → swap non-target faces
└──────────┬──────────────┘
           │
           ▼
Output Frames (anonymized .jpg files saved to output_path)
```

---

## 🛠️ Technologies & Dependencies

| Library | Purpose |
|---|---|
| `ultralytics` | YOLOv8 face detection + ByteTrack tracking |
| `facenet-pytorch` | InceptionResnetV1 face embeddings |
| `insightface` | Face analysis and identity-aware face swapping |
| `onnxruntime` | ONNX model inference for InsightFace and inswapper |
| `opencv-python` | Image I/O, frame processing, crop extraction |
| `torch` + `torchvision` | Deep learning inference backend |
| `numpy` | Numerical operations and array handling |
| `Pillow` | Image preprocessing for FaceNet |
| `huggingface_hub` | Model download for YOLOv8 face detector |
| `pyyaml` | YAML config loading |
| `ipynb` | Cross-notebook imports (Stage 1 → 2 → 3 → 4) |
| `scipy` / `scikit-learn` | Dependencies for InsightFace |

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install ultralytics
pip install ipynb
pip install facenet-pytorch
pip install "numpy<2"
pip install insightface onnxruntime pyyaml
```

> **Note:** `numpy<2` is required due to compatibility constraints with `facenet-pytorch` and InsightFace.

### 2. Prepare Your Data

Organize your files as follows:
- Extract video frames as sequential `.jpg` images into an input folder
- Prepare a clear, frontal photo of the **target person** (the one to *not* anonymize)
- Prepare a clear, frontal photo of the **donor person** (the face to use for anonymization)

### 3. Update `config.yaml`

```yaml
project:
  base_raw_path: "/path/to/your/input/frames/"
  donor: "/path/to/donor_face.jpg"
  output_path: "/path/to/output/frames/"

target:
  image_path: "/path/to/target_person.jpg"
```

### 4. Run on Google Colab

The project is designed to run on **Google Colab** with Google Drive integration:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Upload `config.yaml` and your image ZIP archive to Google Drive, then run the notebooks in order:

1. `detections.ipynb`
2. `tracking.ipynb`
3. `face_embeddings.ipynb`
4. `Anonymization.ipynb`

> Each notebook imports the pipeline from the previous stage using `ipynb.fs.full.*` cross-notebook imports.

---

## 📁 Output

Anonymized frames are saved as zero-padded `.jpg` files (e.g., `00000.jpg`, `00001.jpg`, ...) to the directory specified in `config.yaml → project.output_path`. These frames can then be re-assembled into a video using any video encoding tool (e.g., `ffmpeg`).

```bash
# Reassemble frames into video (optional)
ffmpeg -framerate 30 -i output/%05d.jpg -c:v libx264 anonymized_output.mp4
```
