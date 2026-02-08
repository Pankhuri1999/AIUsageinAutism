# AI-Supported Autism Therapy Monitoring  
## Methods for Non-Verbal and Speech Interpretation and Scoring

Repository:  
https://github.com/Pankhuri1999/AIUsageinAutism

This repository provides a research prototype for **AI-assisted autism therapy monitoring** using four modules:

- **Motion Therapy** (video-to-video motion similarity)  
- **Speech Therapy** (video-to-video mouth/speech movement similarity)  
- **Context Interpretation** (interactive UI + response scoring/logging)  
- **Emotion Recognition** (interactive UI + response scoring/logging)

Each module contains runnable code (Python scripts and/or Jupyter notebooks), experiment artifacts (CSV logs), and supporting analysis notebooks used to validate the experiments.

---

## 📁 Repository Structure

```text
AIUsageinAutism/
│
├── MotionTherapy/
├── SpeechTherapy/
├── ContextInterpretation/
├── EmotionRecognitionActivity/
└── README.md
```

---

## 📂 What’s Inside Each Folder (Detailed)

### 1) `MotionTherapy`

This folder contains **notebooks for running and validating motion-therapy video comparisons**.

**Includes:**
- **Python notebook(s) for direct comparison of two videos**  
  - `MotionTherapyNotebook.ipynb`  
  - Purpose: run **two videos (reference vs test)** and compute similarity metrics for *actual comparison*.
- **Additional notebooks for experimental validation and analysis** (proof notebooks)  
  - e.g., `motionTherapyNotebook1.ipynb`, `motionTherapyNotebook2.ipynb`  
  - Purpose: visualization + analysis used to **validate the experimental claims** (e.g., trends, stability checks).
-  **Python script** for running comparisons outside notebooks.  
- **Screenshots ** of outputs/plots can be found inside the notebook.

**Important (video path update):**  
If you are running the **Python file** (MotionTherapyPythonFile.py), update the paths here:

```python
if __name__ == "__main__":
    VIDEO1_PATH = "degree1.mp4"
    VIDEO2_PATH = "degree2.mp4"
```

---

### 2) `SpeechTherapy`

This folder contains **notebooks for running and validating speech-therapy comparisons** (typically mouth movement / articulation comparison using videos).

**Includes:**
- **Notebook for direct comparison of two videos**
  - `speechTherapyNotebook.ipynb`  
  - Purpose: run **two videos (reference vs test)** and compute similarity metrics for *actual comparison*.
- **Other notebooks for experimental validation and analysis**  
  - Purpose: analysis notebooks used as **proof/validation** of the experiments.
-  **Python script** for running comparisons outside notebooks.
- **Screenshots ** of outputs/plots can be found inside the notebook.

**Important (video path update):**  
If you are running the **Python file** (speechTherapyPythonFile.py), update the paths here:

```python
if __name__ == "__main__":
    video1 = "/content/vidS1.mp4"   # correct/reference
    video2 = "/content/vidS2.mp4"   # incorrect/test

    compare_two_videos(
        ...
    )
```

---

### 3) `ContextInterpretation`

This folder contains an **interactive UI** for context interpretation tasks and the **experiment logs** collected during sessions.

**Includes:**
- **Runnable Python file** to launch the UI  
  - Run it to start the Context Interpretation interface.
- **Jupyter notebook** version for development/analysis  
  - Useful for debugging, analysis, or running in notebook environments.
- **UI screenshot(s)**  
  - A screenshot image of the UI is stored in this folder.
- **CSV response logs**  
  - CSV files store values captured during experiments (responses + scores).

**How to run:**
```bash
python contextInterpretation.py
```

**CSV logs typically include (example fields):**
- timestamp / trial number  
- prompt/image id  
- child response  
- similarity score  
- response time / latency  
- running averages (if logged)

---

### 4) `EmotionRecognitionActivity`

This folder contains an **interactive UI** for emotion recognition therapy activity and the **experiment logs** collected during sessions.

**Includes:**
- **Runnable Python file** to launch the UI  
- **Jupyter notebook** version for development/analysis  
- **UI screenshot(s)** in the folder  
- **CSV response logs** (e.g., `responses_log_cartoon.csv`)  
  - Stores experiment responses and measured values.

**How to run:**
```bash
python emotionTherapy.py
```

---

## 💻 Computing Infrastructure

All experiments were conducted using **Jupyter Notebook environments** on **CPU-based systems**, including **Google Colab** and local machines.

- Operating Systems: **Windows** and **Linux**
- Python version: **≥ 3.8**
- GPU: **Not required**
- Typical memory: **8–12 GB RAM**
- Standard Python libraries used for training/evaluation/analysis, including (as applicable):
  - NumPy, Pandas, OpenCV, scikit-learn
  - FastDTW
  - SentenceTransformers / TF-IDF (for NLP similarity)

---

## 🔁 Reproducibility (How to Re-Run)

### 1) Clone the repository
```bash
git clone https://github.com/Pankhuri1999/AIUsageinAutism.git
cd AIUsageinAutism
```

### 2) Install dependencies (baseline)
```bash
pip install numpy pandas opencv-python scikit-learn fastdtw sentence-transformers
```

### 3) Reproduce Motion Therapy comparisons
- Open `MotionTherapy/MotionTherapyNotebook.ipynb` and run all cells (chang the directory/ path of videos), **OR**
- If a Python script is present, update `VIDEO1_PATH` and `VIDEO2_PATH` then run the script.

### 4) Reproduce Speech Therapy comparisons
- Open `SpeechTherapy/speechTherapyNotebook.ipynb` and run all cells (chang the directory/ path of videos), **OR**
- If a Python script is present, update `video1` and `video2` paths then run the script.

### 5) Reproduce Context + Emotion UI experiments
- Run the Python file in each folder to launch the UI:
```bash
python ContextInterpretation/contextInterpretation.py
python EmotionRecognitionActivity/emotionTherapy.py
```
- The corresponding CSV logs in the folder store the recorded experiment responses.

### 6) Reproduce analysis/validation plots
- Run the additional notebooks inside MotionTherapy/ and SpeechTherapy/ that contain validation/analysis used to support the experiments.

---


## 🎯 Intended Use

This framework is designed as a research prototype to:
- support therapists with objective scoring and progress monitoring,
- enable parents to track progress at home,
- support non-verbal children where language is a barrier.

**Not intended for clinical diagnosis.**

---

