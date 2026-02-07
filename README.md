# AI-Supported Autism Therapy Monitoring  
### Methods for Non-Verbal and Speech Interpretation and Scoring

This repository presents an AI-assisted framework designed to support autism therapy monitoring through structured non-verbal and speech-based interpretation tasks. The system provides automated scoring, response logging, and performance tracking to assist therapists and parents in observing continuous behavioral progress.

The project focuses on four interactive therapy modules:

- Motion Therapy  
- Speech Therapy  
- Context Interpretation  
- Emotion Recognition  

Each module captures child responses, computes similarity metrics, and records behavioral indicators such as latency, consistency, and score stability.

---

## 📁 Repository Structure

AIUsageinAutism/

│
├── motion_therapy/

│ └── Jupyter notebooks + sample logs

│
├── speech_therapy/

│ └── Jupyter notebooks + sample logs

│
├── context_interpretation/

│ └── Jupyter notebooks + CSV response files

│
├── emotion_recognition/

│ └── Jupyter notebooks + CSV response files

│
├── datasets/

│ └── Sample CSV files for interpretation and emotion tasks

│
└── README.md


Each folder contains an independent Jupyter Notebook implementing the respective experiment.

---

## 🧪 Experiments Overview

### 1. Motion Therapy
Evaluates controlled arm and facial movements using temporal similarity metrics (FastDTW). Logs angular deviations and response stability.

### 2. Speech Therapy
Compares spoken responses against reference phrases and measures articulation similarity and reaction time.

### 3. Context Interpretation
Assesses semantic understanding of visual scenarios. Child responses are logged and compared using NLP-based similarity scoring.

### 4. Emotion Recognition
Measures emotion identification accuracy from visual prompts and records consistency across trials.

All experiments generate structured CSV logs capturing:

- Timestamp  
- Trial number  
- Prompt ID  
- Child response  
- Similarity score  
- Processing latency  
- Running averages  

---

## 💻 Computing Infrastructure

All experiments were conducted using Jupyter Notebook environments on CPU-based systems, including Google Colab and local machines. The operating systems used were Windows and Linux, with Python version ≥ 3.8. GPU acceleration was not required.

Typical system specifications included:

- RAM: 8–12 GB  
- CPU: Standard laptop/desktop processors  
- Environment: Jupyter Notebook  

All models and analysis pipelines were implemented using standard Python libraries including:

- OpenCV  
- NumPy  
- Pandas  
- scikit-learn  
- FastDTW  
- SentenceTransformers / TF-IDF  

---

## 🔐 Data Availability

The motion and speech therapy datasets cannot be publicly released as they contain the authors’ own facial expressions, hand gestures, and voice recordings.

CSV logs for the Context Interpretation and Emotion Recognition experiments are included in their respective folders.

---

## 🎯 Intended Use

This framework is designed as a research prototype to:

- Assist therapists with objective behavioral scoring  
- Enable parents to track progress at home  
- Support non-verbal children through automated interpretation tasks  
- Provide quantitative metrics for therapy monitoring  

It is not intended as a diagnostic tool.

---

## 🚀 Getting Started

1. Clone the repository:

```bash
git clone https://github.com/Pankhuri1999/AIUsageinAutism.git
