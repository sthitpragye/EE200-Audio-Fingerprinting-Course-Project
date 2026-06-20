# EE200: Signals, Systems, and Networks – Course Project

This repository contains the complete course project submission for **EE200: Signals, Systems, and Networks (Summer 2026)** at the **Indian Institute of Technology Kanpur**. The project applies fundamental signal processing techniques—including frequency-domain filtering, edge detection, correlation-based anomaly detection, and audio fingerprinting—to solve a diverse set of real-world analytical problems.

## Project Team

* **Devansh Chaturvedi** (Roll No: 240340)
* **Sthit Pragye** (Roll No: 241045)

---

## Live Deployment

Experience the **Zapptain America** audio fingerprinting system:

**https://ee200-music-fingerprint.streamlit.app**

> Note: If the application has been inactive, Streamlit may take a few moments to wake up.

---

## Project Overview

### Q1: Frequency Forensics & Digital Detective

#### Frequency Forensics – *The Ghost Signal*

Recover a corrupted image affected by periodic interference using frequency-domain analysis.

**Key Techniques**

* 2D Discrete Fourier Transform (DFT)
* Frequency spectrum analysis
* Notch filter design
* Inverse Fourier reconstruction

#### Digital Detective – *Missing Boundaries*

Detect structural boundaries in noisy images through gradient-based edge detection.

**Key Techniques**

* Gaussian smoothing
* Sobel edge operators
* Gradient magnitude computation
* Noise suppression and edge enhancement

---

### Q2: The Midnight Episode – *Catching the Arrhythmia*

Detect the onset of cardiac arrhythmias from ECG recordings obtained using Holter monitors.

**Methodology**

* Generate heartbeat templates using windowed ECG segments.
* Apply normalized cross-correlation to compare incoming signals with healthy heartbeat patterns.
* Identify structural deviations indicative of abnormal cardiac activity.

**Key Techniques**

* Signal windowing
* Template matching
* Normalized cross-correlation
* Anomaly detection in biomedical signals

---

### Q3: Sonic Signatures – *Magical Mystery Tune*

Develop a robust audio fingerprinting system capable of identifying songs from short audio clips.

#### Signal Identification

The system converts audio signals into spectrogram representations and extracts prominent spectral peaks to generate compact fingerprints.

**Key Techniques**

* Short-Time Fourier Transform (STFT)
* Spectrogram generation
* Constellation map extraction
* Peak hashing
* Fingerprint matching

#### Zapptain America

An interactive Streamlit application that demonstrates the complete audio identification pipeline.

**Features**

* Single-clip song identification
* Batch audio processing
* Spectrogram visualization
* Fingerprint matching statistics
* Offset histogram analysis
* Interactive user interface

---

## Getting Started

### Prerequisites

* Python 3.x
* Recommended: Virtual Environment (`venv`)

### Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

Install dependencies:

```bash
pip install -r Q3/requirements.txt
```

### Launch the Application

```bash
cd Q3
streamlit run app.py
```

The application will open automatically in your browser.

---

## Repository Structure

```text
.
├── README.md
├── EE200_course_project_summer_2026.pdf
│
├── Q1/
│   ├── EE200_Project_Q1.ipynb
│   └── Q1_Report.pdf
│
├── Q1_data/
│   ├── ghost_signal_input.png
│   └── missing_boundaries_input.avif
│
├── Q2/
│   ├── Q2.pdf
│   ├── Question_2_g.ipynb
│   ├── Question_2_h.ipynb
│   ├── patient_ecg.npy
│   └── template.npy
│
├── Q3/
│   ├── EE200_Question_3_Peaks_in_Pairs.ipynb
│   ├── EE200_Question_3_Single_Point_Peaks.ipynb
│   ├── Question_3.pdf
│   ├── app.py
│   ├── requirements.txt
│   ├── song_fingerprints_peaks_in_pairs.db
│   └── song_fingerprints_single_peaks.db
│
└── Q3_demo/
    ├── EE200_finalproject_2026_demo_video.mp4
    └── song_database.txt
```

---

## Technologies Used

* Python
* NumPy
* SciPy
* OpenCV
* Matplotlib
* Librosa
* SQLite
* Streamlit

---

## Course Information

**Course:** EE200 – Signals, Systems, and Networks
**Institution:** Indian Institute of Technology, Kanpur
**Semester:** Summer 2026
