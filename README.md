# EE200: Signals, Systems, and Networks - Course Project

This repository contains the complete project submission for the **EE200** course (Summer 2026) at the Indian Institute of Technology Kanpur. The project applies signal processing methodologies—ranging from frequency-domain filtering and edge detection to correlation-based anomaly detection and audio fingerprinting—to solve complex analytical tasks.

## Project Team
* **Devansh Chaturvedi** (Roll No: 240340)
* **Sthit Pragye** (Roll No: 241045)

---

## Module Overview

### Q1: Frequency Forensics & Digital Detective
**Frequency Forensics ('The Ghost Signal'):** Recovers images distorted by periodic interference using 2D DFT analysis and notch filtering.
**Digital Detective ('Missing Boundaries'):** Employs Sobel operators to compute intensity gradients and detect structural boundaries, utilizing Gaussian smoothing to suppress noise-induced false edges.

### Q2: The Midnight Episode ('Catching the Arrhythmia')
* Analyzes ECG recordings from Holter monitors to identify arrhythmia onset.
**Methodology:** Implements windowing for template generation and utilizes normalized cross-correlation to compare healthy heartbeats against the input recording, flagging structural deviations.

### Q3: Sonic Signatures ('Magical Mystery Tune')
**Signal Identification:** Builds an audio fingerprinting system using spectrograms and constellation peak hashing for robust song identification.
**Application ('Zapptain America'):** A deployed Streamlit-based interactive application that provides visual analysis, including spectrograms and offset histograms, for both single-clip and batch-mode processing.

---

## Repository Structure

```text
.
├── EE200_course_project_summer_2026.pdf
├── Q1/
│   ├── EE200_Project_Q1.ipynb
│   └── Q1_Report.pdf
├── Q1_data/
│   ├── ghost_signal_input.png
│   └── missing_boundaries_input.avif
├── Q2/
│   ├── Q2.pdf
│   ├── Question_2_g.ipynb
│   ├── Question_2_h.ipynb
│   ├── patient_ecg.npy
│   └── template.npy
├── Q2_data/
│   ├── patient_ecg.npy
│   └── template.npy
├── Q3/
│   ├── EE200_Question_3_Peaks_in_Pairs.ipynb
│   ├── EE200_Question_3_Single_Point_Peaks.ipynb
│   ├── Question_3.pdf
│   ├── app.py
│   ├── requirements.txt
│   ├── song_fingerprints_peaks_in_pairs.db
│   └── song_fingerprints_single_peaks.db
└── Q3_demo/
    ├── EE200_finalproject_2026_demo_video.mp4
    └── song_database.txt
