import os
import tempfile
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
from scipy import signal
from scipy.ndimage import maximum_filter
import sqlite3

# Configuring system
st.set_page_config(
    page_title="EE200 · Audio Fingerprinting",
    layout="wide",
    page_icon="◈",
    initial_sidebar_state="collapsed",
)

DB_PATH = "song_fingerprints.db"
MIN_DELTA_T = 0.1
MAX_DELTA_T = 1.0
NPERSEG = 1024
NOVERLAP = 512

# Frontend
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #050B18 !important;
    color: #C8D6E5 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* HERO */
.hero {
    background: linear-gradient(135deg, #050B18 0%, #091325 50%, #050B18 100%);
    border-bottom: 1px solid #0F2040;
    padding: 28px 48px 20px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(90deg, transparent, transparent 59px, rgba(15,32,64,0.4) 60px);
    pointer-events: none;
}
.hero-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 3px;
    color: #F5A623;
    text-transform: uppercase;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.hero-badge::before {
    content: '';
    display: inline-block;
    width: 6px;
    height: 6px;
    background: #F5A623;
    border-radius: 50%;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.6); }
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -1px;
    color: #F0F6FF;
    margin: 0 0 4px;
    line-height: 1.1;
}
.hero-title span { color: #F5A623; }
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #3D5A80;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.scanline-wrap {
    height: 3px;
    background: #0F2040;
    border-radius: 2px;
    overflow: hidden;
    margin-top: 18px;
    position: relative;
}
.scanline-bar {
    height: 100%;
    width: 30%;
    background: linear-gradient(90deg, transparent, #F5A623, #FFE299, #F5A623, transparent);
    animation: scan 3s ease-in-out infinite;
    border-radius: 2px;
}
@keyframes scan {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(430%); }
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0 !important;
    padding: 0 48px !important;
    border-bottom: 1px solid #0F2040 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #3D5A80 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 16px 28px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
    transition: all 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #C8D6E5 !important; }
.stTabs [aria-selected="true"] {
    color: #F5A623 !important;
    border-bottom-color: #F5A623 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }

/* SECTION LABEL */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 3px;
    color: #3D5A80;
    text-transform: uppercase;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #0F2040;
}

/* UPLOAD */
[data-testid="stFileUploader"] {
    background: #08111F !important;
    border: 1px dashed #1A3A5C !important;
    border-radius: 8px !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #F5A623 !important; }
[data-testid="stFileUploadDropzone"] { background: transparent !important; border: none !important; }
[data-testid="stFileUploadDropzone"] p {
    color: #3D5A80 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}

/* BUTTONS */
.stButton > button, .stDownloadButton > button {
    background: #F5A623 !important;
    color: #050B18 !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 10px 28px !important;
    transition: all 0.15s !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #FFD07A !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(245,166,35,0.3) !important;
}

/* LIB CARDS */
.lib-card {
    background: #08111F;
    border: 1px solid #0F2040;
    border-radius: 8px;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.lib-card:hover { border-color: #1A3A5C; transform: translateY(-2px); }
.lib-card-body { padding: 12px 14px; }
.lib-card-name {
    color: #E8F0FC; font-size: 13px; font-weight: 600;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 4px;
}
.lib-card-count { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #F5A623; }

/* PIPELINE */
.pipeline-rail {
    display: flex;
    align-items: stretch;
    background: #0F2040;
    border: 1px solid #0F2040;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 32px;
    gap: 1px;
}
.pipeline-step {
    flex: 1;
    background: #08111F;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 3px;
}
.pipeline-step-total { flex: 0.8; background: #040D1A; }
.pipeline-step-num { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #1A3A5C; }
.pipeline-step-name { font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 1px; text-transform: uppercase; color: #3D5A80; }
.pipeline-step-time { font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 700; color: #F5A623; }
.pipeline-step-time-teal { color: #1AFFD5; }
.pipeline-step-detail { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #1A3A5C; }

/* MATCH BANNER */
.match-banner {
    background: #040D1A;
    border: 1px solid #0F3020;
    border-left: 4px solid #1AFFD5;
    border-radius: 8px;
    padding: 28px 32px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.match-banner::after {
    content: 'LOCK\00B7ON';
    position: absolute;
    top: 16px;
    right: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 3px;
    color: #1AFFD5;
    opacity: 0.5;
}
.match-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: #1AFFD5; margin-bottom: 10px; }
.match-song-name { font-size: 2.8rem; font-weight: 700; color: #F0F6FF; letter-spacing: -1px; line-height: 1.1; margin-bottom: 14px; }
.match-score-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(26,255,213,0.08); border: 1px solid rgba(26,255,213,0.2);
    border-radius: 100px; padding: 6px 14px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #1AFFD5;
}
.match-dot { width: 8px; height: 8px; background: #1AFFD5; border-radius: 50%; display: inline-block; }

/* STEP HEADERS */
.step-header { display: flex; align-items: center; gap: 16px; margin: 32px 0 16px; }
.step-num-badge {
    font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700;
    color: #050B18; background: #F5A623; width: 24px; height: 24px;
    border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.step-title { font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: #3D5A80; }
.step-divider { flex: 1; height: 1px; background: #0F2040; }

/* CANDIDATES */
.candidate-row {
    display: flex; align-items: center; gap: 16px;
    padding: 10px 16px; background: #08111F;
    border: 1px solid #0F2040; border-radius: 6px; margin-bottom: 8px;
}
.candidate-rank { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #1A3A5C; width: 20px; flex-shrink: 0; }
.candidate-name { font-size: 13px; font-weight: 500; color: #C8D6E5; flex: 1; }
.candidate-bar-wrap { width: 120px; height: 4px; background: #0F2040; border-radius: 2px; overflow: hidden; }
.candidate-bar { height: 100%; border-radius: 2px; background: #1A3A5C; }
.candidate-score { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #3D5A80; width: 80px; text-align: right; }

/* INFO BOX */
.info-box { background: #08111F; border: 1px solid #0F2040; border-radius: 8px; padding: 16px 20px; margin-bottom: 16px; }
.info-box p { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #3D5A80; margin: 0; line-height: 1.7; }

/* EMPTY STATE */
.empty-state { text-align: center; padding: 60px 40px; background: #08111F; border: 1px dashed #0F2040; border-radius: 8px; }
.empty-state-icon { font-size: 2.5rem; margin-bottom: 12px; opacity: 0.3; }
.empty-state-text { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #1A3A5C; letter-spacing: 1px; }

/* STAT CARD */
.stat-card { background: #08111F; border: 1px solid #0F2040; border-radius: 8px; padding: 16px 20px; }
.stat-card-label { font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 2px; color: #3D5A80; margin-bottom: 4px; }
.stat-card-value { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; color: #F5A623; }

/* STREAMLIT OVERRIDES */
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; color: #F5A623 !important; }
[data-testid="stMetricLabel"] { font-family: 'JetBrains Mono', monospace !important; font-size: 10px !important; letter-spacing: 2px !important; text-transform: uppercase !important; color: #3D5A80 !important; }
.stSuccess { background: #040D1A !important; border-color: #1AFFD5 !important; color: #1AFFD5 !important; font-family: 'JetBrains Mono', monospace !important; }
.stInfo { background: #08111F !important; border-color: #1A3A5C !important; color: #3D5A80 !important; }
.stDataFrame { background: #08111F !important; border: 1px solid #0F2040 !important; border-radius: 8px !important; }
.stProgress > div > div { background: #F5A623 !important; }
div[data-testid="stProgressBar"] > div { background: #F5A623 !important; }
audio { width: 100%; border-radius: 4px; margin-bottom: 16px; }

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-badge">EE200 &nbsp;·&nbsp; Signals, Systems &amp; Networks &nbsp;</div>
    <div class="hero-title">Audio <span>Fingerprinting</span></div>
    <div class="hero-sub">Shazam-style identification via spectrogram constellation hashing</div>
    <div class="scanline-wrap"><div class="scanline-bar"></div></div>
</div>
""", unsafe_allow_html=True)

BG    = "#050B18"
PANEL = "#08111F"
GRID  = "#0F2040"
AMBER = "#F5A623"
TEAL  = "#1AFFD5"
MUTED = "#3D5A80"
TEXT  = "#C8D6E5"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID, "text.color": TEXT, "font.family": "monospace",
})


# Database & Audio Processing

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS fingerprints
                 (f1 REAL, f2 REAL, delta_t REAL, song_name TEXT, t1 REAL)''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints (f1, f2, delta_t)')
    conn.commit(); conn.close()

def get_peaks(spec, frequencies, times):
    spec_mag = 20 * np.log10(spec + 1e-10)
    threshold = np.max(spec_mag) - 60
    neighborhood = (15, 15)
    mask = (maximum_filter(spec_mag, size=neighborhood) == spec_mag) & (spec_mag > threshold)
    fi, ti = np.where(mask)
    return list(zip(times[ti], frequencies[fi]))

def add_song(file_path, song_name):
    conn = sqlite3.connect(DB_PATH)
    with conn:
        conn.cursor().execute('DELETE FROM fingerprints WHERE song_name = ?', (song_name,))
    conn.close()
    song, sr = librosa.load(file_path, sr=None)
    freqs, times, spec = signal.spectrogram(song, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    peaks = sorted(get_peaks(spec, freqs, times), key=lambda x: x[0])
    buf = []
    for i in range(len(peaks)):
        t1, f1 = peaks[i]
        for j in range(i+1, len(peaks)):
            t2, f2 = peaks[j]
            dt = t2 - t1
            if dt > MAX_DELTA_T: break
            if dt < MIN_DELTA_T: continue
            buf.append((round(f1,1), round(f2,1), round(dt,2), song_name, round(t1,2)))
    if buf:
        conn = sqlite3.connect(DB_PATH)
        with conn: conn.cursor().executemany('INSERT INTO fingerprints VALUES (?,?,?,?,?)', buf)
        conn.close()
    return len(buf)

def identify_song(file_path):
    song, sr = librosa.load(file_path, sr=None)

    t0 = time.time()
    freqs, times, spec = signal.spectrogram(song, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    t_spec = time.time() - t0

    t0 = time.time()
    peaks = sorted(get_peaks(spec, freqs, times), key=lambda x: x[0])
    t_const = time.time() - t0

    t0 = time.time()
    hashes = []
    for i in range(len(peaks)):
        t1, f1 = peaks[i]
        for j in range(i+1, len(peaks)):
            t2, f2 = peaks[j]
            dt = t2 - t1
            if dt > MAX_DELTA_T: break
            if dt < MIN_DELTA_T: continue
            hashes.append((round(f1,1), round(f2,1), round(dt,2), t1))
    t_hash = time.time() - t0

    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT song_name) FROM fingerprints")
    n_tracks = cursor.fetchone()[0]
    matches = {}
    for f1, f2, dt, t1 in hashes:
        cursor.execute('SELECT song_name, t1 FROM fingerprints WHERE f1=? AND f2=? AND delta_t=?',
                       (f1, f2, dt))
        for sn, t1_db in cursor.fetchall():
            matches.setdefault(sn, []).append(round(t1_db - t1, 2))
    conn.close()
    t_lookup = time.time() - t0

    t0 = time.time()
    best_song, highest_density, winning_offsets, offset0 = "Unknown", 0, [], 0
    candidates = []
    for sn, offsets in matches.items():
        if not offsets: continue
        counts, bins = np.histogram(offsets, bins=200)
        d = int(np.max(counts))
        peak_offset = float(bins[np.argmax(counts)])
        candidates.append((sn, d))
        if d > highest_density:
            highest_density = d; best_song = sn
            winning_offsets = offsets; offset0 = peak_offset
    candidates.sort(key=lambda x: x[1], reverse=True)
    t_score = time.time() - t0

    timing = {
        "spectrogram": t_spec, "constellation": t_const,
        "hashing": t_hash, "db_lookup": t_lookup, "scoring": t_score,
        "n_hashes": len(hashes), "n_peaks": len(peaks),
        "n_tracks": n_tracks, "offset0": round(offset0, 2),
    }
    return best_song, winning_offsets, highest_density, candidates, freqs, times, spec, peaks, timing

def to_tmp(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue()); return tmp.name
import base64

def custom_audio_player(uploaded_file):
    
    bytes_data = uploaded_file.getvalue()
    b64 = base64.b64encode(bytes_data).decode()
    
    file_name = uploaded_file.name.lower()
    mime_type = "audio/wav" if file_name.endswith('.wav') else "audio/mpeg"

    player_html = f"""
    <style>
    .custom-audio-wrapper {{
        background: #08111F;
        border: 1px solid #1A3A5C;
        border-left: 4px solid #F5A623;
        border-radius: 8px;
        padding: 12px 20px;
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    .custom-audio-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #C8D6E5;
        text-transform: uppercase;
        letter-spacing: 2px;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }}
    .custom-audio-label span {{
        color: #3D5A80;
        font-size: 9px;
    }}
    /* Style the native controls to blend in */
    .custom-audio-wrapper audio {{
        flex: 1;
        height: 36px;
        outline: none;
    }}
    .custom-audio-wrapper audio::-webkit-media-controls-panel {{
        background-color: #040D1A;
    }}
    .custom-audio-wrapper audio::-webkit-media-controls-current-time-display,
    .custom-audio-wrapper audio::-webkit-media-controls-time-remaining-display {{
        color: #1AFFD5;
        font-family: 'JetBrains Mono', monospace;
    }}
    </style>
    
    <div class="custom-audio-wrapper">
        <div class="custom-audio-label">
            Query Audio
            <span>{uploaded_file.name}</span>
        </div>
        <audio controls src="data:{mime_type};base64,{b64}"></audio>
    </div>
    """
    st.markdown(player_html, unsafe_allow_html=True)

def get_db_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT song_name, COUNT(*) FROM fingerprints GROUP BY song_name")
    data = c.fetchall(); conn.close(); return data

# Plots
def spec_fig(freqs, times, spec, max_freq=8000):
    spec_db = 20 * np.log10(spec + 1e-10)
    fmask = freqs <= max_freq
    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=BG)
    ax.set_facecolor(PANEL)
    img = ax.pcolormesh(times, freqs[fmask], spec_db[fmask,:], shading="auto", cmap="inferno", rasterized=True)
    cb = fig.colorbar(img, ax=ax, fraction=0.025, pad=0.01)
    cb.ax.tick_params(colors=MUTED, labelsize=8); cb.set_label("dB", color=MUTED, fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9); ax.set_ylabel("Frequency (Hz)", fontsize=9)
    ax.tick_params(labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    fig.tight_layout(pad=0.8)
    return fig

def const_fig(peaks, max_freq=8000):
    ts = [p[0] for p in peaks if p[1] <= max_freq]
    fs = [p[1] for p in peaks if p[1] <= max_freq]
    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=BG)
    ax.set_facecolor(PANEL)
    sc = ax.scatter(ts, fs, s=5, c=fs, cmap="plasma", alpha=0.85, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
    cb.ax.tick_params(colors=MUTED, labelsize=8); cb.set_label("Hz", color=MUTED, fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9); ax.set_ylabel("Frequency (Hz)", fontsize=9)
    ax.tick_params(labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    fig.tight_layout(pad=0.8)
    return fig

def db_match_fig(song_name, offset0, query_dur):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT DISTINCT t1, f1 FROM fingerprints WHERE song_name=?", (song_name,))
    data = c.fetchall()
    conn.close()
    
    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=BG)
    ax.set_facecolor(PANEL)
    
    if not data:
        ax.text(0.5, 0.5, "No database points found", color=MUTED, ha='center', va='center')
        return fig
        
    ts = [row[0] for row in data]
    fs = [row[1] for row in data]
    
    ax.scatter(ts, fs, s=3, c=TEAL, alpha=0.5, linewidths=0)
    
    ax.axvspan(offset0, offset0 + query_dur, color=AMBER, alpha=0.15, zorder=0)
    ax.axvline(offset0, color=AMBER, linestyle="--", linewidth=1.5)
    ax.axvline(offset0 + query_dur, color=AMBER, linestyle="--", linewidth=1.5)
    
    ax.set_ylim(bottom=0, top=np.max(fs) * 1.25)
    ax.text(offset0 + (query_dur / 2), np.max(fs) * 1.1, "QUERY MATCH", 
            color=AMBER, fontsize=9, fontweight="bold", ha="center", va="center",
            bbox=dict(facecolor=BG, edgecolor=AMBER, boxstyle='round,pad=0.3', alpha=0.9))
            
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", fontsize=9)
    ax.tick_params(labelsize=8)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    for sp in ("bottom","left"): ax.spines[sp].set_edgecolor(GRID)
    fig.tight_layout(pad=0.8)
    return fig

def hist_fig(offsets, song_name):
    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=BG)
    ax.set_facecolor(PANEL)
    if offsets:
        counts, bin_edges = np.histogram(offsets, bins=200)
        bw = bin_edges[1] - bin_edges[0]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        max_idx = int(np.argmax(counts))
        colors = [TEAL if i == max_idx else "#1A3A5C" for i in range(len(counts))]
        ax.bar(bin_centers, counts, width=bw*0.9, color=colors, linewidth=0)
        ax.axvline(bin_centers[max_idx], color=TEAL, linewidth=1, linestyle="--", alpha=0.5)
        ax.annotate(f"{counts[max_idx]} aligned",
                    xy=(bin_centers[max_idx], counts[max_idx]),
                    xytext=(18, -8), textcoords="offset points",
                    color=TEAL, fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2))
    ax.set_xlabel("Offset  t_db \u2212 t_query  (s)", fontsize=9)
    ax.set_ylabel("Hash alignment count", fontsize=9)
    ax.tick_params(labelsize=8)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    for sp in ("bottom","left"): ax.spines[sp].set_edgecolor(GRID)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.tight_layout(pad=0.8)
    return fig


init_db()
tab1, tab2, tab3 = st.tabs(["\u25c8  LIBRARY", "\u25ce  IDENTIFY", "\u25a4  BATCH"])


# Tab 1 : Library

with tab1:
    st.markdown('<div style="padding: 32px 48px;">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Index tracks</div>', unsafe_allow_html=True)

    uploaded_db = st.file_uploader(
        "Drop audio files to fingerprint",
        type=["mp3","wav"], accept_multiple_files=True, key="lib_upload",
        label_visibility="collapsed"
    )
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
        'color:#1A3A5C;margin:8px 0 20px;">WAV \u00b7 MP3 \u00b7 FLAC &nbsp;|&nbsp; '
        'Each track is hashed into a SQLite constellation database</p>',
        unsafe_allow_html=True
    )

    if st.button("\u25b6  INDEX FILES", key="idx_btn"):
        if not uploaded_db:
            st.warning("No files selected.")
        else:
            for f in uploaded_db:
                tmp = to_tmp(f)
                name = os.path.splitext(f.name)[0]
                with st.spinner(f"Hashing {name}..."):
                    n = add_song(tmp, name)
                os.remove(tmp)
                st.success(f"\u2713  {name}  \u00b7  {n:,} hashes stored")

    st.markdown('<div class="section-label" style="margin-top:36px;">In the database</div>',
                unsafe_allow_html=True)

    db_stats = get_db_stats()
    if db_stats:
        cols = st.columns(4)
        for i, (name, count) in enumerate(db_stats):
            with cols[i % 4]:
                st.markdown(f"""
                <div class="lib-card">
                    <div class="lib-card-body">
                        <div class="lib-card-name" title="{name}">{name}</div>
                        <div class="lib-card-count">{count:,} hashes</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">\u25c8</div>
            <div class="empty-state-text">Database is empty<br>Upload files above to begin indexing</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# Tab 2: Single song identification

with tab2:
    st.markdown('<div style="padding: 32px 48px;">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Query</div>', unsafe_allow_html=True)

    query_file = st.file_uploader(
        "Upload a clip to identify",
        type=["mp3","wav"], key="query_upload",
        label_visibility="collapsed"
    )
    
    if query_file:
        custom_audio_player(query_file)

    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
        'color:#1A3A5C;margin:8px 0 20px;">'
        'Works on noisy recordings, partial clips, and pitch-shifted versions</p>',
        unsafe_allow_html=True
    )

    run_id = st.button("\u25ce  IDENTIFY", type="primary", key="id_btn")

    if run_id and query_file:
        st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
        t_wall = time.time()
        tmp = to_tmp(query_file)

        with st.spinner("Running fingerprint pipeline..."):
            best, offsets, score, candidates, freqs, times, spec, peaks, timing = identify_song(tmp)
        os.remove(tmp)
        
        query_duration = times[-1] if len(times) > 0 else 0
        total_ms = int((time.time() - t_wall) * 1000)

        def ms(s): return f"{int(s*1000)} ms"

        st.markdown(f"""
        <div class="pipeline-rail">
            <div class="pipeline-step">
                <div class="pipeline-step-num">01</div>
                <div class="pipeline-step-name">Spectrogram</div>
                <div class="pipeline-step-time">{ms(timing['spectrogram'])}</div>
                <div class="pipeline-step-detail">STFT \u00b7 {NPERSEG} window</div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-step-num">02</div>
                <div class="pipeline-step-name">Constellation</div>
                <div class="pipeline-step-time">{ms(timing['constellation'])}</div>
                <div class="pipeline-step-detail">{timing['n_peaks']} peaks</div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-step-num">03</div>
                <div class="pipeline-step-name">Hashing</div>
                <div class="pipeline-step-time">{ms(timing['hashing'])}</div>
                <div class="pipeline-step-detail">{timing['n_hashes']:,} pairs</div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-step-num">04</div>
                <div class="pipeline-step-name">DB Lookup</div>
                <div class="pipeline-step-time">{ms(timing['db_lookup'])}</div>
                <div class="pipeline-step-detail">{timing['n_tracks']} tracks</div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-step-num">05</div>
                <div class="pipeline-step-name">Scoring</div>
                <div class="pipeline-step-time">{ms(timing['scoring'])}</div>
                <div class="pipeline-step-detail">offset {timing['offset0']} s</div>
            </div>
            <div class="pipeline-step pipeline-step-total">
                <div class="pipeline-step-num">\u03a3</div>
                <div class="pipeline-step-name">Total</div>
                <div class="pipeline-step-time pipeline-step-time-teal">{total_ms} ms</div>
                <div class="pipeline-step-detail">wall time</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if best == "Unknown" or score < 5:
            st.markdown(f"""
            <div style="background:#1A0808;border:1px solid #5C1A1A;border-left:4px solid #FF4444;
                        border-radius:8px;padding:24px 28px;margin-bottom:28px;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                            letter-spacing:3px;color:#FF6666;margin-bottom:8px;">NO MATCH</div>
                <div style="font-size:1.4rem;font-weight:600;color:#C8D6E5;">
                    No song identified in the database</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                            color:#5C1A1A;margin-top:8px;">
                    Alignment score: {score} \u2014 threshold is 5</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="match-banner">
                <div class="match-eyebrow">Match found</div>
                <div class="match-song-name">{best}</div>
                <span class="match-score-pill">
                    <span class="match-dot"></span>
                    {score} aligned hashes
                </span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="step-header">
                <div class="step-num-badge">1</div>
                <div class="step-title">Raw Spectrogram</div>
                <div class="step-divider"></div>
            </div>
            """, unsafe_allow_html=True)
            st.pyplot(spec_fig(freqs, times, spec), clear_figure=True)

            st.markdown("""
            <div class="step-header">
                <div class="step-num-badge">2</div>
                <div class="step-title">Constellation Map \u2014 Local Maxima</div>
                <div class="step-divider"></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
                <p>Each dot is a spectrogram peak surviving the 60-point maximum filter.
                   Color encodes frequency. These become the anchor points for hash pair generation.</p>
            </div>
            """, unsafe_allow_html=True)
            st.pyplot(const_fig(peaks), clear_figure=True)

            st.markdown("""
            <div class="step-header">
                <div class="step-num-badge">3</div>
                <div class="step-title">Database Search \u2014 Where in the song?</div>
                <div class="step-divider"></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
                <p>The query hashes were checked against the database. Below is the full fingerprint 
                   of the matching song reconstructed from the database. The highlighted window shows 
                   exactly where the query clip sits inside the full track.</p>
            </div>
            """, unsafe_allow_html=True)
            st.pyplot(db_match_fig(best, timing['offset0'], query_duration), clear_figure=True)

            st.markdown("""
            <div class="step-header">
                <div class="step-num-badge">4</div>
                <div class="step-title">Offset Histogram \u2014 The Alignment Spike</div>
                <div class="step-divider"></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
                <p>For a genuine match, thousands of independent hash pairs agree on a single
                   time offset (t_db \u2212 t_query). The spike height is the match score.
                   Random collisions produce a flat background; a real song produces one sharp peak.</p>
            </div>
            """, unsafe_allow_html=True)
            st.pyplot(hist_fig(offsets, best), clear_figure=True)

    elif run_id and not query_file:
        st.warning("Upload a clip first.")

    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Batch processing

with tab3:
    st.markdown('<div style="padding: 32px 48px;">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Batch evaluation</div>', unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload query clips for batch testing",
        type=["mp3","wav"], accept_multiple_files=True, key="batch_upload",
        label_visibility="collapsed"
    )
   
    if batch_files:
        
        with st.expander(f"🎵 Preview {len(batch_files)} Uploaded Tracks", expanded=False):
            st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
            for f in batch_files:
                custom_audio_player(f)

    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
        'color:#1A3A5C;margin:8px 0 20px;">'
        'Run multiple queries in sequence \u2014 useful for measuring recall rate across a test set</p>',
        unsafe_allow_html=True
    )

    if st.button("\u25a4  RUN BATCH", type="primary", key="batch_btn") and batch_files:
        rows = []
        bar = st.progress(0.0)
        status = st.empty()

        for i, qf in enumerate(batch_files):
            status.markdown(
                f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                f'color:#3D5A80;">Processing {qf.name}...</p>',
                unsafe_allow_html=True
            )
            tmp = to_tmp(qf)
            t0 = time.time()
            best, _, score, candidates, *_ = identify_song(tmp)
            elapsed = int((time.time()-t0)*1000)
            os.remove(tmp)

            prediction = best if score >= 5 else "\u2014 no match \u2014"
            rows.append({"File": qf.name, "Prediction": prediction, "Score": score, "Time (ms)": elapsed})
            bar.progress((i+1) / len(batch_files))

        bar.empty(); status.empty()

        n_matched = sum(1 for r in rows if r["Prediction"] != "\u2014 no match \u2014")
        avg_score = int(np.mean([r["Score"] for r in rows])) if rows else 0
        avg_time  = int(np.mean([r["Time (ms)"] for r in rows])) if rows else 0

        c1, c2, c3 = st.columns(3)
        for col, label, val in zip(
            [c1, c2, c3],
            ["MATCHED", "AVG SCORE", "AVG LATENCY"],
            [f"{n_matched}/{len(rows)}", str(avg_score), f"{avg_time} ms"]
        ):
            with col:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-card-label">{label}</div>
                    <div class="stat-card-value">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
        
        df_results = pd.DataFrame(rows)[["File", "Prediction", "Score", "Time (ms)"]]
        st.dataframe(df_results, use_container_width=True, hide_index=True)
        
        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
        st.download_button(
            label="⬇ DOWNLOAD RESULTS.CSV",
            data=csv_data,
            file_name="batch_results.csv",
            mime="text/csv",
        )

    st.markdown('</div>', unsafe_allow_html=True)