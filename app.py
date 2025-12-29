import streamlit as st
import os
import subprocess
import tempfile
import numpy as np
from pydub import AudioSegment
import noisereduce as nr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# 設定頁面配置
st.set_page_config(page_title="Acoustic Noise Reduction Project", page_icon="📊", layout="wide")

# --- CSS 優化 (讓介面更乾淨) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


# --- 核心功能函數 ---

def extract_audio_from_video(video_path, output_audio_path):
    """Extract audio track from video using FFmpeg."""
    command = [
        "ffmpeg", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
        "-y", output_audio_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg Error: {e}")
        return False


def enhance_audio(input_path, output_path):
    """Read Audio -> Apply Noise Reduction -> Export."""
    try:
        sound = AudioSegment.from_file(input_path)
        samples = np.array(sound.get_array_of_samples())

        # 正規化數據 (float32)
        if sound.sample_width == 2:
            data = samples.astype(np.float32) / 32768.0
        elif sound.sample_width == 4:
            data = samples.astype(np.float32) / 2147483648.0
        else:
            data = samples.astype(np.float32)

        # 應用降噪算法
        reduced_noise_data = nr.reduce_noise(
            y=data,
            sr=sound.frame_rate,
            stationary=False,
            prop_decrease=0.95,  # 稍微提高消除比例以增強視覺對比
            n_std_thresh_stationary=1.5,
            time_constant_s=2.0,
        )

        # 轉回 int16
        reduced_noise_data = (reduced_noise_data * 32768.0).astype(np.int16)

        cleaned_sound = AudioSegment(
            reduced_noise_data.tobytes(),
            frame_rate=sound.frame_rate,
            sample_width=2,
            channels=1
        )
        cleaned_sound.export(output_path, format="mp3")
        return True
    except Exception as e:
        st.error(f"Processing Error: {str(e)}")
        return False


def plot_enhanced_spectrogram(file_path, title):
    """
    Plot spectrogram with Custom Hex Colors and HIGH-CONTRAST Black lines.
    """
    # Load data
    sound = AudioSegment.from_file(file_path)
    samples = np.array(sound.get_array_of_samples())
    if sound.channels == 2:
        samples = samples.reshape((-1, 2))[:, 0]

    # Normalize
    samples = samples.astype(np.float32)
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val

    # Create Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Set Background to White
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    # --- CHANGE: Create a custom High-Contrast Colormap ---
    # This creates a colormap that transitions from pure White to pure Black.
    # Loud sounds will now be drawn in solid black, making them pop out.
    colors = [(1, 1, 1), (0, 0, 0)]  # White -> Black
    cmap_name = 'high_contrast_wb'
    # N=256 gives a smooth transition, you can lower it for an even starker look
    cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    # Draw Spectrogram using the new custom colormap
    Pxx, freqs, bins, im = ax.specgram(
        samples,
        Fs=sound.frame_rate,
        NFFT=2048,
        noverlap=1024,
        cmap=cm,  # <--- Use the custom high-contrast map
        scale='dB',
        vmin=-80,
        vmax=0
    )

    # --- Custom Hex Color Configuration (Same as before) ---
    zones = [
        {"range": (0, 100), "color": "#8B4513", "label": "0-100Hz: Rumble (Noise)"},
        {"range": (100, 1000), "color": "#228B22", "label": "100-1k: Body (Fundamental)"},
        {"range": (1000, 4000), "color": "#FFD700", "label": "1k-4k: Intelligibility"},
        {"range": (4000, 22050), "color": "#DC143C", "label": ">4k: Air (Sibilance)"}
    ]

    # Draw colored overlays
    for zone in zones:
        # alpha=0.25 is good, but you can lower it to 0.2 if the lines are still obscure
        ax.axhspan(zone["range"][0], zone["range"][1], color=zone["color"], alpha=0.25)

    # Text Styling
    ax.set_title(title, color='black', fontsize=14, pad=20)
    ax.set_ylabel('Frequency (Hz)', color='black')
    ax.set_xlabel('Time (s)', color='black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.set_ylim(0, 10000)

    # Custom Legend
    legend_patches = [mpatches.Patch(color=z["color"], label=z["label"], alpha=0.5) for z in zones]
    ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=2, facecolor='#f0f0f0', labelcolor='black')

    # Colorbar (Will now show White to Black)
    cbar = fig.colorbar(im, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

    return fig


# --- 網站主邏輯 ---

st.markdown("<h1 class='main-header'>🎵 Acoustic Project: Video Denoising & Spectral Visualization</h1>", unsafe_allow_html=True)
st.markdown("""
This tool visualizes the effect of **AI Noise Reduction** on the audio spectrum. We divide the frequency into 4 acoustic zones:
* **🟤 0~100Hz (Rumble)**: Low-end noise, wind, AC hum; usually needs removal.
* **🟢 100~1000Hz (Body)**: Fundamental frequencies and thickness of the voice.
* **🟡 1000~4000Hz (Intelligibility)**: The most sensitive area for human ears; affects speech clarity.
* **🔴 >4000Hz (Air)**: High-frequency details and background hiss.
""")

uploaded_file = st.file_uploader("📂 Upload File (Support .mp4, .mov, .wav, .mp3)", type=["mov", "mp4", "mp3", "wav"])

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, uploaded_file.name)
        extracted_audio_path = os.path.join(temp_dir, "extracted_raw.wav")
        final_output_path = os.path.join(temp_dir, "cleaned_output.mp3")

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Determine process flow
        file_extension = os.path.splitext(input_path)[1].lower()
        is_video = file_extension in ['.mov', '.mp4', '.avi', '.mkv']

        status_box = st.status("🚀 System Processing...", expanded=True)

        # 1. Extraction / Loading
        if is_video:
            status_box.write("Extracting audio from video...")
            success = extract_audio_from_video(input_path, extracted_audio_path)
            processing_source = extracted_audio_path
        else:
            status_box.write("Loading audio file...")
            processing_source = input_path
            success = True

        if success:
            # 2. Denoising
            status_box.write("Applying AI Noise Reduction...")
            enhancement_success = enhance_audio(processing_source, final_output_path)

            if enhancement_success:
                status_box.update(label="✅ Processing Complete!", state="complete", expanded=False)

                # --- Results Display ---
                col1, col2 = st.columns(2)

                # Left: Original
                with col1:
                    st.subheader("🎧 Original Audio (Raw)")
                    if is_video:
                        with open(input_path, "rb") as f:
                            st.video(f.read())
                    else:
                        st.audio(processing_source)

                    st.markdown("**Original Spectrogram**")
                    with st.spinner("Rendering Original Plot..."):
                        fig_orig = plot_enhanced_spectrogram(processing_source, "Original Audio Spectrogram")
                        st.pyplot(fig_orig)

                # Right: Processed
                with col2:
                    st.subheader("🎹 Denoised Audio")
                    with open(final_output_path, "rb") as f:
                        processed_bytes = f.read()
                    st.audio(processed_bytes, format='audio/mp3')

                    st.markdown("**Denoised Spectrogram**")
                    st.info("💡 Note: Observe if the **Brown Zone (Rumble)** turns black. This indicates noise removal.")
                    with st.spinner("Rendering Denoised Plot..."):
                        fig_clean = plot_enhanced_spectrogram(final_output_path, "Cleaned Audio Spectrogram")
                        st.pyplot(fig_clean)

                    st.download_button(
                        label="📥 Download Cleaned MP3",
                        data=processed_bytes,
                        file_name="enhanced_audio.mp3",
                        mime="audio/mp3"
                    )

st.markdown("---")
st.caption("Powered by Streamlit, FFmpeg & Noisereduce")