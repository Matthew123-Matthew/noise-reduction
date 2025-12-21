import streamlit as st
import os
import subprocess
import tempfile
import numpy as np
from pydub import AudioSegment
from pydub.utils import mediainfo
import noisereduce as nr
from scipy.io import wavfile

# 設定頁面配置
st.set_page_config(page_title="音訊降噪與增強工具", page_icon="🎵", layout="centered")


def extract_audio_from_video(video_path, output_audio_path):
    """
    使用 FFmpeg 從影片中分離音軌 (整合使用者原本的邏輯)
    """
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",  # 轉為 wav 格式以便後續處理
        "-ar", "44100",  # 設定採樣率
        "-ac", "1",  # 轉為單聲道 (降噪效果通常較好)
        "-y",  # 覆蓋已存在文件
        output_audio_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg 錯誤: {e}")
        return False

def enhance_audio(input_path, output_path):
    """
    針對人聲優化版：讀取音訊 -> 強力降噪 -> 輸出
    """
    try:
        # 1. 使用 Pydub 讀取音訊
        sound = AudioSegment.from_file(input_path)

        # 轉換為 numpy array 以便進行數學運算
        samples = np.array(sound.get_array_of_samples())

        # 正規化數據到 -1.0 到 1.0 之間 (noisereduce 需要 float32)
        if sound.sample_width == 2:
            data = samples.astype(np.float32) / 32768.0
        elif sound.sample_width == 4:
            data = samples.astype(np.float32) / 2147483648.0
        else:
            data = samples.astype(np.float32)

        # 2. 應用降噪算法 (針對人聲優化參數)
        # stationary=False: 啟用非穩態降噪，這對有背景說話聲的影片更有效，但處理時間會變長
        reduced_noise_data = nr.reduce_noise(
            y=data,
            sr=sound.frame_rate,
            stationary=False,  # 關鍵修改：不假設噪音是固定的
            prop_decrease=0.9, # 消除 90% 的偵測噪音
            n_std_thresh_stationary=1.5, # 增加判斷閾值
            time_constant_s=2.0, # 平滑處理
        )

        # 將數據轉回 int16 以便 Pydub 讀取
        reduced_noise_data = (reduced_noise_data * 32768.0).astype(np.int16)

        # 重建 AudioSegment
        cleaned_sound = AudioSegment(
            reduced_noise_data.tobytes(),
            frame_rate=sound.frame_rate,
            sample_width=2,
            channels=1
        )

        # 3. 輸出結果 (暫時關閉 Normalize 以凸顯降噪效果)
        # 為了測試，我們先直接輸出處理後的結果，不自動拉大音量
        cleaned_sound.export(output_path, format="mp3")
        return True

    except Exception as e:
        st.error(f"處理音訊時發生錯誤: {str(e)}")
        return False


# --- 網站介面邏輯 ---

st.title("🎵 影片/音訊 降噪與畫質增強器")
st.markdown("上傳您的影片或錄音檔，我們會自動提取音訊並去除背景雜音。")

# 檔案上傳區
uploaded_file = st.file_uploader("請選擇檔案 (支援 .mov, .mp4, .mp3, .wav)", type=["mov", "mp4", "mp3", "wav", "m4a"])

if uploaded_file is not None:
    # 建立臨時目錄來存放檔案，避免路徑問題
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, uploaded_file.name)
        extracted_audio_path = os.path.join(temp_dir, "extracted_raw.wav")
        final_output_path = os.path.join(temp_dir, "cleaned_output.mp3")

        # 將上傳的檔案寫入暫存區
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info(f"檔案 `{uploaded_file.name}` 上傳成功！準備處理...")

        # 判斷是否為影片
        file_extension = os.path.splitext(input_path)[1].lower()
        is_video = file_extension in ['.mov', '.mp4', '.avi', '.mkv']

        progress_bar = st.progress(0)
        status_text = st.empty()

        # 第一步：獲取音訊
        if is_video:
            status_text.text("正在從影片中提取音訊...")
            success = extract_audio_from_video(input_path, extracted_audio_path)
            processing_source = extracted_audio_path
        else:
            status_text.text("正在讀取音訊檔...")
            processing_source = input_path
            success = True

        progress_bar.progress(40)

        if success:
            # 第二步：降噪與增強
            status_text.text("正在進行 AI 降噪處理 (這可能需要一點時間，請耐心等候)...")
            enhancement_success = enhance_audio(processing_source, final_output_path)
            progress_bar.progress(90)

            if enhancement_success:
                progress_bar.progress(100)
                status_text.text("處理完成！")
                st.success("音訊優化成功！")

                # 顯示結果對比
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🎧 處理前 (原始)")
                    # 【修正點】讀取原始檔 bytes 放入記憶體，避免 Windows 檔案佔用鎖死
                    if is_video:
                        with open(input_path, "rb") as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                    else:
                        with open(input_path, "rb") as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes)

                with col2:
                    st.markdown("### 🎹 處理後 (降噪)")
                    # 讀取處理後的檔案
                    with open(final_output_path, "rb") as f:
                        processed_audio_bytes = f.read()
                    st.audio(processed_audio_bytes, format='audio/mp3')

                    # 下載按鈕
                    st.download_button(
                        label="📥 下載處理後的 MP3",
                        data=processed_audio_bytes,
                        file_name=f"enhanced_{os.path.splitext(uploaded_file.name)[0]}.mp3",
                        mime="audio/mp3"
                    )
            else:
                st.error("降噪處理失敗。")
        else:
            st.error("音訊提取失敗。")

st.markdown("---")
st.caption("由 Streamlit, FFmpeg 與 Noisereduce 強力驅動")