import queue
from collections import deque
from io import BytesIO

import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from gtts import gTTS

st.set_page_config(page_title="Silent Language Coach", page_icon="🤟", layout="wide")

# Free public STUN server -> no account, no API key needed for the video stream.
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

mp_face_mesh = mp.solutions.face_mesh
LIPS_IDX = sorted(set(sum(list(mp_face_mesh.FACEMESH_LIPS), ())))
UPPER_LIP, LOWER_LIP = 13, 14      # inner lip landmarks (openness)
LEFT_CORNER, RIGHT_CORNER = 61, 291  # mouth corners (width)

# Demo vocabulary only — replace _classify_segment with a real trained
# visual-speech-recognition model (e.g. an ONNX lip-reading network) for
# production accuracy. The video pipeline/UI below is built to plug one in.
DEMO_VOCAB = {
    "short_single_open": "YES",
    "short_double_open": "NO",
    "long_wide": "HELLO",
    "long_narrow": "HELP",
    "medium_double": "THANK YOU",
}


def mouth_aspect_ratio(landmarks, w, h):
    top = landmarks[UPPER_LIP]
    bottom = landmarks[LOWER_LIP]
    left = landmarks[LEFT_CORNER]
    right = landmarks[RIGHT_CORNER]
    top_pt = np.array([top.x * w, top.y * h])
    bottom_pt = np.array([bottom.x * w, bottom.y * h])
    left_pt = np.array([left.x * w, left.y * h])
    right_pt = np.array([right.x * w, right.y * h])
    vertical = np.linalg.norm(top_pt - bottom_pt)
    horizontal = np.linalg.norm(left_pt - right_pt)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


class LipReaderProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mar_history = deque(maxlen=45)  # ~1.5s at 30fps
        self.speaking = False
        self.silence_frames = 0
        self.result_queue: "queue.Queue[str]" = queue.Queue()
        self.threshold = 0.35

    def set_threshold(self, value: float):
        self.threshold = value

    def _classify_segment(self, segment):
        if len(segment) < 4:
            return None
        arr = np.array(segment)
        peak = arr.max()
        mean = arr.mean()
        crossings = int(np.sum((arr[:-1] < mean) & (arr[1:] >= mean)))
        wide = peak > 0.55
        if crossings <= 1 and not wide:
            return DEMO_VOCAB["short_single_open"]
        if crossings <= 1 and wide:
            return DEMO_VOCAB["long_wide"]
        if crossings == 2 and not wide:
            return DEMO_VOCAB["short_double_open"]
        if crossings == 2 and wide:
            return DEMO_VOCAB["medium_double"]
        return DEMO_VOCAB["long_narrow"]

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        mar = 0.0
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            mar = mouth_aspect_ratio(landmarks, w, h)
            for idx in LIPS_IDX:
                lm = landmarks[idx]
                cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

        self.mar_history.append(mar)

        if mar > self.threshold:
            self.speaking = True
            self.silence_frames = 0
        elif self.speaking:
            self.silence_frames += 1
            if self.silence_frames > 8:  # ~0.25s silence ends a segment
                phrase = self._classify_segment(list(self.mar_history))
                if phrase:
                    self.result_queue.put(phrase)
                self.speaking = False
                self.mar_history.clear()

        status = "SPEAKING" if self.speaking else "watching..."
        cv2.putText(
            img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 0) if self.speaking else (200, 200, 200), 2,
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def clean_with_context(raw_phrase, history):
    """Lightweight, fully offline clean-up. Swap for a local language
    model if you want smarter context correction later."""
    if not raw_phrase:
        return raw_phrase
    phrase = raw_phrase.strip().capitalize()
    if history and history[-1].lower() == phrase.lower():
        return None  # drop immediate repeats (likely re-triggered noise)
    return phrase


def speak(text: str) -> bytes:
    buf = BytesIO()
    gTTS(text=text, lang="en").write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def main():
    st.title("🤟 Silent Language Coach")
    st.caption("Real-time lip-reading assist for noisy rooms — camera in, cleaned text + whispered audio out.")

    with st.expander("⚠️ How this demo works (read first)", expanded=False):
        st.markdown(
            "- Runs with **no paid API keys** — video uses a free public STUN "
            "server, lip tracking is local (MediaPipe), and speech output uses "
            "keyless gTTS.\n"
            "- Turning arbitrary lip motion into arbitrary words is a hard, "
            "still-evolving research problem. This demo detects **mouth "
            "activity/shape** and matches it to a small demo vocabulary "
            "(YES / NO / HELLO / HELP / THANK YOU) as a proof of concept.\n"
            "- For real accuracy, replace `_classify_segment` with a trained "
            "visual-speech-recognition model — the camera pipeline and UI "
            "here are already wired to plug one in."
        )

    if "history" not in st.session_state:
        st.session_state.history = []

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Settings")
        threshold = st.slider("Mouth-open sensitivity", 0.15, 0.6, 0.35, 0.01)
        st.subheader("Detected phrases")
        history_box = st.empty()

    with col1:
        ctx = webrtc_streamer(
            key="lip-reader",
            video_processor_factory=LipReaderProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )
        display_box = st.empty()
        audio_box = st.empty()

    if ctx.video_processor:
        ctx.video_processor.set_threshold(threshold)
        while ctx.state.playing:
            try:
                raw = ctx.video_processor.result_queue.get(timeout=1)
            except queue.Empty:
                continue
            cleaned = clean_with_context(raw, st.session_state.history)
            if cleaned:
                st.session_state.history.append(cleaned)
                display_box.markdown(f"## 🗣️ {cleaned}")
                try:
                    audio_box.audio(speak(cleaned), format="audio/mp3")
                except Exception:
                    st.warning("Text-to-speech unavailable right now; showing text only.")
            history_box.write(list(reversed(st.session_state.history[-10:])))


if __name__ == "__main__":
    main()
