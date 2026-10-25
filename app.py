"""
YOLOv8n Surveillance Detection GUI
Upload a video → live annotated preview → download result
"""

import os
import cv2
import gradio as gr
import tempfile
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ─── Model setup ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "best.pt"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()
if not MODEL_PATH.is_absolute():
    MODEL_PATH = (BASE_DIR / MODEL_PATH).resolve()


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{model_path}'. Set MODEL_PATH to a valid YOLO weights file."
        )
    return YOLO(str(model_path))


try:
    model = load_model(MODEL_PATH)
    MODEL_ERROR = None
except Exception as exc:
    model = None
    MODEL_ERROR = str(exc)

CLASS_NAMES = model.names if model is not None else {0: 'threew', 1: 'car', 2: 'h_vehicle', 3: 'person', 4: 'two', 5: 'unclassified'}
CLASS_COLORS = {
    0: (255, 165,   0),   # threew      – orange
    1: ( 50, 205,  50),   # car         – lime green
    2: (  0, 120, 255),   # h_vehicle   – blue
    3: (255,  50,  50),   # person      – red
    4: (180,   0, 255),   # two         – purple
    5: (180, 180, 180),   # unclassified– grey
}

DEFAULT_CONF = 0.35

# ─── Inference helpers ────────────────────────────────────────────────────────

def draw_boxes(frame: np.ndarray, results) -> np.ndarray:
    for box in results[0].boxes:
        cls_id          = int(box.cls[0])
        conf            = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color           = CLASS_COLORS.get(cls_id, (200, 200, 200))
        label           = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def run_inference(video_path: str, conf: float):
    """
    Generator → yields (rgb_frame, out_path_or_None).
    Last item has out_path set.
    """
    if model is None:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False,
                                           dir=tempfile.gettempdir())
    out_path = out_file.name
    out_file.close()

    writer    = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    last_rgb  = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results   = model.predict(frame, conf=conf, verbose=False)
            annotated = draw_boxes(frame.copy(), results)
            writer.write(annotated)
            frame_idx += 1

            if frame_idx % 3 == 0 or frame_idx == 1:
                rgb      = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                last_rgb = rgb
                yield rgb, None
    except Exception:
        yield None, None
        return
    finally:
        cap.release()
        if writer.isOpened():
            writer.release()

    if last_rgb is not None:
        yield last_rgb, out_path


# ─── Gradio event handler ─────────────────────────────────────────────────────

def handle_run(video_path, conf):
    """Streaming generator bound to run_btn.click."""
    if model is None:
        yield None, gr.update(visible=False), f"⚠ {MODEL_ERROR}"
        return

    if video_path is None:
        yield None, gr.update(visible=False), "⚠ Upload a video first."
        return

    yield None, gr.update(visible=False), "⏳ Loading video..."

    last_rgb = None
    out_path = None

    for rgb, path in run_inference(video_path, conf):
        if rgb is None and path is None:
            yield None, gr.update(visible=False), "⚠ Inference failed. Check the uploaded video or model configuration."
            return
        last_rgb = rgb
        out_path = path
        if path is None:
            yield rgb, gr.update(visible=False), "🔄 Detecting..."

    if out_path and last_rgb is not None:
        yield last_rgb, gr.update(value=out_path, visible=True), "✅ Done! Download ready."
    else:
        yield last_rgb, gr.update(visible=False), "⚠ Could not process video."


# ─── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html { height: 100% !important; overflow: hidden !important; }

body {
    height: 100vh !important;
    overflow: hidden !important;
    background: #0d0f14 !important;
    margin: 0 !important; padding: 0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Gradio 6 shell */
gradio-app {
    height: 100vh !important;
    overflow: hidden !important;
    background: #0d0f14 !important;
}
gradio-app > .gradio-container {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    padding: 0 !important;
    margin: 0 !important;
    background: #0d0f14 !important;
}
gradio-app > .gradio-container > .main {
    height: 100vh !important;
    overflow: hidden !important;
    padding: 0 !important;
}
gradio-app > .gradio-container > .main > .wrap {
    height: 100vh !important;
    overflow: hidden !important;
    padding: 0 !important;
    gap: 0 !important;
}

footer { display: none !important; }

/* ─── Top bar ──────────────────────────────────────────── */
#topbar {
    height: 46px;
    background: #0f1117;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 22px;
}
.wdot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.wr { background: #ff5f56; }
.wy { background: #ffbd2e; }
.wg { background: #27c93f; }
#tb-title {
    font-size: 12px; font-weight: 700; letter-spacing: .10em;
    text-transform: uppercase; color: #6b7280; margin-left: 6px;
}
#tb-badge {
    margin-left: auto; font-size: 11px; font-weight: 600;
    background: rgba(99,179,237,.10); color: #63b3ed;
    border: 1px solid rgba(99,179,237,.22);
    border-radius: 20px; padding: 3px 12px; letter-spacing: .04em;
}

/* ─── Layout ───────────────────────────────────────────── */
#layout {
    display: grid !important;
    grid-template-columns: 285px 1fr !important;
    height: calc(100vh - 46px) !important;
    overflow: hidden !important;
    gap: 0 !important;
}

/* ─── Left panel ───────────────────────────────────────── */
#left {
    background: #111318 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    padding: 16px 14px !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 10px !important;
}
#left > * { flex-shrink: 0 !important; }

.slabel {
    font-size: 10px; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: #3e4460; margin: 0;
}

/* Upload */
#vu { flex-shrink: 0 !important; }
#vu .wrap {
    background: #181c26 !important;
    border: 2px dashed rgba(99,179,237,.30) !important;
    border-radius: 10px !important;
    min-height: 90px !important;
    transition: border-color .2s, background .2s !important;
}
#vu .wrap:hover {
    border-color: rgba(99,179,237,.65) !important;
    background: rgba(99,179,237,.04) !important;
}

/* Slider label hidden (shows value inline) */
#cslider > .wrap > .block { display: none !important; }
#cslider input[type=range] { accent-color: #63b3ed !important; width: 100% !important; }

/* Run button */
#runbtn {
    background: linear-gradient(135deg, #2d5eff, #4fa8f5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 10px !important;
    box-shadow: 0 4px 18px rgba(45,94,255,.35) !important;
    transition: all .2s !important;
    width: 100% !important;
    cursor: pointer !important;
}
#runbtn:hover {
    box-shadow: 0 6px 26px rgba(45,94,255,.55) !important;
    transform: translateY(-1px) !important;
}

/* Status */
#statusmd p {
    font-size: 12px !important; color: #555a6e !important;
    text-align: center !important; margin: 0 !important;
}

/* Download file */
#dlfile .wrap {
    background: rgba(39,201,63,.06) !important;
    border: 1px solid rgba(39,201,63,.25) !important;
    border-radius: 9px !important;
    padding: 8px 10px !important;
}
#dlfile .wrap a { color: #27c93f !important; font-size: 13px !important; font-weight: 600 !important; }
#dlfile .wrap .thumbnail-sm { display: none !important; }

/* Separator */
.hsep { height: 1px; background: rgba(255,255,255,0.05); flex-shrink: 0; }

/* Legend */
#legend { display: flex; flex-direction: column; gap: 6px; }
.lg { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #7a8099; }
.ld { width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0; }

/* ─── Right panel ───────────────────────────────────────── */
#right {
    background: #080a0e !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    position: relative !important;
}

/* Image component */
#previmg {
    width: 100% !important;
    height: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
#previmg img {
    max-width: 100% !important;
    max-height: calc(100vh - 46px) !important;
    object-fit: contain !important;
    display: block !important;
}
/* Hide image container decoration */
#previmg > .wrap { border: none !important; background: transparent !important; box-shadow: none !important; }

/* Placeholder */
#empty {
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    color: #1b1f2d; font-size: 15px; font-weight: 500;
    pointer-events: none; white-space: nowrap; user-select: none;
}
"""

TOPBAR_HTML = """
<div id="topbar">
  <span class="wdot wr"></span>
  <span class="wdot wy"></span>
  <span class="wdot wg"></span>
  <span id="tb-title">Detection Studio</span>
  <span id="tb-badge">YOLOv8n &nbsp;·&nbsp; 6 classes</span>
</div>
"""

LEGEND_HTML = """
<div id="legend">
  <div class="lg"><div class="ld" style="background:#ffa500"></div>Three-wheeler</div>
  <div class="lg"><div class="ld" style="background:#32cd32"></div>Car</div>
  <div class="lg"><div class="ld" style="background:#0078ff"></div>Heavy Vehicle</div>
  <div class="lg"><div class="ld" style="background:#ff3232"></div>Person</div>
  <div class="lg"><div class="ld" style="background:#b400ff"></div>Two-wheeler</div>
  <div class="lg"><div class="ld" style="background:#b4b4b4"></div>Unclassified</div>
</div>
"""

EMPTY_HTML = '<div id="empty">Upload a video and click ▶ Run Detection</div>'

# ─── Build UI ─────────────────────────────────────────────────────────────────

with gr.Blocks() as demo:

    gr.HTML(TOPBAR_HTML)

    with gr.Row(elem_id="layout"):

        # ── Left ────────────────────────────────────────────────────────────
        with gr.Column(elem_id="left", min_width=285, scale=0):

            gr.HTML('<p class="slabel">Input Video</p>')
            video_in = gr.Video(
                label="", elem_id="vu",
                sources=["upload"], height=100,
                show_label=False,
            )

            gr.HTML('<p class="slabel" style="margin-top:2px">Confidence Threshold</p>')
            conf_sl = gr.Slider(
                minimum=0.10, maximum=0.95, value=DEFAULT_CONF,
                step=0.01, label="Confidence", elem_id="cslider",
            )

            run_btn = gr.Button("▶  Run Detection", elem_id="runbtn", variant="primary")

            status_md = gr.Markdown("Upload a video to begin.", elem_id="statusmd")

            dl_file = gr.File(
                label="📥  Annotated Video",
                elem_id="dlfile",
                visible=False,
                interactive=False,
            )

            gr.HTML('<div class="hsep"></div>')
            gr.HTML('<p class="slabel">Class Legend</p>')
            gr.HTML(LEGEND_HTML)

        # ── Right ────────────────────────────────────────────────────────────
        with gr.Column(elem_id="right", scale=1):
            gr.HTML(EMPTY_HTML)
            preview = gr.Image(
                label="", elem_id="previmg",
                show_label=False,
                buttons=[],          # no buttons (Gradio 6 API)
                interactive=False,
                type="numpy",
            )

    # ── Events ───────────────────────────────────────────────────────────────
    run_btn.click(
        fn=handle_run,
        inputs=[video_in, conf_sl],
        outputs=[preview, dl_file, status_md],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CSS,
        theme=gr.themes.Base(),
        show_error=True,
        quiet=False,
    )
