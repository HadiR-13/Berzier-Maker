import streamlit as st
import cv2
import numpy as np
import potrace
import tempfile
import os
import atexit
import shutil

BILATERAL_FILTER = True
USE_L2_GRADIENT = True


def get_contours(image, nudge=0.33):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if BILATERAL_FILTER:
        median = np.clip(np.median(gray), 10, 245)
        lower = int(max(0, (1 - nudge) * median))
        upper = int(min(255, (1 + nudge) * median))
        filtered = cv2.bilateralFilter(gray, 5, 50, 50)
        edged = cv2.Canny(filtered, lower, upper, L2gradient=USE_L2_GRADIENT)
    else:
        edged = cv2.Canny(gray, 30, 200)

    return np.flipud(edged)


def get_trace(data):
    data = np.where(data > 1, 1, 0).astype(np.uint8)
    bmp = potrace.Bitmap(data)
    
    return bmp.trace(
        turdsize=2, 
        turnpolicy=potrace.TURNPOLICY_MINORITY, 
        alphamax=2.0,
        opticurve=1, 
        opttolerance=1.0
    )


def get_latex(image):
    latex = []
    path = get_trace(get_contours(image))

    for curve in path.curves:
        x0, y0 = curve.start_point
        for seg in curve.segments:
            if seg.is_corner:
                x1, y1 = seg.c
                x2, y2 = seg.end_point
                latex.append(f'((1-t){x0}+t{x1},(1-t){y0}+t{y1})')
                latex.append(f'((1-t){x1}+t{x2},(1-t){y1}+t{y2})')
            else:
                x1, y1 = seg.c1
                x2, y2 = seg.c2
                x3, y3 = seg.end_point
                latex.append(
                    f'((1-t)*((1-t)*((1-t){x0}+t{x1})+t((1-t){x1}+t{x2}))'
                    f'+t((1-t)*((1-t){x1}+t{x2})+t((1-t){x2}+t{x3})),' 
                    f'(1-t)*((1-t)*((1-t){y0}+t{y1})+t((1-t){y1}+t{y2}))'
                    f'+t((1-t)*((1-t){y1}+t{y2})+t((1-t){y2}+t{y3})))'
                )
            x0, y0 = seg.end_point
    return latex


st.set_page_config(
    page_title="Desmos Bezier Renderer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

TEMP_DIR = tempfile.mkdtemp(prefix="desmos_bezier_")

def cleanup_tempdir():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)

atexit.register(cleanup_tempdir)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1000px;
        }
        h1 {
            font-size: 1.8rem !important;
            font-weight: 600 !important;
            margin-bottom: 0.5rem;
        }
        .stCaption {
            color: #777;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Desmos Bezier Renderer")
st.caption("Convert an image into smooth Bezier curve LaTeX expressions for Desmos.")

uploaded = st.file_uploader(
    label="Upload Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

if not uploaded:
    st.info("Upload a JPG or PNG image to begin.")
    st.stop()

for f in os.listdir(TEMP_DIR):
    os.remove(os.path.join(TEMP_DIR, f))

temp_path = os.path.join(TEMP_DIR, uploaded.name)
with open(temp_path, "wb") as f:
    f.write(uploaded.read())

image = cv2.imread(temp_path)

col1, col2 = st.columns([3, 1])
with col1:
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
with col2:
    st.markdown(f"**Filename:** `{uploaded.name}`")
    st.markdown(f"**Resolution:** {image.shape[1]} × {image.shape[0]}")

with st.spinner("Generating LaTeX expressions..."):
    latex_expressions = get_latex(image)

st.markdown("#### LaTeX Output")
latex_text = "\n".join(latex_expressions)
st.text_area(
    label="LaTeX Output",
    value=latex_text,
    height=400,
    label_visibility="collapsed"
)

st.download_button(
    label="Download LaTeX File",
    data=latex_text,
    file_name="desmos_bezier_output.tex",
    mime="text/plain",
    width="stretch"
)