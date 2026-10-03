import streamlit as st
from google import genai
from PIL import Image, ImageDraw, ImageFont
import os
import json
import re
import io

st.set_page_config(
    page_title="SpaceSnap AI",
    page_icon="🌎",
    layout="wide"
)

st.title("🌎 SpaceSnap AI")
st.write(
    "Upload a NASA or Earth-observation image to explore "
    "visible features with AI-generated explanations."
)

st.info(
    "AI-generated feature locations are approximate and "
    "should not be treated as scientifically verified detections."
)

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error(
        "Gemini API key not found. Set GEMINI_API_KEY "
        "in the environment before starting the app."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "webp"]
)

def analyze_image(image):
    client = genai.Client(api_key=api_key)

    prompt = """
    Analyze this Earth or space image for SpaceSnap AI.

    Return valid JSON only with keys:
    image_type, location, features, explanation, limitations.

    Include 3 to 5 visible features. Each feature must contain:
    name, description, bbox.

    bbox must be [x_min, y_min, x_max, y_max] normalized
    from 0 to 1000, measured from the top-left corner.

    Describe only visible features. Do not treat printed map
    text as a physical feature. Coordinates are estimates.
    Mention uncertainty and do not invent measurements.
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[prompt, image]
    )

    text = response.text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return json.loads(text)

def annotate_image(image, features):
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    width, height = result.size

    colors = [
        (255, 70, 70),
        (0, 220, 200),
        (255, 220, 0),
        (120, 180, 255),
        (255, 150, 220)
    ]

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    for i, feature in enumerate(features[:5]):
        bbox = feature.get("bbox", [])

        if len(bbox) != 4:
            continue

        try:
            coords = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue

        if not all(0 <= v <= 1000 for v in coords):
            continue

        x1, y1, x2, y2 = coords

        if x2 <= x1 or y2 <= y1:
            continue

        x1, x2 = int(x1 * width / 1000), int(x2 * width / 1000)
        y1, y2 = int(y1 * height / 1000), int(y2 * height / 1000)

        color = colors[i % len(colors)]
        label = str(feature.get("name", "Feature"))[:40]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        box = draw.textbbox((0, 0), label, font=font)
        label_w = box[2] - box[0]
        label_h = box[3] - box[1]
        label_y = max(0, y1 - label_h - 8)

        draw.rectangle(
            [x1, label_y, x1 + label_w + 8, label_y + label_h + 6],
            fill=color
        )
        draw.text(
            (x1 + 4, label_y + 2),
            label,
            fill="black",
            font=font
        )

    return result

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)

        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Gemini is analyzing your image..."):
                analysis = analyze_image(image)
                annotated = annotate_image(
                    image, analysis.get("features", [])
                )

            st.session_state["analysis"] = analysis
            output = io.BytesIO()
            annotated.save(output, format="JPEG")
            st.session_state["annotated_bytes"] = output.getvalue()

        if "analysis" in st.session_state:
            analysis = st.session_state["analysis"]

            st.subheader("🛰️ Image Analysis")
            st.write("**Image type:**", analysis.get("image_type", "Unknown"))
            st.write("**Location:**", analysis.get("location", "Unknown"))

            left, right = st.columns(2)

            with left:
                st.subheader("Annotated image")
                st.image(
                    st.session_state["annotated_bytes"],
                    use_container_width=True
                )
                st.download_button(
                    "Download annotated image",
                    data=st.session_state["annotated_bytes"],
                    file_name="spacesnap_annotated.jpg",
                    mime="image/jpeg"
                )

            with right:
                st.subheader("Detected features")
                for feature in analysis.get("features", []):
                    st.markdown("**" + str(feature.get("name", "Feature")) + "**")
                    st.write(feature.get("description", ""))
                    st.caption("Approximate box: " + str(feature.get("bbox", [])))

            st.subheader("🌍 Simple explanation")
            st.write(analysis.get("explanation", ""))

            st.subheader("⚠️ Limitations")
            st.write(analysis.get("limitations", ""))

    except Exception as error:
        st.error("Analysis failed: " + str(error))
        st.info("Check your internet connection, API access, and model availability.")
