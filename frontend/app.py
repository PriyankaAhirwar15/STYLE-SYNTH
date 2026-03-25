import streamlit as st
import requests
import base64
from PIL import Image
import io
import time
from config import API_HOST, API_PORT

# Page config
st.set_page_config(
    page_title="🎭 STYLE-SYNTH",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #FF6B6B;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .sub-header {
        text-align: center;
        color: #4ECDC4;
        font-size: 1.5rem;
        margin-bottom: 2rem;
    }
    .quality-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-score { background-color: #D4EDDA; color: #155724; }
    .medium-score { background-color: #FFF3CD; color: #856404; }
    .low-score { background-color: #F8D7DA; color: #721C24; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎭 STYLE-SYNTH</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Face Style Transfer GAN with LLM Quality Checker</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎨 Style Options")
    styles = {
        "anime": "🎌 Anime Style",
        "sketch": "✏️ Pencil Sketch",
        "oil_painting": "🎨 Oil Painting",
        "watercolor": "💧 Watercolor",
        "cartoon": "🦸 Cartoon Style"
    }

    selected_style = st.selectbox(
        "Choose your style:",
        list(styles.keys()),
        format_func=lambda x: styles[x]
    )

    st.header("⚙️ Settings")
    check_quality = st.checkbox("Enable LLM Quality Check", value=True)

    st.header("📊 About")
    st.markdown("""
    **STYLE-SYNTH** transforms your face photos into artistic styles using advanced GAN technology and provides AI-powered quality assessment.

    **Features:**
    - 🎨 Multiple artistic styles
    - 🤖 LLM quality evaluation
    - ⚡ Real-time processing
    - 📱 FastAPI backend
    """)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Your Photo")
    uploaded_file = st.file_uploader(
        "Choose a face photo...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear face photo for best results"
    )

    if uploaded_file:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        # Transform button
        if st.button("🎨 Transform Image", type="primary", use_container_width=True):
            with st.spinner("🎭 Applying style transfer..."):
                try:
                    # Prepare file for API
                    files = {"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
                    data = {"style": selected_style, "check_quality": str(check_quality).lower()}

                    # API call
                    api_url = f"http://127.0.0.1:8002/transform"
                    response = requests.post(api_url, files=files, data=data, timeout=30)

                    if response.status_code == 200:
                        result = response.json()

                        # Store result in session state
                        st.session_state.transform_result = result
                        st.success("✨ Style transfer completed!")
                        st.rerun()
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        st.error(response.text)

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Connection Error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {str(e)}")

with col2:
    st.subheader("🎨 Transformed Result")

    if 'transform_result' in st.session_state:
        result = st.session_state.transform_result

        if result.get("success"):
            # Display styled image
            styled_b64 = result["styled_image"]
            styled_img = Image.open(io.BytesIO(base64.b64decode(styled_b64)))
            st.image(styled_img, caption=f"Style: {selected_style.title()}", use_column_width=True)

            # Processing info
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("⚡ Processing Time", f"{result['processing_time']:.2f}s")
            with col_b:
                st.metric("🎨 Style", selected_style.title())
            with col_c:
                if check_quality and "quality_report" in result:
                    quality_score = result["quality_report"].get("quality_score", 0)
                    st.metric("⭐ Quality Score", f"{quality_score}/10")

            # Quality Report
            if check_quality and "quality_report" in result:
                quality_report = result["quality_report"]

                # Quality score with color coding
                score = quality_report.get("quality_score", 0)
                if score >= 8:
                    score_class = "high-score"
                elif score >= 6:
                    score_class = "medium-score"
                else:
                    score_class = "low-score"

                st.markdown(f'<div class="quality-score {score_class}">⭐ Quality Score: {score}/10</div>', unsafe_allow_html=True)

                # Assessment
                st.subheader("🤖 AI Assessment")
                st.write(quality_report.get("assessment", "No assessment available"))

                # Style Accuracy
                accuracy = quality_report.get("style_accuracy", "unknown")
                if accuracy == "high":
                    st.success(f"🎯 Style Accuracy: {accuracy.title()}")
                elif accuracy == "medium":
                    st.warning(f"🎯 Style Accuracy: {accuracy.title()}")
                else:
                    st.error(f"🎯 Style Accuracy: {accuracy.title()}")

                # Strengths and Improvements
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("✅ Strengths")
                    strengths = quality_report.get("strengths", [])
                    for strength in strengths:
                        st.write(f"• {strength}")

                with col2:
                    st.subheader("🔧 Improvements")
                    improvements = quality_report.get("improvements", [])
                    for improvement in improvements:
                        st.write(f"• {improvement}")

                # Recommendation
                st.subheader("💡 Recommendation")
                st.info(quality_report.get("recommendation", "No recommendation available"))

            # Style Metrics
            if "style_metrics" in result:
                with st.expander("📊 Technical Metrics"):
                    metrics = result["style_metrics"]
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Brightness Change", f"{metrics.get('brightness_change', 0):.1f}%")
                        st.metric("Contrast Score", f"{metrics.get('contrast_score', 0):.2f}")

                    with col2:
                        st.metric("Edge Density", f"{metrics.get('edge_density', 0):.4f}")
                        st.metric("Color Variance", f"{metrics.get('color_variance', 0):.1f}")

        else:
            st.error("❌ Transformation failed!")
            if "error" in result:
                st.error(f"Error: {result['error']}")

    else:
        st.info("👆 Upload an image and click 'Transform Image' to see the result!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Priyanka Ahirwar")
st.markdown("[🎭 STYLE-SYNTH](https://github.com/your-repo/style-synth) | Made for AI Art Generation")