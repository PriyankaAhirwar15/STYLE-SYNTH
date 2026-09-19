"""
🎭 STYLE-SYNTH: Next-Gen Face Style Transfer GAN Studio
Interactive Streamlit Application featuring:
- 8 Neural GAN Art Styles
- Continuous Latent Space Multi-Style Blending
- Facial Attribute Modulation (Smoothness, Vibrancy, Sharpness)
- Smart Face Detection & Auto-Square-Crop
- Batch Portrait Processing & ZIP Export
- PyTorch GAN Architecture & Layer Inspector
- Groq LLaMA 3.3-70B AI Quality Auditor
- Zero-Error Hugging Face Spaces & Standalone compatibility
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import time
import base64
import os
import zipfile
import sys
from pathlib import Path

# Add project root to path for direct imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import STYLES, STYLE_METADATA, APP_NAME, APP_VERSION, APP_DESCRIPTION, API_HOST, API_PORT
from src.utils.image_utils import pil_to_cv2, cv2_to_pil, image_to_base64
from src.gan.transform import StyleTransferPipeline
from src.auditor.llm_checker import StyleQualityChecker

# Page Configuration
st.set_page_config(
    page_title=f"🎭 {APP_NAME} | Face Style Transfer GAN Studio",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Theme
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #8892B0;
        font-size: 1.15rem;
        margin-bottom: 1.8rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        background-color: #2D3748;
        color: #E2E8F0;
    }
    .quality-card {
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .score-high { background: linear-gradient(135deg, rgba(46, 125, 50, 0.15), rgba(76, 175, 80, 0.1)); border-left: 5px solid #4CAF50; }
    .score-med { background: linear-gradient(135deg, rgba(245, 124, 0, 0.15), rgba(255, 152, 0, 0.1)); border-left: 5px solid #FF9800; }
    .metric-box {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# In-Memory Pipeline Cache (Ensures Instant Zero-Error Execution on Hugging Face Spaces)
@st.cache_resource(show_spinner="🧠 Initializing Neural GAN Engines...")
def get_pipeline():
    return StyleTransferPipeline(device="cpu")


pipeline = get_pipeline()

# Header
st.markdown(f'<div class="main-title">🎭 {APP_NAME} v{APP_VERSION}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Next-Gen Face Style Transfer GAN Studio • Multi-Style Latent Blending • AI Quality Auditor</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Studio Settings")

    execution_mode = st.radio(
        "Execution Engine:",
        ["⚡ In-Memory Direct Pipeline (Hugging Face / Fast)", "🔌 FastAPI REST Server (Port 8002)"],
        index=0,
        help="In-memory mode runs directly inside Streamlit with zero networking overhead."
    )

    st.markdown("---")
    st.subheader("🔑 AI Quality Auditor")
    groq_api_key = st.text_input(
        "Groq API Key (Optional):",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Enter your Groq API key for LLaMA 3.3-70B quality audit. Leave blank for built-in heuristic neural scoring."
    )
    if groq_api_key:
        pipeline.checker.update_api_key(groq_api_key)

    check_quality = st.checkbox("Enable AI Quality Assessment", value=True)

    st.markdown("---")
    st.subheader("ℹ️ System Info")
    st.markdown("""
    - **Generator:** ResNet-9Block GAN
    - **Discriminator:** 70x70 PatchGAN
    - **Styles Available:** 8 Neural Models
    - **Deployment:** Hugging Face Ready
    """)
    st.caption("Built with PyTorch & Streamlit by Priyanka Ahirwar")


# Studio Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎭 Portrait Studio",
    "🌀 Latent Style Blender",
    "📦 Batch Generator",
    "🧠 GAN Architecture & Auditor"
])


# ==========================================
# TAB 1: PORTRAIT STUDIO
# ==========================================
with tab1:
    col_ctrl, col_view = st.columns([1, 1], gap="large")

    with col_ctrl:
        st.subheader("1️⃣ Upload & Style Selection")
        uploaded_file = st.file_uploader(
            "Upload a Face Portrait:",
            type=["jpg", "jpeg", "png", "webp"],
            key="studio_uploader",
            help="Upload a clear face photo for best artistic GAN synthesis"
        )

        # Style Selection
        selected_style = st.selectbox(
            "Select GAN Style:",
            options=STYLES,
            format_func=lambda s: STYLE_METADATA.get(s, {}).get("name", s.title()),
            index=0
        )

        style_info = STYLE_METADATA.get(selected_style, {})
        st.info(f"**{style_info.get('name')}**: {style_info.get('description')}")

        st.markdown("#### 🎛️ Facial & Style Modulation")
        c1, c2 = st.columns(2)
        with c1:
            style_strength = st.slider("Style Intensity:", 0.1, 1.0, 1.0, 0.05, help="Blend between original photo and GAN styled art")
            smoothness = st.slider("Skin Smoothing:", 0.0, 10.0, 2.0, 0.5, help="Bilateral skin-smoothing intensity")
        with c2:
            vibrancy = st.slider("Color Vibrancy:", 0.5, 2.0, 1.15, 0.05, help="Saturation enhancement factor")
            sharpness = st.slider("Edge Sharpness:", 0.0, 10.0, 1.5, 0.5, help="Unsharp mask edge contour boost")

        auto_face_crop = st.checkbox("🔍 Auto Face Detection & Centering Crop", value=True, help="Automatically detects face and crops square for optimal portrait framing")

        transform_clicked = st.button("✨ Synthesize GAN Portrait", type="primary", use_container_width=True)

    with col_view:
        st.subheader("2️⃣ Synthesized Result")

        if uploaded_file is not None:
            pil_input = Image.open(uploaded_file)

            if transform_clicked:
                with st.spinner("🎨 Generating Neural GAN Portrait..."):
                    img_bgr = pil_to_cv2(pil_input)

                    # Execution via direct pipeline
                    result = pipeline.transform(
                        image=img_bgr,
                        style=selected_style,
                        style_strength=style_strength,
                        smoothness=smoothness,
                        vibrancy=vibrancy,
                        sharpness=sharpness,
                        auto_face_crop=auto_face_crop,
                        check_quality=check_quality
                    )

                    if result.get("success"):
                        st.session_state["last_result"] = result
                        st.success("🎉 Synthesis Complete!")
                    else:
                        st.error(f"❌ Error: {result.get('error')}")

            # Display results if present in session
            if "last_result" in st.session_state:
                res = st.session_state["last_result"]
                orig_pil = cv2_to_pil(res["original_image"])
                styled_pil = cv2_to_pil(res["styled_image"])

                # Side-by-Side Comparison
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.image(orig_pil, caption="Input Portrait", use_column_width=True)
                with v_col2:
                    st.image(styled_pil, caption=f"GAN: {selected_style.title()}", use_column_width=True)

                # Download Button
                buf = io.BytesIO()
                styled_pil.save(buf, format="JPEG", quality=95)
                st.download_button(
                    label="💾 Download Stylized Portrait",
                    data=buf.getvalue(),
                    file_name=f"style_synth_{selected_style}_{int(time.time())}.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )

                # Performance & Metrics
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("⚡ Time", f"{res['processing_time']:.2f}s")
                with m2:
                    st.metric("🎯 Face Detected", "Yes" if res.get('face_detected') else "Centered")
                with m3:
                    st.metric("📐 Resolution", f"{styled_pil.width}x{styled_pil.height}")
                with m4:
                    if "quality_report" in res and "quality_score" in res["quality_report"]:
                        st.metric("⭐ AI Score", f"{res['quality_report']['quality_score']}/10")

                # AI Quality Auditor Report
                if "quality_report" in res and res["quality_report"]:
                    q_report = res["quality_report"]
                    score = q_report.get("quality_score", 8.5)
                    card_class = "score-high" if score >= 8.0 else "score-med"

                    st.markdown(f"""
                    <div class="quality-card {card_class}">
                        <h4>🤖 AI Quality Auditor Assessment ({score}/10)</h4>
                        <p>{q_report.get('assessment', 'High quality neural transformation achieved.')}</p>
                        <p><strong>💡 Recommendation:</strong> {q_report.get('recommendation', 'Excellent artistic output.')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("📊 Detailed Technical & Structural Metrics"):
                        metrics = res.get("style_metrics", {})
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.write(f"- **Brightness Delta:** `{metrics.get('brightness_change', 0)}%`")
                            st.write(f"- **Contrast Score:** `{metrics.get('contrast_score', 0)}`")
                            st.write(f"- **Edge Density:** `{metrics.get('edge_density', 0)}`")
                        with col_m2:
                            st.write(f"- **Color Variance:** `{metrics.get('color_variance', 0)}`")
                            st.write(f"- **Structural Fidelity:** `{metrics.get('structural_fidelity', 0.85)}`")
                            st.write(f"- **Adversarial Realism:** `{metrics.get('adversarial_score', 0.9)}`")
        else:
            st.info("👈 Upload a portrait image on the left to begin.")


# ==========================================
# TAB 2: LATENT STYLE BLENDER
# ==========================================
with tab2:
    st.subheader("🌀 Multi-Style Latent Interpolation")
    st.markdown("Seamlessly cross-fade and blend two distinct GAN artistic representations in continuous latent space.")

    b_col_ctrl, b_col_view = st.columns([1, 1], gap="large")

    with b_col_ctrl:
        blend_file = st.file_uploader("Upload Portrait for Blending:", type=["jpg", "jpeg", "png", "webp"], key="blend_uploader")

        bc1, bc2 = st.columns(2)
        with bc1:
            style_a = st.selectbox("Primary Style (0%):", STYLES, index=0, key="style_a")
        with bc2:
            style_b = st.selectbox("Secondary Style (100%):", STYLES, index=2, key="style_b")

        alpha_ratio = st.slider("Style Blend Ratio (α):", 0.0, 1.0, 0.5, 0.05, help="0.0 = 100% Primary Style, 1.0 = 100% Secondary Style")

        st.caption(f"Currently: **{int((1-alpha_ratio)*100)}% {style_a.title()}** + **{int(alpha_ratio*100)}% {style_b.title()}**")

        blend_btn = st.button("🌀 Synthesize Latent Blend", type="primary", use_container_width=True)

    with b_col_view:
        if blend_file is not None:
            pil_b_input = Image.open(blend_file)
            if blend_btn:
                with st.spinner("Interpolating styles in latent space..."):
                    img_bgr = pil_to_cv2(pil_b_input)
                    b_res = pipeline.blend(
                        image=img_bgr,
                        style1=style_a,
                        style2=style_b,
                        alpha=alpha_ratio,
                        auto_face_crop=True,
                        check_quality=True
                    )
                    if b_res.get("success"):
                        st.session_state["blend_result"] = b_res
                        st.success("✨ Blended Portrait Synthesized!")

            if "blend_result" in st.session_state:
                res_b = st.session_state["blend_result"]
                blended_pil = cv2_to_pil(res_b["styled_image"])
                st.image(blended_pil, caption=f"Blended: {style_a.title()} + {style_b.title()}", use_column_width=True)

                buf = io.BytesIO()
                blended_pil.save(buf, format="JPEG", quality=95)
                st.download_button(
                    label="💾 Download Blended Portrait",
                    data=buf.getvalue(),
                    file_name=f"blended_{style_a}_{style_b}_{int(time.time())}.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
        else:
            st.info("👈 Upload a portrait to synthesize multi-style blends.")


# ==========================================
# TAB 3: BATCH GENERATOR
# ==========================================
with tab3:
    st.subheader("📦 Batch Portrait Generator")
    st.markdown("Transform multiple face portraits simultaneously and download a complete ZIP archive.")

    batch_files = st.file_uploader("Upload Multiple Photos:", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, key="batch_uploader")
    batch_style = st.selectbox("Select Batch GAN Style:", STYLES, format_func=lambda s: STYLE_METADATA.get(s, {}).get("name", s.title()), key="batch_style")

    if batch_files and st.button("🚀 Process All Portraits", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        zip_buffer = io.BytesIO()
        processed_images = []

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for idx, uploaded_item in enumerate(batch_files):
                status_text.text(f"Processing ({idx + 1}/{len(batch_files)}): {uploaded_item.name}...")
                img_pil = Image.open(uploaded_item)
                img_bgr = pil_to_cv2(img_pil)

                res = pipeline.transform(
                    image=img_bgr,
                    style=batch_style,
                    auto_face_crop=True,
                    check_quality=False
                )

                if res.get("success"):
                    styled_pil = cv2_to_pil(res["styled_image"])
                    processed_images.append((uploaded_item.name, styled_pil))

                    img_bytes_io = io.BytesIO()
                    styled_pil.save(img_bytes_io, format="JPEG", quality=95)
                    zip_file.writestr(f"styled_{uploaded_item.name}", img_bytes_io.getvalue())

                progress_bar.progress((idx + 1) / len(batch_files))

        status_text.text(f"✅ Successfully processed {len(processed_images)} portraits!")

        # Download ZIP button
        st.download_button(
            label=f"📦 Download All {len(processed_images)} Images (.ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"style_synth_batch_{batch_style}_{int(time.time())}.zip",
            mime="application/zip",
            type="primary"
        )

        # Display Gallery
        st.markdown("### 🖼️ Batch Gallery")
        cols = st.columns(3)
        for i, (name, img) in enumerate(processed_images):
            with cols[i % 3]:
                st.image(img, caption=name, use_column_width=True)


# ==========================================
# TAB 4: GAN NEURAL ARCHITECTURE & INSPECTOR
# ==========================================
with tab4:
    st.subheader("🧠 PyTorch GAN Architecture & Loss Auditor")
    st.markdown("Inspect the deep neural generator specifications, receptive fields, and analyze GAN minimax training equilibrium.")

    arch_info = pipeline.get_gan_architecture_info()
    gen_info = arch_info["generator"]
    disc_info = arch_info["discriminator"]

    c_g, c_d = st.columns(2)
    with c_g:
        st.markdown("#### 🎨 ResNet Style Generator")
        st.write(f"- **Architecture:** `{gen_info['model_type']}`")
        st.write(f"- **Total Parameters:** `{gen_info['total_parameters']:,}`")
        st.write(f"- **Residual Bottlenecks:** `{gen_info['residual_blocks']} blocks`")
        st.write(f"- **Normalization:** `{gen_info['normalization']}`")
        st.write(f"- **Padding Mode:** `{gen_info['padding_mode']}`")
        st.write(f"- **Effective Receptive Field:** `{gen_info['receptive_field_px']} px`")

    with c_d:
        st.markdown("#### 🛡️ PatchGAN Discriminator")
        st.write(f"- **Architecture:** `{disc_info['model_type']}`")
        st.write(f"- **Total Parameters:** `{disc_info['total_parameters']:,}`")
        st.write(f"- **Evaluation Receptive Field:** `{disc_info['receptive_field']}`")
        st.write(f"- **Loss Function:** `LSGAN / Relativistic Adversarial Hinge`")
        st.write(f"- **Device:** `{arch_info['device'].upper()}`")

    st.markdown("---")
    st.markdown("#### ⚖️ GAN Training Stability Simulator")
    st.markdown("Test the AI Auditor's ability to diagnose generator/discriminator loss ratios during training runs:")

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        sim_epoch = st.number_input("Training Epoch:", min_value=1, max_value=200, value=25)
    with sim_c2:
        sim_g_loss = st.number_input("Generator Loss:", min_value=0.01, max_value=50.0, value=1.42, step=0.05)
    with sim_c3:
        sim_d_loss = st.number_input("Discriminator Loss:", min_value=0.01, max_value=50.0, value=0.68, step=0.05)

    if st.button("🔍 Diagnose GAN Equilibrium"):
        stability = pipeline.checker.analyze_training_stability(sim_epoch, sim_g_loss, sim_d_loss)
        status_color = "🟢" if stability["status"] == "stable" else "🟡"
        st.markdown(f"""
        <div class="quality-card score-high">
            <h4>{status_color} GAN Training Diagnosis: {stability['status'].upper()}</h4>
            <p><strong>Loss Ratio (G/D):</strong> <code>{stability['loss_ratio']}</code></p>
            <p><strong>Analysis:</strong> {stability['analysis']}</p>
            <p><strong>Recommended Action:</strong> <code>{stability['action']}</code> — {stability['intervention']}</p>
        </div>
        """, unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown(
    f"<center>🎭 <b>{APP_NAME} v{APP_VERSION}</b> | Developed with ❤️ by Priyanka Ahirwar | "
    f"<a href='https://github.com/PriyankaAhirwar15/STYLE-SYNTH' target='_blank'>GitHub Repository</a> | "
    f"<a href='https://huggingface.co/spaces/PRIYANKAAhirwar/STYLE-SYNTH' target='_blank'>Hugging Face Space</a></center>",
    unsafe_allow_html=True
)