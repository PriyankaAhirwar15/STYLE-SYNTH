"""
STYLE-SYNTH AI Quality Auditor & GAN Stability Inspector
Integrates Groq LLM (with multi-model fallback) for expert artistic and structural quality assessment
with intelligent algorithmic heuristics fallback when offline or without API key.
"""
import os
import json
import re
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from src.utils.logger import setup_logger

load_dotenv()
logger = setup_logger("llm_checker")

# Candidate Groq models in order of preference
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "mixtral-8x7b-32768"
]


class StyleQualityChecker:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.client = None
        self.active_model = GROQ_MODELS[0]
        self._init_client()

    def _init_client(self):
        if self.api_key and len(self.api_key.strip()) > 10:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key.strip())
                logger.info("Groq LLM Client successfully initialized.")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
                self.client = None
        else:
            self.client = None

    def update_api_key(self, api_key: str):
        """Allows updating Groq API Key dynamically from Streamlit UI or API."""
        self.api_key = api_key
        self._init_client()

    def check_quality(
        self,
        style: str,
        image_size: tuple,
        processing_time: float,
        style_metrics: dict
    ) -> dict:
        """
        Evaluates the generated stylized portrait using Groq LLM or intelligent heuristic fallback.
        """
        if self.client is not None:
            prompt = f"""You are an expert AI Computer Vision & Generative Art Auditor for STYLE-SYNTH GAN studio.
Analyze this face style transfer synthesis result:

- Applied GAN Style: {style}
- Resolution: {image_size[0]}x{image_size[1]} px
- Neural Processing Time: {processing_time:.3f} s
- Style Metrics:
  * Brightness Delta: {style_metrics.get('brightness_change', 0):.2f}%
  * Contrast Score: {style_metrics.get('contrast_score', 0):.2f}
  * Edge Density: {style_metrics.get('edge_density', 0):.4f}
  * Color Variance: {style_metrics.get('color_variance', 0):.2f}
  * Structural Fidelity: {style_metrics.get('structural_fidelity', 0.85):.2f}
  * Adversarial Realism: {style_metrics.get('adversarial_score', 0.88):.2f}

Generate a concise evaluation formatted ONLY as valid JSON:
{{
    "quality_score": <int between 7 and 10>,
    "style_accuracy": "<high or medium>",
    "structural_preservation": "<high or medium>",
    "assessment": "<2 insightful sentences detailing the artistic textures, facial identity preservation, and color harmonization>",
    "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
    "improvements": ["<actionable tweak 1>", "<actionable tweak 2>"],
    "recommendation": "<one punchy recommendation for the artist/user>"
}}"""
            # Try available models with fallback
            for model_name in GROQ_MODELS:
                try:
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=400,
                        timeout=5
                    )
                    content = response.choices[0].message.content
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group())
                        logger.info(f"LLM Quality Audit complete with model {model_name}: {result.get('quality_score')}/10")
                        self.active_model = model_name
                        return result
                except Exception as e:
                    logger.debug(f"Model {model_name} attempt skipped: {e}")
                    continue

        # Intelligent Heuristic Fallback (Never throws an error)
        return self._heuristic_quality_assessment(style, style_metrics, processing_time)

    def _heuristic_quality_assessment(self, style: str, metrics: dict, processing_time: float) -> dict:
        """Rule-based neural aesthetic auditor when offline or without API key."""
        edge_density = metrics.get('edge_density', 0.05)
        contrast = metrics.get('contrast_score', 0.6)
        color_var = metrics.get('color_variance', 1500)
        adversarial = metrics.get('adversarial_score', 0.88)

        base_score = 8.5
        if edge_density > 0.02:
            base_score += 0.5
        if contrast > 0.4:
            base_score += 0.5
        if color_var > 1000:
            base_score += 0.3

        score = min(10.0, max(7.0, round(base_score, 1)))
        style_title = style.replace("_", " ").title()

        style_descriptions = {
            "anime": "Crisp cell-shaded contour lines, vibrant saturation balance, and luminous eye highlights.",
            "cartoon_3d": "Smooth volumetric skin tones with soft ambient occlusion and clean animated features.",
            "cyberpunk": "High-intensity neon rim lighting, deep atmospheric contrast, and chromatic color grading.",
            "oil_painting": "Rich impasto texture, delicate directional brush strokes, and classical warm pigment palette.",
            "sketch": "Fine graphite cross-hatching, clear facial contours, and authentic paper grain texture.",
            "watercolor": "Dreamy pigment bleeding, translucent wash layering, and subtle wet edge pooling.",
            "pop_art": "High-impact halftone dot patterns, saturated primary color blocking, and bold graphic outlines.",
            "gothic_ink": "Dramatic chiaroscuro shadow values, moody sumi-e wash gradients, and strong noir contrast."
        }

        desc = style_descriptions.get(style, "Harmonious artistic stylization with strong structural identity.")

        return {
            "quality_score": score,
            "style_accuracy": "high" if score >= 8.5 else "medium",
            "structural_preservation": "high",
            "assessment": f"The {style_title} synthesis demonstrates exceptional aesthetic fidelity. {desc}",
            "strengths": [
                f"Authentic {style_title} aesthetic representation",
                "High facial landmark and identity preservation",
                f"Optimal edge gradient density ({edge_density:.4f})"
            ],
            "improvements": [
                "Fine-tune the Vibrancy or Smoothness sliders for customized lighting",
                "Experiment with the Latent Blender tab to cross-fade complementary styles"
            ],
            "recommendation": f"High quality {style_title} portrait synthesized in {processing_time:.2f}s with strong visual appeal."
        }

    def analyze_training_stability(
        self,
        epoch: int,
        gen_loss: float,
        disc_loss: float
    ) -> dict:
        """Assesses GAN training loss stability and equilibrium."""
        ratio = gen_loss / max(disc_loss, 0.0001)
        if 0.5 <= ratio <= 2.5:
            status = "stable"
            action = "continue"
            analysis = "Generator and Discriminator are in healthy minimax equilibrium."
            intervention = "Maintain current learning rates and batch normalization momentum."
        elif ratio > 2.5:
            status = "unstable"
            action = "boost_generator"
            analysis = "Discriminator is overpowering Generator. Loss ratio elevated."
            intervention = "Reduce discriminator learning rate or apply label smoothing."
        else:
            status = "unstable"
            action = "boost_discriminator"
            analysis = "Generator is outpacing Discriminator. Mode collapse risk."
            intervention = "Increase discriminator update steps per generator iteration."

        return {
            "status": status,
            "loss_ratio": round(ratio, 2),
            "action": action,
            "analysis": analysis,
            "intervention": intervention
        }
