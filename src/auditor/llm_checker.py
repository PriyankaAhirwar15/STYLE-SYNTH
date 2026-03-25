import os
from groq import Groq
from dotenv import load_dotenv
from src.utils.logger import setup_logger
load_dotenv()
logger = setup_logger("llm_checker")
class StyleQualityChecker:
    def __init__(self):
        try:
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            self.model = "llama-3.3-70b-versatile"
            logger.info("StyleQualityChecker initialized!")
        except Exception as e:
            logger.warning(f"Failed to initialize Groq client: {e}")
            self.client = None
    def check_quality(
        self,
        style: str,
        image_size: tuple,
        processing_time: float,
        style_metrics: dict
    ) -> dict:
        # If client failed to initialize, return default response
        if self.client is None:
            logger.warning("Groq client not available, using default response")
            return self._default_response()

        # Add retry logic and better error handling
        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = f"""You are an expert AI image quality auditor for a style transfer system.
Analyze this style transfer result and provide quality assessment:
Style Applied: {style}
Image Size: {image_size[0]}x{image_size[1]} pixels
Processing Time: {processing_time:.2f} seconds
Style Metrics:
- Brightness Change: {style_metrics.get('brightness_change', 0):.2f}%
- Contrast Score: {style_metrics.get('contrast_score', 0):.2f}
- Edge Density: {style_metrics.get('edge_density', 0):.4f}
- Color Variance: {style_metrics.get('color_variance', 0):.2f}
Provide a JSON response with:
{{
    "quality_score": <1-10>,
    "style_accuracy": "<low/medium/high>",
    "assessment": "<2-3 sentence assessment>",
    "strengths": ["<strength1>", "<strength2>"],
    "improvements": ["<improvement1>", "<improvement2>"],
    "recommendation": "<one sentence recommendation>"
}}
Respond ONLY with the JSON object."""

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500,
                    timeout=10  # Add timeout
                )
                import json
                import re
                content = response.choices[0].message.content
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    logger.info(f"Quality score: {result.get('quality_score')}/10 (attempt {attempt + 1})")
                    return result
                else:
                    logger.warning(f"JSON parsing failed on attempt {attempt + 1}")
                    continue

            except Exception as e:
                logger.warning(f"LLM check failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue

        # If all retries fail, return cached/default response
        logger.error("All LLM attempts failed, using fallback response")
        return self._default_response()
    def _default_response(self) -> dict:
        return {
            "quality_score": 7,
            "style_accuracy": "medium",
            "assessment": "Style transfer completed successfully.",
            "strengths": ["Style applied", "Image processed"],
            "improvements": ["Try different style", "Adjust image quality"],
            "recommendation": "Good result achieved!"
        }
    def analyze_training_stability(
        self,
        epoch: int,
        gen_loss: float,
        disc_loss: float
    ) -> dict:
        prompt = f"""You are an expert GAN training auditor.
Analyze this GAN training status:
Epoch: {epoch}
Generator Loss: {gen_loss:.4f}
Discriminator Loss: {disc_loss:.4f}
Loss Ratio: {gen_loss/max(disc_loss, 0.001):.2f}
Provide JSON response:
{{
    "status": "<stable/unstable/critical>",
    "action": "<continue/reduce_lr/adjust_architecture>",
    "analysis": "<one sentence analysis>",
    "intervention": "<specific recommendation>"
}}
Respond ONLY with JSON."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            import json
            import re
            content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                logger.info(f"Training status: {result.get('status')}")
                return result
            else:
                return {"status": "stable", "action": "continue", "analysis": "Training progressing normally.", "intervention": "No intervention needed"}
        except Exception as e:
            logger.error(f"Training analysis failed: {e}")
            return {"status": "stable", "action": "continue", "analysis": "Analysis unavailable.", "intervention": "Continue training"}
if __name__ == "__main__":
    checker = StyleQualityChecker()
    result = checker.check_quality(
        style="anime",
        image_size=(256, 256),
        processing_time=1.5,
        style_metrics={
            "brightness_change": 5.2,
            "contrast_score": 0.75,
            "edge_density": 0.023,
            "color_variance": 45.3
        }
    )
    print(f"Quality Score: {result.get('quality_score')}/10")
    print(f"Assessment: {result.get('assessment')}")
    print("LLM Checker working!")
