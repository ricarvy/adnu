"""Inference-time demonstration of the cultural prompt templates.

This script showcases Innovation 3: cultural-aware prompt templates and
prior injection for Chinese scenarios such as Dragon Boat Festival and
Mid-Autumn Festival.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from accessory.data.template import CulturalPromptTemplate


def run_inference_demo() -> None:
    """Run a small console demo of cultural-aware prompt injection."""
    print("--- Cultural-aware Prompt Injection Demo ---")

    detected_objects = ["person", "boat", "zongzi"]
    base_prompt = "Describe the food in the image."

    print(f"\nScenario 1: Detected objects: {detected_objects}")
    print(f"Base Prompt: {base_prompt}")

    enhanced_prompt = CulturalPromptTemplate.inject_prior(base_prompt, detected_objects)
    print(f"Enhanced Prompt: {enhanced_prompt}")

    template = CulturalPromptTemplate.get_template(
        "zh", "festival", {"festival": "端午节", "object": "粽子"}
    )
    print(f"Specific Template (ZH): {template}")

    detected_objects_2 = ["moon", "mooncake"]
    base_prompt_2 = "What is the round object?"

    print(f"\nScenario 2: Detected objects: {detected_objects_2}")
    print(f"Base Prompt: {base_prompt_2}")

    enhanced_prompt_2 = CulturalPromptTemplate.inject_prior(base_prompt_2, detected_objects_2)
    print(f"Enhanced Prompt: {enhanced_prompt_2}")


if __name__ == "__main__":
    run_inference_demo()
