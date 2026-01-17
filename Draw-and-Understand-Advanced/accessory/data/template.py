"""Cultural-aware prompt templates and prior injection utilities.

This module implements Innovation 3: addressing multilingual and
cultural bias by injecting language- and culture-specific priors into
the textual prompt.
"""


class CulturalPromptTemplate:
    """Utility class for constructing culturally informed text prompts."""

    TEMPLATES = {
        "zh": {
            "default": "请描述选定区域的内容。",
            "festival": "结合{festival}的传统习俗，描述这个{object}。",
            "idiom": "用成语概括这个场景：{context}。",
        },
        "en": {
            "default": "Describe the content of the selected region.",
            "festival": "Describe this {object} in the context of {festival} traditions.",
        },
    }

    FESTIVAL_KEYWORDS = {
        "zongzi": "Dragon Boat Festival",
        "mooncake": "Mid-Autumn Festival",
        "dumpling": "Spring Festival",
    }

    @staticmethod
    def get_template(lang: str, key: str = "default", context=None) -> str:
        """Retrieve a formatted template string for a given language and key.

        Args:
            lang: Language code, e.g. "zh" or "en".
            key: Template key within the language dictionary.
            context: Optional formatting dictionary for template
                placeholders.

        Returns:
            A fully formatted template string if found, otherwise an
            empty string.
        """
        tmpl = CulturalPromptTemplate.TEMPLATES.get(lang, {}).get(key, "")
        if context:
            return tmpl.format(**context)
        return tmpl

    @staticmethod
    def inject_prior(prompt_text: str, detected_objects):
        """Augment a base prompt with cultural priors inferred from objects.

        Args:
            prompt_text: Original textual prompt.
            detected_objects: List of object labels obtained from the
                visual encoder or detector.

        Returns:
            The prompt possibly enriched with a short cultural note if
            any detected object matches a known cultural keyword.
        """
        for obj in detected_objects:
            if obj in CulturalPromptTemplate.FESTIVAL_KEYWORDS:
                festival = CulturalPromptTemplate.FESTIVAL_KEYWORDS[obj]
                return f"{prompt_text} Note: This is related to {festival}."
        return prompt_text
