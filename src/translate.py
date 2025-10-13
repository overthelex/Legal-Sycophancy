import asyncio
from typing import List, Optional
from googletrans import Translator
from src.utils import tqdm_gather
from tqdm.asyncio import tqdm


class TranslationClient:
    """Simple translation client using Google Translate."""

    def __init__(self, delay: float = 0.1):
        """
        Initialize the translation client.

        Args:
            delay: Delay between translation requests to avoid rate limiting
        """
        self.translator = Translator()
        self.delay = delay

    async def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
        retry_count: int = 3,
    ) -> str:
        """
        Translate text to target language.

        Args:
            text: Text to translate
            target_lang: Target language code (e.g., 'en', 'es', 'fr')
            source_lang: Source language code ('auto' for auto-detection)
            retry_count: Number of retry attempts

        Returns:
            Translated text
        """
        if source_lang == target_lang:
            return text

        for attempt in range(retry_count):
            try:
                # Add delay to respect rate limits
                if self.delay > 0:
                    await asyncio.sleep(self.delay)

                result = await self.translator.translate(
                    text, src=source_lang, dest=target_lang
                )

                if result and hasattr(result, "text"):
                    return result.text
                else:
                    print(
                        f"Warning: Translation attempt {attempt + 1} failed - no text returned"
                    )

            except Exception as e:
                print(f"Translation attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    print(
                        f"Failed to translate after {retry_count} attempts, returning original text"
                    )
                    return text

        return text

    async def translate_to_english(self, text: str, source_lang: str = "auto") -> str:
        """
        Translate text to English.

        Args:
            text: Text to translate
            source_lang: Source language code ('auto' for auto-detection)

        Returns:
            Text translated to English
        """
        return await self.translate(text, "en", source_lang)

    async def translate_batch(
        self,
        texts: List[str],
        target_lang: str,
        source_lang: str = "auto",
        batch_size: int = 20,
        desc: str = "Translating",
    ) -> List[str]:
        """
        Translate multiple texts concurrently.

        Args:
            texts: List of texts to translate
            target_lang: Target language code
            source_lang: Source language code ('auto' for auto-detection)
            batch_size: Number of texts to process concurrently

        Returns:
            List of translated texts
        """
        translated = []

        # Process in smaller batches to avoid overwhelming the API
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch_texts = texts[i : i + batch_size]

            # Create translation tasks for this batch
            batch_tasks = [
                self.translate(text, target_lang, source_lang) for text in batch_texts
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle any exceptions in the batch
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"Batch translation error: {result}")
                    translated.append("")  # Use original text
                else:
                    translated.append(result)

        return translated

    async def detect_language(self, text: str) -> Optional[str]:
        """
        Detect the language of text.

        Args:
            text: Text to analyze

        Returns:
            Language code or None if detection fails
        """
        try:
            detection = await self.translator.detect(text)
            return getattr(detection, "lang", None)
        except Exception as e:
            print(f"Language detection failed: {e}")
            return None


# Convenience function for quick usage
async def translate_text(
    text: str, target_lang: str, source_lang: str = "auto", delay: float = 0.1
) -> str:
    """
    Quick function to translate text.

    Args:
        text: Text to translate
        target_lang: Target language code
        source_lang: Source language code ('auto' for auto-detection)
        delay: Delay between requests

    Returns:
        Translated text
    """
    client = TranslationClient(delay=delay)
    return await client.translate(text, target_lang, source_lang)


async def main():
    """Example usage of the TranslationClient."""
    client = TranslationClient(delay=0.5)

    # Single translation
    print("=== Single Translation Example ===")
    text = "Hello, how are you today?"
    spanish_text = await client.translate(text, "es")
    print(f"English: {text}")
    print(f"Spanish: {spanish_text}")

    # Batch translation
    print("\n=== Batch Translation Example ===")
    texts = ["Good morning", "Thank you", "Goodbye"]
    french_texts = await client.translate_batch(texts, "fr")
    print(f"Batch translation to French:")
    for orig, trans in zip(texts, french_texts):
        print(f"{orig} -> {trans}")

    # Language detection
    print("\n=== Language Detection Example ===")
    mystery_text = "Bonjour le monde"
    detected_lang = await client.detect_language(mystery_text)
    print(f"Detected language for '{mystery_text}': {detected_lang}")


if __name__ == "__main__":
    asyncio.run(main())
