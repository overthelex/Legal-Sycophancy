"""
Placeholder replacement logic for ECHR case texts.
"""

import re
from typing import Dict, List


def check_conflicts(keys: List[str], strategies: Dict) -> None:
    """
    Check if strategy keys have conflicting placeholder replacements.

    Args:
        keys: List of strategy keys to check
        strategies: Dictionary of all strategies

    Raises:
        ValueError: If conflicts are detected
    """
    seen_placeholders = {}

    for key in keys:
        strategy = strategies[key]

        # Check simple replacements
        for placeholder in strategy.get('replacements', {}).keys():
            if placeholder in seen_placeholders:
                raise ValueError(
                    f"Conflict detected: placeholder '{placeholder}' is defined in both "
                    f"'{seen_placeholders[placeholder]}' and '{key}'. "
                    f"Each placeholder can only be replaced by one strategy."
                )
            seen_placeholders[placeholder] = key

        # Check complex replacements
        for placeholder in strategy.get('complex_replacements', {}).keys():
            if placeholder in seen_placeholders:
                raise ValueError(
                    f"Conflict detected: placeholder '{placeholder}' is defined in both "
                    f"'{seen_placeholders[placeholder]}' and '{key}'. "
                    f"Each placeholder can only be replaced by one strategy."
                )
            seen_placeholders[placeholder] = key


def apply_complex_replacement(text: str, placeholder: str, rule: Dict) -> str:
    """
    Apply complex replacement rule (e.g., contextual pronouns).

    Args:
        text: Text to apply replacement to
        placeholder: Placeholder to replace
        rule: Replacement rule dictionary

    Returns:
        Text with replacement applied
    """
    if rule['type'] == 'contextual_pronoun':
        subject = rule['subject']
        object_form = rule['object']
        object_verbs = rule['object_verbs']

        # Build pattern for object form (after verbs)
        object_verb_pattern = r'(' + '|'.join(object_verbs) + r')\s+' + re.escape(placeholder)
        text = re.sub(object_verb_pattern, rf'\1 {object_form}', text, flags=re.IGNORECASE)

        # Replace remaining with subject form, preserving capitalization
        def replace_pronoun(match):
            return subject.capitalize() if match.group(0)[0].isupper() else subject

        text = re.sub(re.escape(placeholder), replace_pronoun, text)

    return text


def apply_replacements(text: str, keys: List[str], strategies: Dict) -> str:
    """
    Apply all replacements from the given strategy keys.

    Args:
        text: Text to apply replacements to
        keys: List of strategy keys to apply
        strategies: Dictionary of all strategies

    Returns:
        Text with all replacements applied
    """
    # Collect all replacements
    simple_replacements = {}
    complex_replacements = {}

    for key in keys:
        strategy = strategies[key]
        simple_replacements.update(strategy.get('replacements', {}))
        complex_replacements.update(strategy.get('complex_replacements', {}))

    # Apply simple replacements (order matters - longer patterns first)
    for placeholder in sorted(simple_replacements.keys(), key=len, reverse=True):
        replacement = simple_replacements[placeholder]
        text = text.replace(placeholder, replacement)

    # Apply complex replacements
    for placeholder, rule in complex_replacements.items():
        text = apply_complex_replacement(text, placeholder, rule)

    return text
