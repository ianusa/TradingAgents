import re


def trim_llm_invocation_result(text):
    """Trim LLM invocation result content."""
    if not isinstance(text, str):
        return text

    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    # # Replace multiple newlines with double newline
    # text = re.sub(r'\n{3,}', '\n\n', text)
    # # Remove spaces at the beginning and end of lines
    # text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)
    # # Replace tabs with single space
    # text = re.sub(r'\t+', ' ', text)
    # Remove trailing whitespace
    return text.strip()
