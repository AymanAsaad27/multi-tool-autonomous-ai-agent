def text_of(message) -> str:
    """
    Convert an LLM message's content into plain text.

    Gemini may return structured content instead of a simple string,
    so this helper keeps the rest of the agent code simple.
    """

    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                text = item.get("text")

                if text:
                    parts.append(str(text))

        return "\n".join(parts)

    return str(content)
