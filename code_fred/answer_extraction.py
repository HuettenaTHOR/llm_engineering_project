import re


def extract_float(text: str) -> float | None:
    """
    Given a model output string, tries to find #### and extract the last float from it. If not found, returns None.
    """
    findings = re.findall(r"####\s*(-?\d[\d,]*)", text)
    if findings:
        return float(findings[-1].replace(",", ""))  # commas are thousands separators

    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if matches:
        value = matches[-1].replace(",", "")
        try:
            return float(value)
        except ValueError:
            return None
    return None

if __name__ == "__main__":
    # Example usage
    test_strings = [
        "The answer is #### 42.",
        "The answer is 42.",
        "The answer is #### -42.",
        "The answer is -42.",
        "Multiple numbers: 10 and #### 20.",
        "multiple numbers: #### 10 and 20.",
        "Multiple numbers: #### 10 and #### 20.",
    ]

    for s in test_strings:
        extracted = extract_float(s)
        print(f"Input: {s} -> Extracted: {extracted}")