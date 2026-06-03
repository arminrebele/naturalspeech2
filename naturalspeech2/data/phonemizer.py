import re
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

PUNCTUATION_SET = set(".,!?;:\"'()[]{}–—…-")
CLOSING_PUNCTUATION_SET = set(".,!?;:)}…]")

# Split on punctuation; keep contractions (don't) by only splitting ' when not word-surrounded.
SPLIT_REGEX = re.compile(r"([.,!?;:\"()\[\]{}\-–—…]|(?<!\w)'|'(?!\w))")

# Space before opening brackets if not preceded by whitespace
OPENING_BRACKETS_REGEX = re.compile(r'(?<!\s)([(\[{])')
# Space after closing punct if not followed by whitespace/digit
CLOSING_PUNCTUATION_REGEX = re.compile(r'([.,!?;:)}\]…])(?!\s|\d)')


class PhonemizerWrapper:
    def __init__(self, language: str = "en-us", with_stress: bool = True):
        self.language = language
        
        self.backend = EspeakBackend(
            language=language,
            preserve_punctuation=False,
            with_stress=with_stress,
            language_switch='remove-flags' 
        )
        
        # Separator: | between phones, space between words
        self.separator = Separator(phone="|", word=" ", syllable="")

    def _text_to_phonemes(self, text: str) -> list[str]:
        # Normalize spacing around punctuation
        text = OPENING_BRACKETS_REGEX.sub(r' \1', text)
        text = CLOSING_PUNCTUATION_REGEX.sub(r'\1 ', text)
        
        parts = [p for p in SPLIT_REGEX.split(text) if p]
        
        to_phonemize_indices = []
        to_phonemize_text = []
        
        for i, part in enumerate(parts):
            if part in PUNCTUATION_SET:
                continue
            if not part.strip():
                continue
            
            to_phonemize_indices.append(i)
            to_phonemize_text.append(part)
            
        if not to_phonemize_text:
            return self._reconstruct(parts, {})

        phonemized_list = self.backend.phonemize(
            to_phonemize_text, 
            strip=True, 
            separator=self.separator
        )
        
        phoneme_map = {}
        for idx, p_str in zip(to_phonemize_indices, phonemized_list):
            # Split on phone separator → individual phonemes
            p_str = p_str.replace(" ", "| |")
            raw_phonemes = [p for p in p_str.split('|') if p]

            # stress marks attached to phoneme (with_stress=True)
            phoneme_map[idx] = raw_phonemes

        return self._reconstruct(parts, phoneme_map)

    def _reconstruct(self, parts: list[str], phoneme_map: dict) -> list[str]:
        result = []
        for i, part in enumerate(parts):
            if i in phoneme_map:
                phonemes = phoneme_map[i]
                
                if part.startswith(' '):
                    result.append(' ')
                
                result.extend(phonemes)
                
                if part.endswith(' ') and len(part.strip()) > 0:
                     result.append(' ')
                     
            elif part in PUNCTUATION_SET:
                result.append(part)
            elif not part.strip() and ' ' in part:
                result.append(' ')

        # Clean up spaces
        final_result = []
        for token in result:
            if token == ' ':
                # Skip leading spaces and double spaces
                if not final_result or final_result[-1] == ' ':
                    continue
                # Skip spaces after opening brackets
                if final_result[-1] in {'(', '[', '{'}:
                    continue
            elif token in CLOSING_PUNCTUATION_SET:
                # Remove space before closing punctuation
                if final_result and final_result[-1] == ' ':
                    final_result.pop()
            final_result.append(token)
        
        # Remove trailing space
        if final_result and final_result[-1] == ' ':
            final_result.pop()
            
        return final_result

    def __call__(self, text: str) -> list[str]:
        return self._text_to_phonemes(text)


if __name__ == "__main__":
    phonemizer = PhonemizerWrapper()
    
    test_cases = [
        "Hello, world! This is a test.",                # basic sentence with punctuation
        "Hello, world!This is a test.",                 # punctuation without space
        "I read a book yesterday.",                     # homograph test (read can be present or past tense)
        "I will read a book tomorrow.",
        "The wind is strong.",                          # homograph test (wind can be noun or verb)
        "Please wind the clock.",                   
        "The apple vs. the computer.",                  # phonetic environment test (the vs. "thi")
        "  Leading and trailing spaces.  ",             # leading/trailing spaces
        "Multiple   internal   spaces.",                # multiple internal spaces
        "Spaces around punctuation , like this .",      # spaces around punctuation
        "!!!",                                          # only punctuation
        "I said: \"Hello, don't do that!\"",            # punctuation with quotes and contractions
        "I said: \'Hello, don't do that!\'",            # punctuation with single quotes and contractions
        "I said: \' Hello, don't do that! \' ",         # spaces around punctuation
        "He wouldn't have done that.",                  # contractions
        "LLMs ( Large Language Models )are amazing.",    # parentheses and acronyms
        "Mr. Smith went to 123 Main St.",               # numbers and abbreviations
        "Mr Smith went to 123 Main St today.",          # abbreviations without points
        "$5 for coffee, €3 for tea.",                   # special characters and currency symbols
        "$ 5 and 5 $, with µ=5.",                       # currency symbols with spaces and special characters
        "100 is less than 103.58 and $3 < $5.3."        # numbers with decimals and currency symbols
    ]
    
    for text in test_cases:
        print(f"Input: '{text}'")
        print(f"Output: {phonemizer(text)}")
        print("-" * 20)
