import re
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

PUNCTUATION_REGEX = r"[.,!?;:\"'()\[\]{}\-–—…]"
SPLIT_REGEX = f"({PUNCTUATION_REGEX})"


class PhonemizerWrapper:
    def __init__(self, language: str = "en-us", with_stress: bool = True):
        self.language = language
        
        self.backend = EspeakBackend(
            language=language,
            preserve_punctuation=False,
            with_stress=with_stress,
            language_switch='remove-flags' 
        )
        
        # Define a separator that puts spaces between phones
        self.separator = Separator(phone="|", word=" ", syllable="")

    def _text_to_phonemes(self, text: str) -> list[str]:
        parts = [p for p in re.split(SPLIT_REGEX, text) if p]
        
        to_phonemize_indices = []
        to_phonemize_text = []
        
        for i, part in enumerate(parts):
            if re.fullmatch(PUNCTUATION_REGEX, part):
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
            # Split by the phone separator to get individual phonemes
            p_str = p_str.replace(" ", "| |")
            raw_phonemes = [p for p in p_str.split('|') if p]
            
            # stress marks are attached to the phoneme, if with_stress=True
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
                     
            elif re.fullmatch(PUNCTUATION_REGEX, part):
                result.append(part)
            elif not part.strip() and ' ' in part:
                result.append(' ')
        
        # Clean up double spaces if they occur
        final_result = []
        for token in result:
            if token == ' ' and final_result and final_result[-1] == ' ':
                continue
            final_result.append(token)
            
        return final_result

    def __call__(self, text: str) -> list[str]:
        return self._text_to_phonemes(text)


if __name__ == "__main__":
    phonemizer = PhonemizerWrapper()
    
    test_cases = [
        "Hello, world! This is a test.",
        "I read a book yesterday.",
        "I will read a book tomorrow.",
        "The wind is strong.",
        "Please wind the clock.",
        "The apple vs. the computer.",
        "  Leading and trailing spaces.  ",
        "Multiple   internal   spaces.",
        "Spaces around punctuation , like this .",
        "!!!",
        "Mr. Smith went to 123 Main St.",
        "Heteronyms: The dove dove into the bushes.",
        "Complex: I live in Live Oak.",
    ]
    
    for text in test_cases:
        print(f"Input: '{text}'")
        print(f"Output: {phonemizer(text)}")
        print("-" * 20)
