import re
import subprocess
import sys

PUNCTUATION_REGEX = r"[.,!?;:\"'()\[\]{}\-–—…]"
SPLIT_REGEX = f"({PUNCTUATION_REGEX}| )"


class EspeakPhonemizer:
    def __init__(self, language: str = "en-us"):
        self.language = language
        self._check_espeak_installed()

    def _check_espeak_installed(self):
        try:
            subprocess.run(['espeak-ng', '--version'], capture_output=True, text=True, check=True, encoding='utf-8')
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("ERROR: espeak-ng not installed!", file=sys.stderr)
            raise

    def _call_espeak_g2p(self, text: str) -> list[str]:
        command = ['espeak-ng', f'-v{self.language}', '--ipa=1', '-q', text]
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')

        phoneme_string = result.stdout.strip()
        phonemes = [p for p in phoneme_string.split('_') if p]
        return phonemes

    def _text_to_phonemes(self, text: str) -> list[str]:
        text_parts = [p for p in re.split(SPLIT_REGEX, text) if p]
        phonemeized_text = []
        
        for part in text_parts:
            if re.fullmatch(PUNCTUATION_REGEX, part):
                phonemeized_text.append(part)
            elif part == ' ':
                phonemeized_text.append(' ')
            else:
                phonemes = self._call_espeak_g2p(part)
                phonemeized_text.extend(phonemes)

        return phonemeized_text

    def __call__(self, text: str) -> list[str]:
        return self._text_to_phonemes(text)



if __name__ == "__main__":
    text = "This is a test!"
    phonemizer = EspeakPhonemizer()
    phonemized_text = phonemizer(text)
    print(phonemized_text)