import json
import re
import subprocess
from tqdm import tqdm

from audiodeepfakedetection_ddim_inversion.data.vctk import VCTKDataset
from audiodeepfakedetection_ddim_inversion.paths import PHONEME_VOCAB_PATH

PUNCTUATION_REGEX = r"[.,!?;:\"'()\[\]{}\-–—…]"
SPLIT_REGEX = f'({PUNCTUATION_REGEX}| )'
SPECIAL_TOKENS = ['<pad>', '<unk>', '<bos>', '<eos>']

def _check_espeak_installed():
    try:
        result = subprocess.run(['espeak-ng', '--version'], capture_output=True, text=True, check=True, encoding='utf-8')
        return True
    except Exception:
        raise

def _call_espeak_g2p(text: str, language: str) -> list[str]:
    command = ['espeak-ng', f'-v{language}', '--ipa=1', '-q', text]
    result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')

    phoneme_string = result.stdout.strip()
    phonemes = [p for p in phoneme_string.split('_') if p]
    return phonemes

def _text_to_phonemes(text: str, language: str) -> list[str]:
    text_parts = [p for p in re.split(SPLIT_REGEX, text) if p]
    phonemeized_text = []
    
    for part in text_parts:
        if re.fullmatch(PUNCTUATION_REGEX, part):
            phonemeized_text.append(part)
        elif part == ' ':
            phonemeized_text.append(' ')
        else:
            phonemes = _call_espeak_g2p(part, language)
            phonemeized_text.extend(phonemes)

    return phonemeized_text


class PhonemeTokenizer:

    def __init__(self):
        self.language = "en-us"
        self.vocabulary = self._load_vocabulary()


    def _load_vocabulary(self):
        pass

    def __call__(self, text: str) -> list[int]:
        pass



def build_vocabulary():
    pass



if __name__ == "__main__":
    #build_vocabulary()
    language = "en-us"
    text = "123 Hello, world! This is a test."
    phonemized_text = _text_to_phonemes(text, language)
    print(phonemized_text)
