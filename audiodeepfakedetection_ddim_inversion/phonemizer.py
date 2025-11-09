import json
import re
import subprocess
import sys
from tqdm import tqdm
from collections.abc import Iterable

from audiodeepfakedetection_ddim_inversion.data.vctk import VCTKDataset
from audiodeepfakedetection_ddim_inversion.paths import TOKEN_VOCABULARY_PATH

PUNCTUATION_REGEX = r"[.,!?;:\"'()\[\]{}\-–—…]"
SPLIT_REGEX = f'({PUNCTUATION_REGEX}| )'
SPECIAL_TOKENS = ['<pad>', '<unk>', '<bos>', '<eos>']

def _check_espeak_installed():
    try:
        subprocess.run(['espeak-ng', '--version'], capture_output=True, text=True, check=True, encoding='utf-8')
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: espeak-ng not installed!", file=sys.stderr)
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
        self.token_vocabulary = {} # token to id
        self.id_to_token = {}

        _check_espeak_installed()
        self._load_token_vocabulary()

    def _load_token_vocabulary(self):
        try:
            with open(TOKEN_VOCABULARY_PATH, 'r', encoding='utf-8') as f:
                self.token_vocabulary = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: {TOKEN_VOCABULARY_PATH} not found", file=sys.stderr)
            raise


    def __call__(self, text: str) -> list[int]:
        pass


def build_token_vocabulary(dataset: Iterable[dict], language: str = "en-us") -> dict[str, int]:

    _check_espeak_installed()

    tokens_from_text = set()
    for sample in tqdm(dataset):
        text = sample["text"]
        phonemized_text = _text_to_phonemes(text, language)
        tokens_from_text.update(phonemized_text)

    tokens = SPECIAL_TOKENS + sorted(tokens_from_text)

    token_vocabulary = {token: idx for idx, token in enumerate(tokens)}

    with open(TOKEN_VOCABULARY_PATH, "w", encoding="utf-8") as f:
        json.dump(token_vocabulary, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    language = "en-us"
    text = "123 Hello, world! This is a test."
    phonemized_text = _text_to_phonemes(text, language)
    print(phonemized_text)
