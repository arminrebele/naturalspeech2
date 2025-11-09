import json
import sys
from tqdm import tqdm
from collections.abc import Iterable

from audiodeepfakedetection_ddim_inversion.data.espeak_phonemizer import EspeakPhonemizer
from audiodeepfakedetection_ddim_inversion.paths import TOKEN_VOCABULARY_PATH

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]



class PhonemeTokenizer:
    def __init__(self, phonemizer: EspeakPhonemizer, token_vocabulary_path: str = TOKEN_VOCABULARY_PATH):
        self.phonemizer = phonemizer
        self.token_vocabulary_path = token_vocabulary_path
        self.token_vocabulary = {} # token to id
        self.id_to_token = {}
        self._load_token_vocabulary()

    def _load_token_vocabulary(self):
        try:
            with open(self.token_vocabulary_path, 'r', encoding='utf-8') as f:
                self.token_vocabulary = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: {self.token_vocabulary_path} not found", file=sys.stderr)
            raise

        self.id_to_token = {v: k for k, v in self.token_vocabulary.items()}

    def __call__(self, text: str) -> list[int]:
        tokens =  ["<bos>"] + self.phonemizer(text) + ["<eos>"]
        unk_id = self.token_vocabulary.get("<unk>")
        return [self.token_vocabulary.get(token, unk_id) for token in tokens]

    def decode_tokens(self, token_ids: list[int]) -> list[str]:
        """Wandelt Token-IDs zurück in Tokens (Phoneme oder Sonderzeichen)."""
        return [self.id_to_token.get(i, "<unk>") for i in token_ids]



def build_token_vocabulary(dataset: Iterable[dict], phonemizer: EspeakPhonemizer, save_path: str = TOKEN_VOCABULARY_PATH) -> dict[str, int]:
    
    tokens_from_text = set()

    for sample in tqdm(dataset):
        text = sample["text"]
        phonemized_text = phonemizer(text)
        tokens_from_text.update(phonemized_text)

    tokens = SPECIAL_TOKENS + sorted(tokens_from_text)

    token_vocabulary = {token: idx for idx, token in enumerate(tokens)}

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(token_vocabulary, f, ensure_ascii=False, indent=2)

    return token_vocabulary

if __name__ == "__main__":
    from audiodeepfakedetection_ddim_inversion.data.vctk import VCTKDataset

    dataset = VCTKDataset()
    dataset.dataset = dataset.dataset.select(range(100))
    phonemizer = EspeakPhonemizer()
    token_vocabulary = build_token_vocabulary(dataset, phonemizer)
    print(len(token_vocabulary))

    tokenizer = PhonemeTokenizer(phonemizer)
    text = "Hello, world!"
    print(tokenizer(text))
