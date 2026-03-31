import json
from tqdm import tqdm
from collections.abc import Iterable

from naturalspeech2.data.phonemizer_wrapper import PhonemizerWrapper


class PhonemeTokenizer:
    def __init__(self, phonemizer: PhonemizerWrapper = None, token_vocabulary_path: str = None, with_backend: bool = True):
        self.phonemizer = phonemizer
        if self.phonemizer is None and with_backend:
            self.phonemizer = PhonemizerWrapper()
            
        self.token_vocabulary_path = token_vocabulary_path
        self.token_vocabulary = {} # token to id
        self.id_to_token = {}
        self._load_token_vocabulary()
        self.unk_id = self.token_vocabulary.get("<unk>")

    def _load_token_vocabulary(self):
        if self.token_vocabulary_path is None:
            raise ValueError("token_vocabulary_path must be provided")

        with open(self.token_vocabulary_path, "r", encoding="utf-8") as f:
            self.token_vocabulary = json.load(f)

        self.id_to_token = {v: k for k, v in self.token_vocabulary.items()}

    @property
    def token_vocabulary_size(self) -> int:
        return len(self.token_vocabulary)
    
    def __call__(self, input_data) -> list[int]:
        if isinstance(input_data, str):
            phonemes = self.phonemizer(input_data)
        elif isinstance(input_data, list):
            phonemes = input_data
        tokens =  ["<bos>"] + phonemes + ["<eos>"]
        return [self.token_vocabulary.get(token, self.unk_id) for token in tokens]

    def decode_tokens(self, token_ids: list[int]) -> list[str]:
        """Converts token-IDs back into strings (phonemes or special characters)."""
        return [self.id_to_token.get(i, "<unk>") for i in token_ids]



def build_token_vocabulary(
        dataset: Iterable[dict],
        special_tokens: list[str] = ["<pad>", "<unk>", "<bos>", "<eos>"], 
        save_path: str = None) -> dict[str, int]:
    
    tokens_from_text = set()
    for sample in tqdm(dataset, desc="Building token vocabulary"):
        tokens_from_text.update(sample["phonemes"])

    tokens = special_tokens + sorted(tokens_from_text)
    token_vocabulary = {token: idx for idx, token in enumerate(tokens)}

    if save_path is None:
        raise ValueError("save_path must be provided")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(token_vocabulary, f, ensure_ascii=False, indent=2)

    return token_vocabulary

if __name__ == "__main__":

    # # Code to build token vocabulary based on VCTK dataset
    # from naturalspeech2.data.vctk import VCTKDataset
    # dataset = VCTKDataset()
    # #dataset.dataset = dataset.dataset.select(range(200))
    # token_vocabulary = build_token_vocabulary(dataset)
    # print(len(token_vocabulary))

    # Test code to verify tokenizer
    # tokenizer = PhonemeTokenizer(token_vocabulary_path="path/to/vocab.json")
    # text = "This is a test!"
    # print(tokenizer(text))
    pass
