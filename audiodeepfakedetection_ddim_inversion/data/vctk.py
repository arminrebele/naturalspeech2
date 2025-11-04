from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, Audio

from audiodeepfakedetection_ddim_inversion.paths import VCTK_DIR, VCTK_PROCESSED_DIR

class VCTKDataset(Dataset):
    def __init__(self, sampling_rate=24000):
        self.sampling_rate = sampling_rate
        self.dataset = self._process_dataset()

    def _process_dataset(self):
        try:
            return load_from_disk(str(VCTK_PROCESSED_DIR))
        except Exception:
            VCTK_DIR.mkdir(parents=True, exist_ok=True)
            dataset = load_dataset("sanchit-gandhi/vctk", split="train", cache_dir=str(VCTK_DIR))
            dataset = dataset.filter(lambda sample: "_mic2" in sample, input_columns=["file"])
            dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
            VCTK_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(VCTK_PROCESSED_DIR))
            return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "audio": item["audio"]["array"],
            "sampling_rate": item["audio"]["sampling_rate"],
            "transcript": item["text"],
            "speaker_id": item["speaker_id"],
            "age": int(item["age"]), 
            "gender": item["gender"],
            "accent": item["accent"],
            "region": item["region"],
        }



if __name__ == "__main__":
    
    # test code to verify dataset loading
    dataset = VCTKDataset()
    print(len(dataset))
    sample = dataset[0]
    print(sample)