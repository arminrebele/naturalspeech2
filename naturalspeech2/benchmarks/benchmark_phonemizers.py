import time
from tqdm import tqdm
from datasets import load_dataset

from naturalspeech2.testing.espeak_phonemizer import EspeakPhonemizer
from naturalspeech2.data.phonemizer_wrapper import PhonemizerWrapper

def benchmark():
    print("--- Phonemizer Benchmark ---")
    
    # 1. Load Dataset
    print("Loading VCTK dataset (full download)...")
    dataset = load_dataset("sanchit-gandhi/vctk", split="train")
    
    # Collect a subset of samples
    num_samples = 1000
    print(f"Collecting {num_samples} samples for benchmarking...")
    samples = dataset.select(range(num_samples))["text"]

    print(f"Benchmarking on {len(samples)} sentences.")

    # 2. Initialize Phonemizers
    print("\nInitializing EspeakPhonemizer (Old approach)...")
    try:
        old_phonemizer = EspeakPhonemizer()
    except Exception as e:
        print(f"Failed to init old phonemizer: {e}")
        return

    print("Initializing PhonemizerWrapper (New approach)...")
    try:
        new_phonemizer = PhonemizerWrapper()
    except Exception as e:
        print(f"Failed to init new phonemizer: {e}")
        return

    # 3. Run Benchmark - Old
    print("\nRunning EspeakPhonemizer...")
    start_old = time.time()
    for text in tqdm(samples):
        _ = old_phonemizer(text)
    end_old = time.time()
    time_old = end_old - start_old

    # 4. Run Benchmark - New
    print("\nRunning PhonemizerWrapper...")
    start_new = time.time()
    for text in tqdm(samples):
        _ = new_phonemizer(text)
    end_new = time.time()
    time_new = end_new - start_new

    # 5. Results
    print("\n--- Results ---")
    print(f"EspeakPhonemizer (Old): {time_old:.4f} seconds ({time_old/len(samples):.4f} s/sentence)")
    print(f"PhonemizerWrapper (New): {time_new:.4f} seconds ({time_new/len(samples):.4f} s/sentence)")
    
    if time_new > 0:
        speedup = time_old / time_new
        print(f"Speedup: {speedup:.2f}x")
    
    print("\n--- Output Comparison (First Sample) ---")
    print(f"Text: {samples[0]}")
    print(f"Old: {old_phonemizer(samples[0])}")
    print(f"New: {new_phonemizer(samples[0])}")

if __name__ == "__main__":
    benchmark()