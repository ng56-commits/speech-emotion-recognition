from src.data.loader import load_all_datasets
_, labels, _ = load_all_datasets("data/raw/ravdess", "data/raw/tess")
print(sorted(set(labels)))