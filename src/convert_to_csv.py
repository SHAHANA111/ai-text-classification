import pandas as pd

# 1️⃣ Load raw dataset
raw_path = "../data/raw/SMSSpamCollection"
df = pd.read_csv(raw_path, sep='\t', header=None, names=["label", "text"], encoding='latin-1')

# 2️⃣ Convert labels to numbers
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 3️⃣ Save as processed CSV
processed_path = "../data/processed/spam.csv"
df.to_csv(processed_path, index=False)

print(f"Conversion done! Processed file saved at: {processed_path}")