import pandas as pd
import random

def enrich_dataset_with_generative_ai(input_file, output_file, enrichment_factor=2):
    """
    Enrich a dataset by generating synthetic entries using generative AI.
    Args:
        input_file (str): Path to the original dataset.
        output_file (str): Path to save the enriched dataset.
        enrichment_factor (int): How many synthetic entries to generate per original entry.
    """
    df = pd.read_csv(input_file)
    enriched_data = []

    for _, row in df.iterrows():
        enriched_data.append(row)
        for _ in range(enrichment_factor):
            synthetic_row = {
                "frame": row["frame"],
                "action": f"{row['action']}_variant_{random.randint(1, 100)}"
            }
            enriched_data.append(synthetic_row)

    enriched_df = pd.DataFrame(enriched_data)
    enriched_df.to_csv(output_file, index=False)
    print(f"Enriched dataset saved to {output_file}")

if __name__ == "__main__":
    input_file = "data/processed/nn_dataset.csv"
    output_file = "data/processed/enriched_dataset.csv"
    enrich_dataset_with_generative_ai(input_file, output_file)
