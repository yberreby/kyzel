from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def plot_token_distribution(tokenizer, dataset):
    # Analyze token lengths
    lengths = []
    for sample in tqdm(dataset):
        tokens = tokenizer(sample["text"], return_tensors="pt", truncation=False)
        lengths.append(len(tokens.input_ids[0]))

    # Visualize distribution
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, edgecolor='black')
    plt.title('Distribution of Sample Lengths (in tokens)')
    plt.xlabel('Length in Tokens')
    plt.ylabel('Count')

    plt.grid(True, alpha=0.3)
    plt.show()

    # Add stats
    stats = f"""
    Token count distribution across sessions:
      Mean: {np.mean(lengths):.1f}
      Median: {np.median(lengths):.1f}
      Max: {np.max(lengths)}
      95th %ile: {np.percentile(lengths, 95):.1f}
    """
    print(stats)
