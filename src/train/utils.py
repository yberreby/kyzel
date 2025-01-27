from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Dict

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



def extract_metrics(history: List[Dict]) -> Dict[str, List[float]]:
    """Extract training and eval metrics from history."""
    metrics = {
        'train_loss': [],
        'eval_loss': [],
        'epochs': [],
        'eval_epochs': []
    }

    for entry in history:
        if 'loss' in entry:  # Training entry
            metrics['train_loss'].append(entry['loss'])
            metrics['epochs'].append(entry['epoch'])
        elif 'eval_loss' in entry:  # Eval entry
            metrics['eval_loss'].append(entry['eval_loss'])
            metrics['eval_epochs'].append(entry['epoch'])

    return metrics

def plot_training_loss(history: List[Dict], figsize=(10, 6)):
    """Plot training and evaluation loss over epochs."""
    metrics = extract_metrics(history)

    fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.header_visible = False

    # Plot training loss
    train_line, = ax.plot(
        metrics['epochs'],
        metrics['train_loss'],
        'b.-',
        label='Training Loss',
        markersize=8
    )

    # Plot eval loss
    eval_line, = ax.plot(
        metrics['eval_epochs'],
        metrics['eval_loss'],
        'r.-',
        label='Evaluation Loss',
        markersize=8
    )

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add some padding to the y-axis limits
    all_losses = metrics['train_loss'] + metrics['eval_loss']
    y_min, y_max = min(all_losses), max(all_losses)
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    plt.tight_layout()
    return fig, ax, (train_line, eval_line)
