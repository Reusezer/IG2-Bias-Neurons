#!/usr/bin/env python3
"""
Run Original IG² analysis for bias neuron identification.

Usage:
    python scripts/run_ig2.py \
        --model bert-base-cased \
        --data_file data/gender_negative.json \
        --d1 female --d2 male \
        --output_dir results/bert-base-cased/gender_female_male
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ig2_original import OriginalIG2, extract_bias_neurons


def load_model(model_name: str, device: str = "cuda"):
    """Load BERT model and tokenizer."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    return model, tokenizer


def load_data(data_file: str) -> list:
    """Load data from JSON file."""
    with open(data_file, "r") as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description="Run Original IG² analysis")

    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-cased",
        help="Model name (bert-base-cased, bert-base-uncased, roberta-base)"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to data file (JSON format)"
    )
    parser.add_argument(
        "--d1",
        type=str,
        required=True,
        help="First demographic word (e.g., female)"
    )
    parser.add_argument(
        "--d2",
        type=str,
        required=True,
        help="Second demographic word (e.g., male)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of Riemann sum steps (default: 50)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Neuron selection threshold (default: 0.2)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    # Initialize IG² analyzer
    ig2 = OriginalIG2(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        num_steps=args.num_steps,
    )

    # Load data
    print(f"Loading data from: {args.data_file}")
    samples = load_data(args.data_file)

    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Using {len(samples)} samples")

    # Add d1/d2 to each sample
    for sample in samples:
        sample["d1"] = args.d1
        sample["d2"] = args.d2

    # Compute IG² scores
    print(f"\nComputing IG² scores...")
    print(f"  d1: {args.d1}")
    print(f"  d2: {args.d2}")
    print(f"  num_steps: {args.num_steps}")
    print()

    avg_ig2_d1, avg_ig2_d2, avg_ig2_gap = ig2.compute_ig2_gap_batch(
        samples,
        d1_key="d1",
        d2_key="d2",
        verbose=True,
    )

    # Save raw scores
    np.savez(
        output_dir / "ig2_scores.npz",
        ig2_d1=avg_ig2_d1,
        ig2_d2=avg_ig2_d2,
        ig2_gap=avg_ig2_gap,
        num_samples=len(samples),
    )
    print(f"\nSaved IG² scores to: {output_dir / 'ig2_scores.npz'}")

    # Extract bias neurons
    print(f"\nExtracting bias neurons (threshold: {args.threshold})...")
    bias_d1, bias_d2 = extract_bias_neurons(avg_ig2_gap, threshold=args.threshold)

    print(f"  Neurons biased toward {args.d1}: {len(bias_d1)}")
    print(f"  Neurons biased toward {args.d2}: {len(bias_d2)}")

    # Save bias neurons
    neurons_data = {
        "d1": args.d1,
        "d2": args.d2,
        "threshold": args.threshold,
        "bias_toward_d1": bias_d1,
        "bias_toward_d2": bias_d2,
        "num_bias_d1": len(bias_d1),
        "num_bias_d2": len(bias_d2),
    }

    with open(output_dir / "bias_neurons.json", "w") as f:
        json.dump(neurons_data, f, indent=2)
    print(f"Saved bias neurons to: {output_dir / 'bias_neurons.json'}")

    # Save metadata
    metadata = {
        "model": args.model,
        "d1": args.d1,
        "d2": args.d2,
        "num_samples": len(samples),
        "num_steps": args.num_steps,
        "threshold": args.threshold,
        "num_layers": ig2.num_layers,
        "intermediate_size": ig2.intermediate_size,
        "data_file": args.data_file,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {output_dir / 'metadata.json'}")

    # Print summary
    print("\n" + "=" * 50)
    print("IG² Analysis Complete!")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Samples: {len(samples)}")
    print(f"Layers: {ig2.num_layers}")
    print(f"Neurons per layer: {ig2.intermediate_size}")
    print(f"\nBias neurons found:")
    print(f"  Toward {args.d1}: {len(bias_d1)}")
    print(f"  Toward {args.d2}: {len(bias_d2)}")
    print(f"  Total: {len(bias_d1) + len(bias_d2)}")
    print()


if __name__ == "__main__":
    main()
