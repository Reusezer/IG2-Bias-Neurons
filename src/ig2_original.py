"""
Original IG² Implementation for Bias Neuron Identification

This implements the IG² algorithm from "The Devil is in the Neurons" paper.
Formula: IG²(w, d) = w̄ · (1/m) · Σₖ ∂P(d | α·w̄)/∂w |_{α=k/m}

For comparison with SignedIG² in proanti-SignedIG2.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
from tqdm import tqdm


def scaled_input(
    activation: torch.Tensor,
    num_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create scaled inputs for Riemann sum approximation.
    Interpolates from zeros (baseline) to original activation.

    Args:
        activation: Original activation value [intermediate_size]
        num_steps: Number of interpolation steps (m)

    Returns:
        scaled: Interpolated values [num_steps, intermediate_size]
        step: Step size for Riemann sum [intermediate_size]
    """
    baseline = torch.zeros_like(activation)
    step = (activation - baseline) / num_steps
    scaled = torch.stack([baseline + step * k for k in range(1, num_steps + 1)], dim=0)
    return scaled, step


class OriginalIG2:
    """
    Computes Original IG² scores for bias neuron identification.

    Unlike SignedIG², this computes:
    - IG²(w, d1): Attribution for demographic d1
    - IG²(w, d2): Attribution for demographic d2
    - IG²_gap = IG²(d1) - IG²(d2): Gap score

    Positive gap → neuron biased toward d1
    Negative gap → neuron biased toward d2
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        num_steps: int = 50,
    ):
        """
        Args:
            model: BERT model (BertForMaskedLM)
            tokenizer: BERT tokenizer
            device: Device to use
            num_steps: Number of Riemann sum steps (default: 50)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_steps = num_steps

        # Get model architecture info
        if hasattr(model, 'bert'):
            self.base_model = model.bert
            self.encoder = model.bert.encoder
            self.is_roberta = False
        elif hasattr(model, 'roberta'):
            self.base_model = model.roberta
            self.encoder = model.roberta.encoder
            self.is_roberta = True
        else:
            raise ValueError("Model must have 'bert' or 'roberta' encoder")

        self.num_layers = len(self.encoder.layer)
        self.intermediate_size = self.encoder.layer[0].intermediate.dense.out_features

        self.model.eval()
        self.model.to(device)

    def _get_mask_position(self, input_ids: torch.Tensor) -> int:
        """Find the [MASK] token position in input."""
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[1]
        if len(mask_positions) == 0:
            raise ValueError("No [MASK] token found in input")
        return mask_positions[0].item()

    def _get_prediction_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get prediction scores from hidden states."""
        if self.is_roberta:
            return self.model.lm_head(hidden_states)
        else:
            return self.model.cls(hidden_states)

    def compute_ig2_for_layer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_label_id: int,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Compute IG² scores for all neurons in a layer for a specific target label.

        This manually runs through transformer layers to properly maintain
        the computation graph for gradient computation.

        Args:
            input_ids: Input token IDs [1, seq_len]
            attention_mask: Attention mask [1, seq_len]
            target_label_id: Token ID of target demographic word
            layer_idx: Layer index to compute IG² for

        Returns:
            ig2_scores: IG² scores for each neuron [intermediate_size]
        """
        mask_pos = self._get_mask_position(input_ids)

        # Forward pass up to target layer (no grad needed)
        with torch.no_grad():
            embeddings = self.base_model.embeddings(input_ids)
            hidden_states = embeddings

            # Create attention mask for transformer layers
            extended_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            extended_mask = (1.0 - extended_mask) * -10000.0

            # Run through layers up to target layer
            for i in range(layer_idx):
                layer_output = self.encoder.layer[i](hidden_states, extended_mask)
                hidden_states = layer_output[0]

            # Get attention output for target layer
            attn_output = self.encoder.layer[layer_idx].attention(hidden_states, extended_mask)[0]

        # Get original intermediate activations at mask position
        intermediate_module = self.encoder.layer[layer_idx].intermediate
        with torch.no_grad():
            original_intermediate = intermediate_module(attn_output)
            # Shape: [1, seq_len, intermediate_size]
            original_acts_at_mask = original_intermediate[0, mask_pos, :].clone()

        # Create scaled activations for Riemann sum
        # Shape: [num_steps, intermediate_size]
        scaled_acts, step_sizes = scaled_input(original_acts_at_mask, self.num_steps)

        # Accumulate gradients for each step
        grad_sum = torch.zeros(self.intermediate_size, device=self.device)

        for k in range(self.num_steps):
            # Create input that requires grad for gradient computation
            attn_output_grad = attn_output.clone().detach().requires_grad_(True)

            # Forward through intermediate
            intermediate_out = intermediate_module(attn_output_grad)

            # Replace activations at mask position with scaled values
            modified_intermediate = intermediate_out.clone()
            modified_intermediate[0, mask_pos, :] = scaled_acts[k]

            # Forward through output layer of this transformer block
            output_module = self.encoder.layer[layer_idx].output
            layer_output = output_module(modified_intermediate, attn_output_grad)

            # Forward through remaining layers
            hidden_states_rest = layer_output
            for i in range(layer_idx + 1, self.num_layers):
                layer_out = self.encoder.layer[i](hidden_states_rest, extended_mask)
                hidden_states_rest = layer_out[0]

            # Get prediction scores
            prediction_scores = self._get_prediction_scores(hidden_states_rest)
            logits_at_mask = prediction_scores[0, mask_pos, :]
            probs = F.softmax(logits_at_mask, dim=-1)

            # Target probability
            target_prob = probs[target_label_id]

            # Compute gradients w.r.t. intermediate input
            grad = torch.autograd.grad(
                target_prob,
                attn_output_grad,
                retain_graph=False,
                create_graph=False,
            )[0]

            # Sum gradients at mask position across hidden dimension
            # This gives us the gradient contribution to intermediate neurons
            grad_at_mask = grad[0, mask_pos, :]
            grad_sum += grad_at_mask

        # Compute final IG² score: w̄ · (1/m) · Σ gradients
        # step_sizes = w̄/m, so IG² = step_sizes * grad_sum
        ig2_scores = step_sizes * grad_sum

        return ig2_scores

    def compute_ig2_gap(
        self,
        sentence: str,
        d1_word: str,
        d2_word: str,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute IG² gap scores for all layers.

        Args:
            sentence: Input sentence with [MASK] token
            d1_word: First demographic word (e.g., "female")
            d2_word: Second demographic word (e.g., "male")
            verbose: Whether to print progress

        Returns:
            ig2_d1: IG² scores for d1 [num_layers, intermediate_size]
            ig2_d2: IG² scores for d2 [num_layers, intermediate_size]
            ig2_gap: Gap scores (d1 - d2) [num_layers, intermediate_size]
        """
        # Tokenize
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Get token IDs for demographic words
        d1_token_id = self.tokenizer.convert_tokens_to_ids(d1_word)
        d2_token_id = self.tokenizer.convert_tokens_to_ids(d2_word)

        if d1_token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Word '{d1_word}' is not in vocabulary")
        if d2_token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Word '{d2_word}' is not in vocabulary")

        # Initialize result arrays
        ig2_d1 = np.zeros((self.num_layers, self.intermediate_size))
        ig2_d2 = np.zeros((self.num_layers, self.intermediate_size))

        # Compute IG² for each layer
        layer_iter = range(self.num_layers)
        if verbose:
            layer_iter = tqdm(layer_iter, desc="Layers")

        for layer_idx in layer_iter:
            # IG² for d1
            ig2_d1_layer = self.compute_ig2_for_layer(
                input_ids, attention_mask, d1_token_id, layer_idx
            )
            ig2_d1[layer_idx] = ig2_d1_layer.cpu().detach().numpy()

            # IG² for d2
            ig2_d2_layer = self.compute_ig2_for_layer(
                input_ids, attention_mask, d2_token_id, layer_idx
            )
            ig2_d2[layer_idx] = ig2_d2_layer.cpu().detach().numpy()

        # Compute gap
        ig2_gap = ig2_d1 - ig2_d2

        return ig2_d1, ig2_d2, ig2_gap

    def compute_ig2_gap_batch(
        self,
        samples: List[dict],
        d1_key: str = "d1",
        d2_key: str = "d2",
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute average IG² gap scores over multiple samples.

        Args:
            samples: List of dicts with 'sentence', d1_key, d2_key
            d1_key: Key for d1 word in sample dict
            d2_key: Key for d2 word in sample dict
            verbose: Whether to print progress

        Returns:
            avg_ig2_d1: Average IG² scores for d1 [num_layers, intermediate_size]
            avg_ig2_d2: Average IG² scores for d2 [num_layers, intermediate_size]
            avg_ig2_gap: Average gap scores [num_layers, intermediate_size]
        """
        total_d1 = np.zeros((self.num_layers, self.intermediate_size))
        total_d2 = np.zeros((self.num_layers, self.intermediate_size))

        sample_iter = samples
        if verbose:
            sample_iter = tqdm(samples, desc="Samples")

        for sample in sample_iter:
            sentence = sample["sentence"]
            d1_word = sample[d1_key]
            d2_word = sample[d2_key]

            ig2_d1, ig2_d2, _ = self.compute_ig2_gap(
                sentence, d1_word, d2_word, verbose=False
            )

            total_d1 += ig2_d1
            total_d2 += ig2_d2

        n_samples = len(samples)
        avg_d1 = total_d1 / n_samples
        avg_d2 = total_d2 / n_samples
        avg_gap = avg_d1 - avg_d2

        return avg_d1, avg_d2, avg_gap


def extract_bias_neurons(
    ig2_gap: np.ndarray,
    threshold: float = 0.2,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Extract bias neurons based on IG² gap scores.

    Args:
        ig2_gap: Gap scores [num_layers, intermediate_size]
        threshold: Selection threshold (fraction of max score)

    Returns:
        bias_toward_d1: List of (layer, neuron) tuples with positive gap
        bias_toward_d2: List of (layer, neuron) tuples with negative gap
    """
    max_abs = np.abs(ig2_gap).max()
    threshold_value = max_abs * threshold

    bias_toward_d1 = []  # Positive gap → biased toward d1
    bias_toward_d2 = []  # Negative gap → biased toward d2

    for layer_idx in range(ig2_gap.shape[0]):
        for neuron_idx in range(ig2_gap.shape[1]):
            score = ig2_gap[layer_idx, neuron_idx]
            if score >= threshold_value:
                bias_toward_d1.append((layer_idx, neuron_idx))
            elif score <= -threshold_value:
                bias_toward_d2.append((layer_idx, neuron_idx))

    return bias_toward_d1, bias_toward_d2
