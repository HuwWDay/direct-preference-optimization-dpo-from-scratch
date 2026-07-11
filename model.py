"""
Direct Preference Optimization (DPO) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - log_softmax
import numpy as np

def log_softmax(logits, axis=-1):
    # TODO: convert logits into numerically stable log-probabilities along axis
    m = np.max(logits, axis=axis, keepdims=True)
    shift = logits - m 
    return shift - np.log(np.sum(np.exp(shift), axis=-1, keepdims=True))

# Step 2 - softmax
def softmax(logits, axis=-1):
    # TODO: Convert an array of logits into a probability distribution along a given axis
    m = np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits - m)
    return exp / np.sum(exp, axis=axis, keepdims=True)

# Step 3 - gather_token_logprobs
import numpy as np

def gather_token_logprobs(log_probs, token_ids):
    """
    Extracts the log-probability of each observed token from a full vocab log-prob tensor.
    Handles both 1D arrays of shape (T,) and 2D batched arrays of shape (B, T).
    """
    # Use Ellipsis (...) to automatically grab all leading dimensions (works for both 1D and 2D)
    expand = token_ids[..., np.newaxis]
    
    # Extract the target values along the vocabulary axis (-1)
    gathered = np.take_along_axis(log_probs, expand, axis=-1)
    
    # Remove the trailing singleton vocabulary axis we added for take_along_axis
    return gathered.squeeze(-1)

# Step 4 - masked_sequence_logprob
def masked_sequence_logprob(token_logprobs, mask):
    # TODO: Sum per-token log-probabilities under a binary mask to obtain a single sequence log-probability per example.
    return np.sum(token_logprobs*mask, axis=-1)

# Step 5 - init_policy_params
def init_policy_params(vocab_size, d_model, rng=None):
    # TODO: Initialize the policy language-model parameters with small random values
    if rng is None:
        rng = np.random.default_rng()
    return {"embed":rng.normal(loc=0.0, scale=0.02, size=(vocab_size, d_model)),
    "W_out":rng.normal(loc=0.0, scale=0.02, size=(d_model, vocab_size)), 
    "b_out":np.zeros(vocab_size)}

# Step 6 - policy_token_logits
def policy_token_logits(params, token_ids):
    # TODO: Compute next-token logits for every position from policy params and token ids.
    embed, W_out, b_out = params["embed"], params["W_out"], params["b_out"]
    return embed[token_ids] @ W_out + b_out

# Step 7 - policy_sequence_logprob
def policy_sequence_logprob(params, token_ids, mask):
    # TODO: Compute the total masked sequence log-probability under the current policy...
    logits = policy_token_logits(params, token_ids)
    log_probs = log_softmax(logits, axis=-1)
    token_logprobs = gather_token_logprobs(log_probs, token_ids)
    return masked_sequence_logprob(token_logprobs, mask)

# Step 8 - sequence_logprob_grad
def sequence_logprob_grad(params, token_ids, mask):
    # TODO: Compute gradients of the summed sequence log-probability w.r.t. params
    
    embed = params['embed']
    W_out = params['W_out']
    b_out = params['b_out']
    
    B, T = token_ids.shape
    V = b_out.shape[0]
    D = embed.shape[1]
    
    probs = softmax(policy_token_logits(params, token_ids))
    # 2. Gradient w.r.t Logits
    # Initialize with -P(j)
    d_logits = -probs
    # Add 1.0 for the actual target token positions
    # Using advanced indexing to target the exact token positions
    batch_idx = np.arange(B)[:, None]
    time_idx = np.arange(T)
    d_logits[batch_idx, time_idx, token_ids] += 1.0
    
    # Apply the sequence mask: shape (B, T, V)
    d_logits *= mask[:, :, None]
    
    # 3. Gradients w.r.t Output Layer Parameters
    # db_out sums over batch (0) and time (1) dimensions
    db_out = np.sum(d_logits, axis=(0, 1))
    h = embed[token_ids]
    # dW_out is computed by reshaping to matrix dimensions and performing a dot product
    h_flat = h.reshape(-1, D)
    d_logits_flat = d_logits.reshape(-1, V)
    dW_out = np.dot(h_flat.T, d_logits_flat)
    
    # 4. Gradient w.r.t Hidden States (Embeddings per token)
    dh = np.dot(d_logits, W_out.T) # shape (B, T, D)
    
    # 5. Gradient w.r.t the Embedding Matrix
    dembed = np.zeros_like(embed)
    # Accumulate gradients into the embedding rows matching token_ids
    np.add.at(dembed, token_ids, dh)
    
    return {
        'embed': dembed,
        'W_out': dW_out,
        'b_out': db_out
    }

# Step 9 - bradley_terry_loss
def bradley_terry_loss(reward_chosen, reward_rejected):
    # TODO: Compute the mean Bradley-Terry pairwise preference loss...
    margins = reward_chosen - reward_rejected
    return np.mean(np.logaddexp(0.0, -margins))

# Step 10 - reward_accuracy
def reward_accuracy(reward_chosen, reward_rejected):
    # TODO: Fraction of pairs where chosen reward is strictly higher than rejected.
    return np.mean((reward_chosen > reward_rejected))

# Step 11 - build_preference_pairs
def build_preference_pairs(prompts, chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    # TODO: Package raw arrays into a list of preference-pair dictionaries
    out = []
    for i in range(len(prompts)):
        temp = {"prompt":prompts[i],"chosen_ids":chosen_ids[i], "rejected_ids": rejected_ids[i],
        "chosen_mask":chosen_mask[i], "rejected_mask":rejected_mask[i]}
        out.append(temp)
    return out

# Step 12 - sample_preference_batch
import numpy as np

def sample_preference_batch(pairs, batch_size, rng=None):
    """
    Samples a mini-batch of preference pairs and stacks their fields.
    
    Args:
        pairs: list of dicts, where each dict contains:
               'chosen_ids', 'rejected_ids', 'chosen_mask', 'rejected_mask'
               and optionally 'prompt'.
        batch_size: int, number of pairs to sample.
        rng: optional np.random.Generator instance.
        
    Returns:
        batch_dict: dict of stacked NumPy arrays with a leading batch dimension.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    n = len(pairs)
    # Choose with replacement only if requested batch_size exceeds available dataset size
    replace = batch_size > n
    choices = rng.choice(n, size=batch_size, replace=replace)
    
    # Gather the sampled dictionary items
    sampled_pairs = [pairs[idx] for idx in choices]
    
    # Stack the core sequence tracking matrices
    batch = {
        'chosen_ids': np.stack([p['chosen_ids'] for p in sampled_pairs], axis=0),
        'rejected_ids': np.stack([p['rejected_ids'] for p in sampled_pairs], axis=0),
        'chosen_mask': np.stack([p['chosen_mask'] for p in sampled_pairs], axis=0),
        'rejected_mask': np.stack([p['rejected_mask'] for p in sampled_pairs], axis=0)
    }
    
    # Check if 'prompt' exists in the first item to safely include it
    if 'prompt' in sampled_pairs[0]:
        # np.stack handles prompt text/token arrays cleanly
        batch['prompt'] = np.stack([p['prompt'] for p in sampled_pairs], axis=0)
        
    return batch

# Step 13 - freeze_reference_logprobs
import numpy as np

def freeze_reference_logprobs(ref_params, pairs):
    """
    Precomputes and freezes reference-model sequence log-probabilities 
    for every chosen and rejected response in a preference dataset.
    
    Args:
        ref_params: dict containing the frozen reference model parameter tensors.
        pairs: list of dicts, where each dict contains:
               'chosen_ids', 'rejected_ids', 'chosen_mask', 'rejected_mask'
               
    Returns:
        frozen_logprobs: list of dicts strictly aligned with `pairs`, each containing:
                         'chosen': scalar log-prob of the chosen response
                         'rejected': scalar log-prob of the rejected response
    """
    frozen_logprobs = []
    
    for pair in pairs:
        # Compute reference log-probability for the chosen sequence
        ref_chosen_lp = policy_sequence_logprob(
            ref_params, 
            pair['chosen_ids'], 
            pair['chosen_mask']
        )
        
        # Compute reference log-probability for the rejected sequence
        ref_rejected_lp = policy_sequence_logprob(
            ref_params, 
            pair['rejected_ids'], 
            pair['rejected_mask']
        )
        
        # Safely convert to ordinary Python floats to avoid dimension mismatches downstream
        chosen_scalar = float(np.asarray(ref_chosen_lp).reshape(-1)[0])
        rejected_scalar = float(np.asarray(ref_rejected_lp).reshape(-1)[0])
        
        # Append maintaining exact alignment and the target key schema ('chosen', 'rejected')
        frozen_logprobs.append({
            'chosen': chosen_scalar,
            'rejected': rejected_scalar
        })
        
    return frozen_logprobs

# Step 14 - policy_reference_logratio
def policy_reference_logratio(policy_logprob, reference_logprob):
    # TODO: Compute the per-sequence log-ratio log pi_theta(y) - log pi_ref(y)
    return policy_logprob - reference_logprob

# Step 15 - dpo_pair_margin
def dpo_pair_margin(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # TODO: Compute the scaled DPO pair margin for a batch of preference pairs
    return beta*(policy_logprob_chosen-ref_logprob_chosen-policy_logprob_rejected+ref_logprob_rejected)

# Step 16 - dpo_loss
def dpo_loss(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # TODO: return the mean logistic loss on the DPO pair margins as a scalar float
    margin = dpo_pair_margin(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta) 
    return float(np.mean(np.logaddexp(0.0, -margin)))

# Step 17 - dpo_loss_grad
import numpy as np

def dpo_loss_grad(params, batch, ref_logprobs_batch, beta):
    """
    Evaluates the DPO logistic loss on one preference batch and returns 
    parameter gradients for the policy.
    
    Args:
        params: dict of policy parameters ('embed', 'W_out', 'b_out')
        batch: dict containing 'chosen_ids', 'rejected_ids', 'chosen_mask', 'rejected_mask' arrays of shape (B, T)
        ref_logprobs_batch: dict containing 'chosen' and 'rejected' reference log-probs of shape (B,)
        beta: float, the KL regularization temperature
        
    Returns:
        loss: ordinary Python float of the average DPO loss
        grads: dict containing accumulated parameter gradients matching the shapes of params
    """
    # 1. Extract inputs
    chosen_ids = batch['chosen_ids']
    chosen_mask = batch['chosen_mask']
    rejected_ids = batch['rejected_ids']
    rejected_mask = batch['rejected_mask']
    
    ref_c = ref_logprobs_batch['chosen']
    ref_r = ref_logprobs_batch['rejected']
    
    B = chosen_ids.shape[0]
    
    # 2. Forward pass to get policy sequence log-probabilities
    lp_c = policy_sequence_logprob(params, chosen_ids, chosen_mask)
    lp_r = policy_sequence_logprob(params, rejected_ids, rejected_mask)
    
    # 3. Compute DPO margins and loss
    margins = dpo_pair_margin(lp_c, lp_r, ref_c, ref_r, beta)
    # Average softplus loss over the batch: log(1 + exp(-m))
    loss = np.mean(np.logaddexp(0.0, -margins))
    
    # 4. Initialize zero-filled gradients dictionary
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    
    # 5. Compute per-example weights and accumulate gradients
    # Weight formula: w_i = -sigmoid(-m_i) * beta / B
    # Note: 1 / (1 + np.exp(margins)) is mathematically equivalent to sigmoid(-margins)
    sig_neg_m = 1.0 / (1.0 + np.exp(margins))
    weights = -sig_neg_m * beta / B
    
    for i in range(B):
        w_i = weights[i]
        
        # Keep the 2D layout with length-1 batch slices [i:i+1]
        g_chosen = sequence_logprob_grad(params, chosen_ids[i:i+1], chosen_mask[i:i+1])
        g_rejected = sequence_logprob_grad(params, rejected_ids[i:i+1], rejected_mask[i:i+1])
        
        # Accumulate w_i * (g_chosen - g_rejected) into the tracking grads dict
        for k in grads.keys():
            grads[k] += w_i * (g_chosen[k] - g_rejected[k])
            
    return float(loss), grads

# Step 18 - dpo_train_step
import numpy as np

def dpo_train_step(params, batch, ref_logprobs_batch, beta, learning_rate):
    """
    Executes one gradient-descent update of the policy under the DPO objective.
    
    Args:
        params: dict of policy parameters ('embed', 'W_out', 'b_out')
        batch: dict containing 'chosen_ids', 'rejected_ids', 'chosen_mask', 'rejected_mask'
        ref_logprobs_batch: dict containing reference log-probs ('chosen', 'rejected')
        beta: float, the KL regularization temperature
        learning_rate: float, SGD step size
        
    Returns:
        updated_params: dict containing the new policy parameters after the SGD step
        metrics: dict containing scalar evaluation tracking metrics (e.g., {'loss': loss})
    """
    # 1. Compute the DPO loss and parameter gradients
    loss, grads = dpo_loss_grad(params, batch, ref_logprobs_batch, beta)
    
    # 2. Apply a plain SGD update rule without mutating the original dictionary
    updated_params = {}
    for key in params.keys():
        updated_params[key] = params[key] - learning_rate * grads[key]
        
    # 3. Package metrics tracking structure
    metrics = {
        'loss': float(loss)
    }
    
    return updated_params, metrics

# Step 19 - train_dpo
import numpy as np

def train_dpo(params, pairs, ref_logprobs, beta, learning_rate, num_steps, batch_size, rng=None):
    """
    Executes the full DPO training loop over a given number of optimization steps.
    
    Args:
        params: dict of initial policy parameters ('embed', 'W_out', 'b_out')
        pairs: list of preference pair dicts
        ref_logprobs: dict of arrays containing all precomputed reference log-probs
        beta: float, the KL regularization temperature
        learning_rate: float, SGD step size
        num_steps: int, total training updates to run
        batch_size: int, number of preference pairs per gradient step
        rng: optional np.random.Generator instance
        
    Returns:
        params: dict of optimized policy parameters
        history: list of dicts recording training progress metrics
    """
    if rng is None:
        rng = np.random.default_rng()
        
    n = len(pairs)
    replace = batch_size > n
    history = []
    
    for step in range(num_steps):
        # 1. Sample indices shared by pairs and reference log-probs
        indices = rng.choice(n, size=batch_size, replace=replace)
        
        # 2. Extract and stack sequence batch data for these indices
        sampled_pairs = [pairs[idx] for idx in indices]
        batch = {
            'chosen_ids': np.stack([p['chosen_ids'] for p in sampled_pairs], axis=0),
            'rejected_ids': np.stack([p['rejected_ids'] for p in sampled_pairs], axis=0),
            'chosen_mask': np.stack([p['chosen_mask'] for p in sampled_pairs], axis=0),
            'rejected_mask': np.stack([p['rejected_mask'] for p in sampled_pairs], axis=0)
        }
        
        # 3. Extract the aligned reference log-probs for the batch
        ref_logprobs_batch = {
            'chosen': ref_logprobs['chosen'][indices],
            'rejected': ref_logprobs['rejected'][indices]
        }
        
        # 4. Perform the DPO gradient update step
        params, metrics = dpo_train_step(params, batch, ref_logprobs_batch, beta, learning_rate)
        
        # 5. Record step metrics
        history.append({
            'step': step,
            'loss': float(metrics['loss'])
        })
        
    return params, history

# Step 20 - length_normalized_logprob
def length_normalized_logprob(seq_logprob, mask):
    # TODO: Normalize sequence log-probabilities by their valid token counts.
    return seq_logprob / np.sum(mask, axis=-1)

# Step 21 - ipo_loss
def ipo_loss(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # TODO: Evaluate mean squared IPO loss on unscaled log-ratio margins
    chosen = policy_reference_logratio(policy_logprob_chosen, ref_logprob_chosen)
    reject = policy_reference_logratio(policy_logprob_rejected, ref_logprob_rejected)
    h = (chosen - reject - 1 /(2.0 * beta))**2
    return np.mean(h, axis=0)

# Step 22 - implicit_reward (not yet solved)
# TODO: implement

# Step 23 - preference_accuracy (not yet solved)
# TODO: implement

# Step 24 - kl_to_reference (not yet solved)
# TODO: implement

# Step 25 - reward_margin_stats (not yet solved)
# TODO: implement

# Step 26 - evaluate_dpo (not yet solved)
# TODO: implement

# Step 27 - run_dpo_pipeline (not yet solved)
# TODO: implement

