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
def gather_token_logprobs(log_probs, token_ids):
    # TODO: Extract the log-probability of each observed token from a full vocab log-prob tensor...
    expand = token_ids[:,:,np.newaxis]
    return np.take_along_axis(log_probs, expand, axis=-1).squeeze(-1)

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

# Step 9 - bradley_terry_loss (not yet solved)
# TODO: implement

# Step 10 - reward_accuracy (not yet solved)
# TODO: implement

# Step 11 - build_preference_pairs (not yet solved)
# TODO: implement

# Step 12 - sample_preference_batch (not yet solved)
# TODO: implement

# Step 13 - freeze_reference_logprobs (not yet solved)
# TODO: implement

# Step 14 - policy_reference_logratio (not yet solved)
# TODO: implement

# Step 15 - dpo_pair_margin (not yet solved)
# TODO: implement

# Step 16 - dpo_loss (not yet solved)
# TODO: implement

# Step 17 - dpo_loss_grad (not yet solved)
# TODO: implement

# Step 18 - dpo_train_step (not yet solved)
# TODO: implement

# Step 19 - train_dpo (not yet solved)
# TODO: implement

# Step 20 - length_normalized_logprob (not yet solved)
# TODO: implement

# Step 21 - ipo_loss (not yet solved)
# TODO: implement

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

