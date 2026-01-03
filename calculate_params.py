"""
Calculate GPT-2 Medium parameters breakdown
GPT-2 Medium config: n_layer=24, n_head=16, n_embd=1024
"""

def calculate_gpt2_medium_params():
    # GPT-2 Medium configuration
    vocab_size = 50257  # GPT-2 vocab size
    block_size = 1024   # context length
    n_layer = 24        # number of transformer blocks
    n_head = 16         # number of attention heads
    n_embd = 1024       # embedding dimension
    bias = False        # whether to use bias (False in your config)
    
    # Feed-forward network size (always 4x embedding size in GPT)
    ffw_size = 4 * n_embd  # 4096
    
    print("=" * 70)
    print("GPT-2 Medium Parameter Calculation")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  n_layer (L) = {n_layer}")
    print(f"  n_head (H) = {n_head}")
    print(f"  n_embd (d_model) = {n_embd}")
    print(f"  vocab_size = {vocab_size}")
    print(f"  block_size = {block_size}")
    print(f"  bias = {bias}")
    print(f"  ffw_size = 4 * n_embd = {ffw_size}")
    print()
    
    # ========== EMBEDDINGS ==========
    print("1. EMBEDDINGS:")
    # Token embeddings (shared with output layer via weight tying)
    token_emb = n_embd * vocab_size
    print(f"   Token embeddings (wte): {n_embd} × {vocab_size:,} = {token_emb:,}")
    
    # Position embeddings
    pos_emb = n_embd * block_size
    print(f"   Position embeddings (wpe): {n_embd} × {block_size:,} = {pos_emb:,}")
    
    # Note: Position embeddings are excluded from "non-embedding" count
    # Token embeddings are included because they're shared with output layer
    print(f"   Total embeddings: {token_emb + pos_emb:,}")
    print()
    
    # ========== TRANSFORMER BLOCKS (repeated n_layer times) ==========
    print(f"2. TRANSFORMER BLOCKS (× {n_layer} layers):")
    
    # Per-block components:
    
    # LayerNorm 1 (before attention)
    ln1_params = n_embd if bias else n_embd  # weight only if no bias
    print(f"   LayerNorm 1: {ln1_params:,} params")
    
    # Attention module
    # Q, K, V projections: 3 × (n_embd × n_embd) weights
    # + 3 × n_embd biases (if bias=True)
    attn_qkv_weights = 3 * n_embd * n_embd
    attn_qkv_bias = 3 * n_embd if bias else 0
    attn_qkv = attn_qkv_weights + attn_qkv_bias
    print(f"   Attention QKV (c_attn): {attn_qkv_weights:,} weights + {attn_qkv_bias:,} bias = {attn_qkv:,}")
    
    # Attention output projection
    attn_proj_weights = n_embd * n_embd
    attn_proj_bias = n_embd if bias else 0
    attn_proj = attn_proj_weights + attn_proj_bias
    print(f"   Attention output (c_proj): {attn_proj_weights:,} weights + {attn_proj_bias:,} bias = {attn_proj:,}")
    
    attn_total = attn_qkv + attn_proj
    print(f"   Total attention: {attn_total:,}")
    
    # LayerNorm 2 (before MLP)
    ln2_params = n_embd if bias else n_embd
    print(f"   LayerNorm 2: {ln2_params:,} params")
    
    # MLP (Feed-Forward Network)
    # First linear: n_embd → ffw_size (4*n_embd)
    mlp_fc_weights = n_embd * ffw_size
    mlp_fc_bias = ffw_size if bias else 0
    mlp_fc = mlp_fc_weights + mlp_fc_bias
    print(f"   MLP first layer (c_fc): {mlp_fc_weights:,} weights + {mlp_fc_bias:,} bias = {mlp_fc:,}")
    
    # Second linear: ffw_size → n_embd
    mlp_proj_weights = ffw_size * n_embd
    mlp_proj_bias = n_embd if bias else 0
    mlp_proj = mlp_proj_weights + mlp_proj_bias
    print(f"   MLP second layer (c_proj): {mlp_proj_weights:,} weights + {mlp_proj_bias:,} bias = {mlp_proj:,}")
    
    mlp_total = mlp_fc + mlp_proj
    print(f"   Total MLP: {mlp_total:,}")
    
    # Total per block
    params_per_block = ln1_params + attn_total + ln2_params + mlp_total
    print(f"   Total per block: {params_per_block:,}")
    
    # All blocks
    transformer_params = n_layer * params_per_block
    print(f"   Total for all {n_layer} blocks: {transformer_params:,}")
    print()
    
    # ========== FINAL LAYERS ==========
    print("3. FINAL LAYERS:")
    # Final LayerNorm
    ln_f_params = n_embd if bias else n_embd
    print(f"   Final LayerNorm (ln_f): {ln_f_params:,}")
    
    # Output head (lm_head) - weight tied with token embeddings, so 0 additional params
    lm_head_params = 0  # weight tying: uses wte weights
    print(f"   Output head (lm_head): {lm_head_params:,} (weight-tied with token embeddings)")
    print()
    
    # ========== TOTAL ==========
    print("=" * 70)
    print("TOTAL PARAMETERS:")
    
    # Method 1: Including all embeddings
    total_with_all_emb = token_emb + pos_emb + transformer_params + ln_f_params + lm_head_params
    print(f"   Including all embeddings: {total_with_all_emb:,} ({total_with_all_emb/1e6:.2f}M)")
    
    # Method 2: Non-embedding count (standard, excludes position embeddings)
    # Token embeddings are included because they're shared with output layer
    total_non_embedding = token_emb + transformer_params + ln_f_params + lm_head_params
    print(f"   Non-embedding count (standard): {total_non_embedding:,} ({total_non_embedding/1e6:.2f}M)")
    
    print()
    print("Expected GPT-2 Medium: ~350M parameters")
    print("=" * 70)
    
    return total_non_embedding

if __name__ == "__main__":
    params = calculate_gpt2_medium_params()
    print(f"\nCalculated: {params/1e6:.2f}M parameters")

