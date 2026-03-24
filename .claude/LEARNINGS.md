# TabICL Learnings

## Architecture
- Three-stage pipeline: ColEmbedding (TF_col) → RowInteraction (TF_row) → ICLearning (TF_icl)
- ICL dim = embed_dim (128) × row_num_cls (4) = 512
- QASSMax uses `n = k.size(-2)` (source seq length) for log(n) scaling
- Train/test separation: `k = v = q[..., :train_size, :]` in MultiheadAttentionBlock

## Device Handling
- **Friction point**: ColEmbedding and RowInteraction use InferenceManager in eval mode, which auto-moves tensors to CUDA. When wrapping the model (e.g., for fine-tuning), use `.train()` on these submodules to bypass InferenceManager and keep device handling explicit.
- The training codepath is simpler and doesn't involve InferenceManager.

## Model Initialization
- `MultiheadAttentionBlock.init_weights()` zeros out `out_proj` and `linear2` weights, making each block an identity mapping initially. This means randomly initialized models won't show gradient flow through attention. Only pretrained models with learned non-zero weights will propagate gradients through the attention mechanism.

## Checkpoint Format
- `checkpoint["config"]` contains TabICL constructor kwargs
- `checkpoint["state_dict"]` contains model weights
- Checkpoints on HuggingFace Hub: `jingang/TabICL`
- Classifier: `tabicl-classifier-v2-20260212.ckpt`
- Regressor: `tabicl-regressor-v2-20260212.ckpt`

## Testing
- Tests at `tests/test_sklearn.py`, run with `pytest tests/ -x -q`
- 105 tests + 2 skipped as of 2026-02-23
