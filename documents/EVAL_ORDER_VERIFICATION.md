# Evaluation Order Verification

## Summary

**✓ VERIFIED**: The index `i` in `compute_metrics` corresponds to `eval_dataset[i]` - predictions are collected in the same order as the dataset.

## Verification Details

A verification script (`verify_eval_order.py`) was created to test that:
1. The DataLoader processes samples in sequential order
2. Predictions are concatenated batch-by-batch in order
3. The index `i` in `compute_metrics` matches `eval_dataset[i]`

The verification **PASSED** - indices match expected order.

## How Order is Preserved

1. **DataLoader Sampler**: The Trainer uses `SequentialSampler` for evaluation (no shuffling)
   - See `trainer.py` line 1188-1189: `return SequentialSampler(eval_dataset)` when `world_size <= 1`

2. **Batch Processing**: Batches are processed sequentially in `prediction_loop`
   - See `trainer.py` line 5298: `for step, inputs in enumerate(dataloader):`

3. **Prediction Concatenation**: Predictions are concatenated sequentially using `nested_concat`
   - See `trainer.py` line 5309: `preds_host = logits if preds_host is None else nested_concat(preds_host, logits, ...)`

## Assumptions & Edge Cases

The order preservation relies on these assumptions:

1. **`group_by_length=False`** (default) - If enabled, `LengthGroupedSampler` groups by length, which could change order
2. **Single-process training** (`world_size=1`) - Distributed training uses different samplers
3. **No custom sampler** - If a custom sampler is provided, it must maintain order
4. **Sequential batch processing** - Predictions are concatenated in the order batches are processed

## Current Configuration

From `train.py`:
- No `group_by_length` set (defaults to `False`)
- No distributed training settings visible
- Uses default Trainer evaluation behavior

## Conclusion

**The current implementation is correct** - `eval_dataset[i]` corresponds to the `i`-th prediction in `compute_metrics`.

However, if any of the assumptions above change (e.g., enabling `group_by_length` or distributed training), the order may not be preserved and the code would need to be updated accordingly.

