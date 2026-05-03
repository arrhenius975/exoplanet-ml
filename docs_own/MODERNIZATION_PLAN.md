# AstroNet Modernization Plan - Detailed Analysis

## Overview
This document outlines a systematic approach to modernize the AstroNet package from TensorFlow 1.x APIs to TensorFlow 2.x, **preserving all functionality and structure**.

---

## Current State Assessment

### Deprecated APIs Found (15+ files)

**Entry Points (3 files):**
- `train.py`, `evaluate.py`, `predict.py` use `tf.app.run()` and `tf.logging`

**Data Pipeline (2 files):**
- `data/generate_input_records.py`: `tf.python_io.TFRecordWriter`, `tf.gfile.MakeDirs`, `tf.logging`
- `ops/dataset_ops.py`: `tf.gfile.Glob`, `tf.logging`

**Training Infrastructure (2 files):**
- `ops/training.py`: `tf.train.*` optimizers, `tf.contrib.opt`, `tf.contrib.training`, `tf.train.polynomial_decay`
- `util/estimator_util.py`: `tf.contrib.tpu.*` (still works but deprecated)

**Metrics (1 file):**
- `ops/metrics.py`: `tf.GraphKeys`, graph-mode variable management

**Models (4 files):**
- `astro_model/astro_model.py`: `tf.train.get_or_create_global_step()`
- `*_test.py` files use `tf.train.Scaffold()`

---

## 5-Phase Modernization Plan

### Phase 0: Entry Points (SAFE)
**Files:** `train.py`, `evaluate.py`, `predict.py` (3 files)  
**Changes:**
- Replace `tf.app.run()` → Direct `main()` function call
- Replace `tf.logging` → Python `logging` module
- Remove `FLAGS` global behavior from tf.app
- Keep CLI interface identical

**Risk:** None - pure refactoring  
**Testing:** Verify CLI args work identically

---

### Phase 1: File I/O & Data (SAFE)
**Files:** `data/generate_input_records.py`, `ops/dataset_ops.py` (2 files)  
**Changes:**
- `tf.python_io.TFRecordWriter` → `tf.io.TFRecordWriter`
- `tf.gfile.MakeDirs` → `pathlib.Path.mkdir(parents=True, exist_ok=True)`
- `tf.gfile.Glob` → `glob.glob()` or `pathlib.Path.glob()`
- `tf.logging.*` → Python `logging` module

**Risk:** Low - straightforward 1:1 replacements  
**Testing:** Generate TFRecords, verify format unchanged

---

### Phase 2: Training Infrastructure (CAREFUL)
**Files:** `ops/training.py`, `util/estimator_util.py` (2 files)  
**Changes:**

**In `ops/training.py`:**
- `tf.train.polynomial_decay` → `tf.keras.optimizers.schedules.PolynomialDecay`
- Optimizer classes:
  - `tf.train.MomentumOptimizer` → `tf.keras.optimizers.SGD(momentum=...)`
  - `tf.train.GradientDescentOptimizer` → `tf.keras.optimizers.SGD`
  - `tf.train.AdagradOptimizer` → `tf.keras.optimizers.Adagrad`
  - `tf.train.AdamOptimizer` → `tf.keras.optimizers.Adam`
  - `tf.RMSPropOptimizer` → `tf.keras.optimizers.RMSprop`
- Weight decay: Use optimizer's native weight_decay parameter instead of `tf.contrib.opt`
- Gradient clipping: Migrate from `tf.contrib.training.clip_gradient_norms_fn` to Keras equivalent

**In `util/estimator_util.py`:**
- Keep `tf.contrib.tpu.*` as-is (still functional in TF2)
- Verify EstimatorSpec.train_op works correctly

**Risk:** Medium - Learning rate schedule API differs, requires testing  
**Testing:** Train on test dataset, verify loss decreases properly

---

### Phase 3: Metrics (MODERATE)
**Files:** `ops/metrics.py` (1 file)  
**Changes:**
- Remove `tf.GraphKeys` usage
- Refactor `_metric_variable` to use explicit `tf.compat.v1.Variable` with collections
- Option: Migrate to `tf.keras.metrics` (larger change, optional)

**Risk:** Medium - Graph-mode metric evaluation is critical  
**Testing:** Verify accuracy, confusion matrix, AUC compute during evaluation

---

### Phase 4: Models (CAREFUL)
**Files:** `astro_model/astro_model.py`, test files (5 files)  
**Changes:**
- `tf.train.get_or_create_global_step()` → Create explicit global step variable
- Keep `tf.train.Scaffold()` in tests (tests can stay in compat mode)

**Risk:** High - Global step management affects Estimator training  
**Testing:** Full train/eval cycle, verify checkpoints save correctly

---

## Summary Table

| Phase | Files | Risk | Changes | Testing |
|-------|-------|------|---------|---------|
| 0 | 3 | Low | CLI refactoring | CLI interface |
| 1 | 2 | Low | File I/O APIs | TFRecord generation |
| 2 | 2 | Medium | Optimizers, LR schedule | Training convergence |
| 3 | 1 | Medium | Metrics variables | Evaluation metrics |
| 4 | 5 | High | Global step | Full train/eval |

---

## API Migration Reference

### Simple Replacements (No Testing Needed)
```python
# Logging
tf.logging.info("msg")          → import logging; logging.info("msg")

# File operations
tf.gfile.MakeDirs(dir)          → from pathlib import Path; Path(dir).mkdir(parents=True, exist_ok=True)
tf.gfile.Glob(pattern)          → from glob import glob; glob(pattern)

# TFRecord writing
tf.python_io.TFRecordWriter     → tf.io.TFRecordWriter

# Main function
tf.app.run(main=fn)             → fn(args)
```

### Tested Replacements (Require Validation)
```python
# Optimizers
tf.train.MomentumOptimizer      → tf.keras.optimizers.SGD(momentum=...)
tf.train.AdamOptimizer          → tf.keras.optimizers.Adam

# Learning rate decay
tf.train.polynomial_decay       → tf.keras.optimizers.schedules.PolynomialDecay
```

### Keep As-Is (Still Works)
```python
tf.contrib.tpu.*                (Estimator TPU support - still functional)
tf.train.Scaffold()             (In tests only - use compat.v1)
```

---

## Execution Checklist

### Before Starting: DO THIS ONCE
- [ ] Create feature branch
- [ ] Save current state/commit
- [ ] Add Python logging imports where needed
- [ ] Review this plan

### Phase 0 (Entry Points)
- [ ] Update train.py
- [ ] Update evaluate.py
- [ ] Update predict.py
- [ ] Test: Run `python astronet/train.py --help`

### Phase 1 (File I/O)
- [ ] Update generate_input_records.py
- [ ] Update dataset_ops.py
- [ ] Test: Generate TFRecords from sample data

### Phase 2 (Training)
- [ ] Update ops/training.py
- [ ] Verify util/estimator_util.py compatibility
- [ ] Test: Train on small test dataset

### Phase 3 (Metrics)
- [ ] Update ops/metrics.py
- [ ] Test: Verify metrics output during evaluation

### Phase 4 (Models)
- [ ] Update astro_model.py global_step
- [ ] Test: Full train/eval cycle

### Final
- [ ] Run all tests
- [ ] Verify predictions work
- [ ] Compare results with baseline

---

## Notes

✅ **Keeping as-is (scope excludes):**
- light_curve/ - utility package
- tf_util/ - utility package  
- astrowavenet/, beam/, experimental/ - separate packages
- Tests - can remain in TF1 compat mode
- pyproject.toml - no packaging setup needed

✅ **Core principle:** 
- Same directory structure
- Same file names
- Same model registry (models.py)
- Same CLI interface
- Same TFRecord format
- **Only APIs change, behavior stays identical**

---

## Questions for Review

1. Should we migrate to `tf.keras.metrics` in Phase 3, or keep tf.compat.v1 approach?
2. Should we add modern pyproject.toml packaging after modernization? (Out of scope currently)
3. Any specific TensorFlow version target? (Assumed 2.13+)

---

**Status:** Ready for Phase 0 implementation
