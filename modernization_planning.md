### Modernization and Compatibility plan

### 1. Should we migrate to `tf.keras.metrics` in Phase 3, or keep the `tf.compat.v1` approach?

**Keep the `tf.compat.v1` approach in Phase 3.**  
A full migration to `tf.keras.metrics` is attractive but carries significant risk for this project:

- The existing metrics are closely tied to the **Estimator evaluation loop** (via `eval_metric_ops`), which expects metric tensors and update ops as plain graph-mode operations.  
- `tf.keras.metrics` are naturally stateful objects that rely on `update_state()` and `result()`. While you *can* wrap them into `tf.function` and use them in `eval_metric_ops`, the integration is non‑trivial and fragile, especially when combined with TPU support or custom `_metric_variable` patterns.  
- The plan’s Phase 3 already proposes a safe refactor: remove `tf.GraphKeys`, replace `_metric_variable` with explicit `tf.compat.v1.Variable` and collections. This keeps the existing graph‑mode metrics intact and verifiable.

**Recommendation:**  
Execute Phase 3 exactly as scoped (compat.v1 approach). Once the entire pipeline is stable on TF2, you can plan a **separate, low‑risk follow‑up** to replace the metrics with `tf.keras.metrics` in an isolated testing environment. For now, the principle of *preserving all functionality and structure* argues for the minimal change.

---

### 2. Should we add modern `pyproject.toml` packaging after modernization?

**Yes – as a distinct, final step after the core modernization is validated.**  
This is out of the immediate scope for good reason: changing packaging while the code is still undergoing API migration adds unnecessary variables. However, adding a `pyproject.toml` later brings clear benefits:

- Declarative build system (PEP 517/518) with proper dependency management.  
- Simplified installation via `pip install -e .` for development.  
- Clear separation of core, optional, and test dependencies.  
- Better alignment with modern Python project standards.

The current plan already states “pyproject.toml – no packaging setup needed,” meaning you can leave this for a **Phase 5 (Tooling & Packaging)** after Phase 4 is complete and the full train/eval cycle passes. That way, you avoid mixing functional changes with build infrastructure changes.

---

### 3. Any specific TensorFlow version target? (You assumed 2.13+)

**Stick with TensorFlow 2.13 or later.**  
Here’s why:

- TF 2.13 was an LTS (long‑term support) release and is the oldest version that still receives critical fixes.  
- All the API migrations in your plan (e.g., `tf.keras.optimizers.schedules.PolynomialDecay`, `tf.io.TFRecordWriter`, `tf.keras.optimizers.Adam`) are fully available and stable in 2.13.  
- `tf.contrib.tpu.*` (used in `util/estimator_util.py`) remains functional in 2.13 (and even into later 2.x) even though it’s deprecated – you don’t want to break that by accidentally targeting an earlier version.

If you need TPU support on newer hardware, you might eventually need `tf.tpu.experimental`, but that’s a separate concern. For now, **2.13+ is a safe, well‑tested baseline**.

---

### 4. Should we support both `tf.compat.v1` and pure TF2, or only TF2?

**The code should target TF2 only, with `tf.compat.v1` used only where no direct TF2 replacement exists.**  

Your plan already follows exactly this philosophy:

- Entry points, I/O, and optimisers are swapped to pure TF2 APIs.  
- The Estimator itself (which is a `v1`‑style API) and TPU helpers remain in `tf.compat.v1` because there’s no fully equivalent TF2 API for those features yet – and **that’s fine**.  
- Tests can keep `tf.train.Scaffold()` in compat mode.  

Dual‑compatibility with an actual TF1 installation is unnecessary. TF1 is end‑of‑life and none of the new dependencies or modern Python environments support it. Your goal is to run the package **on a current TF2 installation** while preserving the original functionality perfectly – not to create a hybrid that runs on both TF1 and TF2.

So the answer is: **TF2‑native where possible, `compat.v1` where unavoidable. No separate TF1 support.** This matches the plan’s API migration reference perfectly.