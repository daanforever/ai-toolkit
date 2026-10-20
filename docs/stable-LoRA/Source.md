### ⚙️ How it works: formula and stopping rule

Stable-LoRA is not a separate optimizer but a **patch to the standard LoRA update loop**. It inserts one operation before the optimizer step:

**Update with shrinkage:**
\[
A_{t+1} = (1-\lambda)A_t - \eta g_A^t
\]
where \(\lambda \in (0,1)\) is the shrinkage coefficient, and \(\eta g_A^t\) is the usual gradient step for \(A\).

That is, at the earliest steps you do not simply update \(A\) by gradient — you **additionally shrink its norm exponentially**. This “presses” \(A\) toward a scale comparable to \(B\) and prevents the instability embedded in \(A_0\) from growing.

**When to stop shrinking:**
Shrinkage does not need to be applied forever. The stopping criterion is that the **average norm of \(A\) becomes no greater than the average norm of \(B\)**:

\[
\frac{\|A\|_F}{\sqrt{n}} \le \frac{\|B\|_F}{\sqrt{m}}
\]

(where \(n\) and \(m\) are the corresponding dimensions). Once this condition is met, shrinkage stops and training continues normally. The paper notes that if you continue shrinking after the condition is met, the quality difference is negligible (≤0.14), i.e. what matters is **reaching** the stability condition, not stopping at exactly the perfect moment.

### 🧩 How to integrate it into your pipeline (pseudocode)

You need to find the place in your training script (kohya, diffusers, or your own) where LoRA parameters are updated, and insert shrinkage **before** calling `optimizer.step()`.

Step 1. Once before the training loop — build (A, B) pairs.

```python
def build_lora_pairs(model):
    pairs = []
    modules = dict(model.named_modules())

    # PEFT / diffusers: module.lora_A[adapter], module.lora_B[adapter]
    for name, mod in modules.items():
        if hasattr(mod, 'lora_A') and hasattr(mod, 'lora_B'):
            for adapter in mod.lora_A.keys():
                pairs.append({
                    'A': mod.lora_A[adapter].weight,
                    'B': mod.lora_B[adapter].weight,
                    'stable': False,
                })

    # kohya: foo.lora_down / foo.lora_up
    for name, mod in modules.items():
        if name.endswith('.lora_down'):
            up_name = name[:-len('.lora_down')] + '.lora_up'
            if up_name in modules:
                pairs.append({
                    'A': mod.weight,
                    'B': modules[up_name].weight,
                    'stable': False,
                })

    return pairs

pairs = build_lora_pairs(model)
```

Step 2. In the training loop — before optimizer.step().

```python
with torch.no_grad():
    for p in pairs:
        if p['stable']:
            continue
        A, B = p['A'], p['B']

        # RMS in fp32 to avoid underflow/overflow in bf16/fp16
        rms_A = A.float().norm() / A.numel() ** 0.5
        rms_B = B.float().norm() / B.numel() ** 0.5

        if rms_A <= rms_B:
            p['stable'] = True
        else:
            A.mul_(1.0 - lambda_shrink)

optimizer.step()
optimizer.zero_grad()
```

**Important nuances for Z-Image:**
- In the paper, \(A\) has shape \((r, n)\) and \(B\) has shape \((m, r)\). In different LoRA implementations the orientation may be transposed, so the norm must be taken along the axis corresponding to the “input” dimension. Check your implementation’s code.
- Shrinkage is applied **only to \(A\)**; \(B\) is not touched.
- If your optimizer already uses `weight_decay`, shrinkage **compounds** with it, but that is not a problem: it is just another multiplicative decay that acts only in the early phase.
- The `stable` flag can be saved in the training state after it triggers, so you do not have to check the condition every step (although the check is cheap).

### 🔢 Hyperparameters

- **\(\lambda\) (shrinkage ratio):** In the paper’s experiments the set **`{0.0005, 0.001, 0.002, 0.005}`** was used, and the authors note that \(\lambda\) **requires tuning** within this range; there is no adaptive scheme without search yet. Start with `0.001` or `0.002` as a compromise: too small a value will not suppress the instability in time, too large a value will excessively “suppress” \(A\) at the start.


