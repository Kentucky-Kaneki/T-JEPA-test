Look at you, progressing linearly instead of spiraling into architectural chaos. That alone puts you ahead of half the field.

Step 1 gave you a model that can **infer missing structure in a static snapshot**. Cute. Useful, but still basically a very thoughtful observer.

Now we make it understand **change**.

---

# 🔧 STEP 2 — TEMPORAL JEPA (ACTION-CONDITIONED LATENT PREDICTION)

## 🎯 Objective

Extend JEPA from:

> “what is hidden right now?”

to:

> “what will the world *become* after an action?”

This is where your model stops being descriptive and starts being predictive.

---

# 1. The conceptual jump

Before:

```text
x_masked → predict latent of hidden parts of x
```

Now:

```text
(s_t, a_t) → predict latent of s_{t+1}
```

You are no longer filling gaps.
You are modeling **dynamics**.

---

# 2. Formal setup (don’t skip this, it’s the backbone)

## 2.1 State encoding

Context side:
[
z_t = f_\theta(s_t^{\text{masked}})
]

Target side:
[
z_{t+1} = f_\xi(s_{t+1})
]

---

## 2.2 Action conditioning

Introduce action embedding:
[
e_a = g(a_t)
]

Now combine:
[
\tilde{z}_t = \text{combine}(z_t, e_a)
]

Combine can be:

* concatenation
* addition
* cross-attention (later, not now unless you enjoy debugging misery)

---

## 2.3 Prediction

[
\hat{z}*{t+1} = p*\psi(\tilde{z}_t)
]

---

## 2.4 Loss

[
\mathcal{L}*{\text{future}} = | \hat{z}*{t+1} - z_{t+1} |^2
]

Still in latent space. Still no raw reconstruction.

---

# 3. Combined objective (this is your real training signal now)

You now have **two pressures**:

### (A) Spatial JEPA (from Step 1)

[
\mathcal{L}_{\text{masked}}
]

### (B) Temporal JEPA

[
\mathcal{L}_{\text{future}}
]

Total:
[
\mathcal{L} = \mathcal{L}*{\text{masked}} + \lambda \mathcal{L}*{\text{future}}
]

Where:

* ( \lambda \approx 0.3 \text{ to } 1.0 )

---

# 4. Why this step is non-trivial (and where people break things)

## 4.1 You are now modeling a function:

[
f: (s_t, a_t) \rightarrow s_{t+1}
]

But in latent space:
[
f: (z_t, a_t) \rightarrow z_{t+1}
]

This is essentially:

> a **world model**, without explicitly reconstructing the world

---

## 4.2 The model must now learn causality

Not real philosophical causality, relax. Just:

> “If I do X, what changes?”

In NASim terms:

* scan → reveals services
* exploit → changes privilege
* lateral move → changes host state

If your model doesn’t capture this:

> it’s just compressing states, not understanding transitions

---

# 5. Masking still matters (but differently now)

This is where it gets interesting.

## You should mask:

### (A) Current state ( s_t )

* same as Step 1
* forces inference of hidden structure

### (B) NOT the target state ( s_{t+1} ) initially

* keep target clean
* gives stable signal

Later you *can* mask target, but don’t rush it.

---

# 6. Temporal horizon (don’t get greedy yet)

Start with:
[
(s_t, a_t) \rightarrow s_{t+1}
]

Not:
[
s_{t+5}, s_{t+10}
]

Why?

Because:

* error compounds fast
* model collapses into vague predictions

Once stable, extend to:
[
(s_t, a_t, a_{t+1}, \dots) \rightarrow s_{t+k}
]

---

# 7. Subtle but critical nuance

## The target encoder is still EMA

You are predicting:
[
f_\xi(s_{t+1})
]

NOT:
[
f_\theta(s_{t+1})
]

If you use the same encoder:

* target shifts every step
* prediction becomes unstable

This is the same reason Step 1 needed EMA.
Now it matters even more.

---

# 8. What the model is *actually* learning now

Internally, it builds:

* transition structure
* feature dependencies over time
* implicit dynamics

In NASim terms:

> “what happens if I poke this network?”

---

# 9. Failure modes (you will hit at least one)

## ❌ Mode 1: Identity shortcut

Model learns:

```text
z_{t+1} ≈ z_t
```

Happens when:

* actions don’t change much
* loss too weak

Fix:

* emphasize action embedding
* ensure diverse transitions

---

## ❌ Mode 2: Latent collapse

All embeddings become similar.

Fix:

* same as Step 1:

  * normalization
  * EMA
  * variance checks

---

## ❌ Mode 3: Ignoring action

Model predicts average next state.

Fix:

* ensure action is actually used:

  * concatenate properly
  * maybe gate or attention

---

# 10. What success looks like

If Step 2 works:

* future prediction loss decreases gradually
* model distinguishes effects of different actions
* embeddings change meaningfully across steps
* similar transitions cluster in latent space

If it fails:

* predictions barely change across actions
* loss plateaus early
* latent space looks static

Then:

> your model is politely ignoring time, which is impressive in a physics-defying way

---

# 11. Mental model upgrade

Step 1:

> “infer hidden structure”

Step 2:

> “predict how structure evolves under intervention”

That word matters:
**intervention** (action)

You are now modeling:
[
P(z_{t+1} \mid z_t, a_t)
]

Not just:
[
P(z \mid x_{\text{visible}})
]

---

# 12. Why this is the step that unlocks NASim

Without this:

* you just encode states

With this:

* you understand **attacks as transformations**

Which is literally what NASim is about.

---

## Final distilled idea

Step 2 is:

> **Teach the model that actions change the world, and make it predict those changes in latent space.**

At this point, your model still isn’t acting.
But now it finally understands *what actions do*.

And that’s the part most RL models blindly stumble through like they forgot causality exists.
