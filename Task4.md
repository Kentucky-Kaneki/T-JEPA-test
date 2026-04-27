Next is **Step 4: turn the latent model into an agent that chooses actions**. At last, the machine gets to do something instead of just staring at the network like it owes it money.

---

# Step 4 - Attach an RL head to the JEPA latent

## Goal

You already have:

* a **representation model** that turns NASim state into a latent
* a **temporal predictor** that understands how actions change that latent

Now you add:

* a **policy head** that chooses actions
* a **value head** that estimates how good a state is

This is the point where the model stops being a scholar and starts being an attacker.

---

## 1. What the latent should feed into

Your JEPA encoder should output a state embedding:
[
z_t = f_\theta(s_t)
]

That latent becomes the input to the RL part.

From that latent, the agent should produce:

### Policy

[
\pi(a_t \mid z_t)
]

This says:

> given the current latent state, what action should I take?

### Value

[
V(z_t)
]

This says:

> how promising is this state for eventual reward?

---

## 2. Why this step comes after self-supervised learning

Because the RL head should not learn on garbage.

If you attach RL too early:

* the policy learns slowly
* the gradients are noisy
* the encoder chases reward before it understands the environment

If you pretrain first:

* the encoder already knows structure
* the policy learns on meaningful features
* training becomes much less cursed

So the order matters:

1. learn structure
2. learn dynamics
3. learn control

Humans, naturally, prefer the opposite and then blame the optimizer.

---

## 3. What the RL head should look at

The RL head should **not** see raw NASim state.

It should see:

* the JEPA latent for the current state
* optionally the predicted future latent
* optionally action history

The clean version is:

[
z_t \rightarrow \text{policy/value heads}
]

The more advanced version is:

[
(z_t, \hat{z}_{t+1}) \rightarrow \text{policy/value heads}
]
