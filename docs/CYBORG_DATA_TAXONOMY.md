# CybORG Data & Observation Representation Taxonomy

This document outlines the data structures exposed by **CybORG 3.1** (specifically `Scenario1b` / CAGE Challenge 2) and defines candidate tokenization strategies for input into a **Joint-Embedding Predictive Architecture (JEPA)**.

---

## 1. CybORG Observation Structures

CybORG exposes three distinct observation representations via built-in wrappers:

| Representation Level | Python Type / Shape | Content & Semantics | Best Use Case |
| :--- | :--- | :--- | :--- |
| **Raw Observation** | `dict` (nested per host) | Detailed interfaces (IP, subnets), active sessions (`VELOCIRAPTOR_CLIENT`, PIDs), process tables, user account lists, and OS specs. | Deep host-level parsing, custom graph/token builders. |
| **Tabular Table** (`BlueTableWrapper`) | `PrettyTable` / `dict` | 13 host rows (`Defender`, `Enterprise0-2`, `Op_Host0-2`, `Op_Server0`, `User0-4`) $\times$ 5 columns (`Subnet`, `IP Address`, `Hostname`, `Activity`, `Compromised`). | Structured host-token JEPA input & human debugging. |
| **Flat Vector** (`ChallengeWrapper`) | `np.ndarray` `(52,)` | Fixed-size 1D numerical vector encoding discretized status across 13 hosts. | Baseline standard RL / naive tabular T-JEPA comparison. |

---

## 2. Action Space Structure

Blue actions in CybORG 3.1 are parameterized abstract actions:
* **Abstract Action Types**: `Sleep`, `Monitor`, `Analyse`, `Remove`, `Misinform`, `Restore`.
* **Action Parameters**: `hostname` / `ip_address`, `session_id`, `subnet`.
* **Discrete Action Space** (`ChallengeWrapper`): Maps valid action combinations to a discrete index range $[0, 65]$.

---

## 3. Candidate JEPA Tokenization Strategies

To determine what representation best supports action-conditioned latent world modeling, we evaluate four candidate tokenization schemes:

### Representation A: Flat Feature Tokens (Standard T-JEPA Adaptation)
* **Concept**: Flatten 52 vector features into individual tokens $[x_1, x_2, \dots, x_{52}]$.
* **Embeddings**: Linear projection + feature index embedding + feature type embedding.
* **Pros**: Direct reuse of existing [FeatureEmbedding](file:///c:/Users/rohil/Documents/GenAI%20Micro%20Project/T-JEPA-test/model.py#L16-L79).
* **Cons**: Ignores multi-host network topology and spatial relationships.

### Representation B: Host-Centric Tokens (Recommended Hypothesis)
* **Concept**: Each of the 13 network nodes forms an individual token:
  $$\text{HostToken}_i = \text{Embed}(\text{Subnet}_i, \text{IP}_i, \text{Activity}_i, \text{Compromise}_i)$$
* **Embeddings**: $E_{\text{host\_id}} + E_{\text{subnet}} + E_{\text{activity}} + E_{\text{compromise}} + E_{\text{time\_step}}$.
* **Pros**: Preserves network node semantics; attention mechanisms operate natively over network topology.

### Representation C: Temporal Host Sequence Tokens
* **Concept**: Sequence of host tokens across history window $h$ ($t-h \dots t$):
  $$Z_{history} = [\mathbf{h}_{1, t-h}, \dots, \mathbf{h}_{13, t-h}, \dots, \mathbf{h}_{1, t}, \dots, \mathbf{h}_{13, t}]$$
* **Pros**: Handles partial observability by tracking state evolution over time.

### Representation D: Action Token Integration
* **Concept**: Condition predictor on Blue's action $a_t^{Blue}$:
  $$\mathbf{a}_t = E_{\text{action\_type}}(a_{\text{type}}) + E_{\text{target\_host}}(a_{\text{host}})$$
* **Pros**: Allows predictor to simulate hypothetical outcomes of alternative defensive actions.

---

## 4. Dataset Taxonomy and Identifiers

The dataset consists of **18 shards**, constructed by the Cartesian product of:
* **2 Red Policies**: `bline`, `meander`
* **3 Blue Policies**: `sleep`, `random`, `coverage`
* **3 Collection Seeds**: `1001`, `2003`, `3005`

Each shard contains 100 trajectories of length up to 50 steps, yielding a total of **1,800 trajectories** and **90,000 transitions**. 

To strictly enforce data integrity, the following identification contracts are maintained:
* **`trajectory_id`**: Globally unique across red_policy, blue_policy, collection_seed, and episode index (e.g., `traj_{red}_{blue}_{seed}_{ep}`).
* **`transition_id`**: Globally unique within a trajectory and step index (e.g., `{trajectory_id}_t{t}`).
* **`split_group_id`**: Consists of collection_seed plus episode index (e.g., `group_{seed}_{ep}`). This ID is strictly used to pair test cohorts, ensuring trajectories sharing the same initial state (seed/index) remain in the exact same deterministic split (Train, Val, or Holdout) regardless of policy.
