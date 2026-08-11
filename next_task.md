You’re right—the amendment became too compressed and depended on the previous prompt. Here is the complete, self-contained prompt combining the remaining corrections with Phase 3 execution.

```text
# Cyber-JEPA Phase 3 — Structure Preservation, Reproducibility and Causal Aggregation Study

## Role

You are implementing and executing Phase 3 of the defender-oriented Cyber-JEPA research project.

Repository:

C:\Users\rohil\Documents\GenAI Micro Project\T-JEPA-test

This phase must proceed immediately while resolving the remaining Phase 2 infrastructure issues. Do not create another standalone Phase 2 correction cycle, and do not rerun the previous broad Phase 2 matrix.

The unresolved reproducibility, cohort, statistical and structured-input issues are Phase 3 Milestone 0. Once those gates pass, continue directly into the Phase 3 architecture pilot and experiment.

Do not pause between milestones unless:

- dataset identity is invalid;
- the fixed evaluation cohort cannot be constructed;
- tests cannot pass without changing the research design;
- the one-seed architecture pilot remains unstable after diagnosis;
- required GPU resources are unavailable.

Do not implement an autonomous Blue action-selection policy. Cyber-JEPA remains an action-conditioned latent world model evaluated for defender-oriented representation learning.

---

# 1. Phase 3 research question

The primary research question is:

> Do structured Cyber-JEPA representations genuinely underperform flat temporal representations, or does their current performance result from premature compression at the representation-to-predictor boundary?

The specific causal hypothesis is:

> Preserving feature, host and hierarchical tokens through the JEPA predictor will improve future cyber-state representation quality compared with reducing those tokens to a single vector before prediction.

Phase 3 must distinguish five different components:

1. Input semantics and tokenization.
2. Temporal/entity contextual encoding.
3. Context aggregation.
4. Action-conditioned latent prediction.
5. Frozen-probe evaluation.

Do not use “structured representation failure” when only the aggregation interface has been tested.

Do not use “mean pooling causes collapse” unless an aggregation intervention—holding the remaining representation design fixed—supports that conclusion statistically.

---

# 2. Evidence inherited from Phase 2

Treat the corrected Phase 2 results as preliminary historical evidence:

- The identity/oracle corruption was corrected.
- The regenerated dataset contains 90,000 unique transitions.
- It contains 1,800 globally unique trajectories.
- It contains 300 intentional split groups.
- Every transition has one corresponding oracle row.
- Current flat implementations produced higher holdout probe scores than current structured implementations.
- The current structured implementations exhibited low effective rank.
- Flat h4 and h8 also crossed the existing low-rank threshold, so effective rank alone does not explain probe performance.
- Fully controlled training seeds, common evaluation cohorts and trajectory-level paired statistics were not completed.
- Mean pooling remains a candidate mechanism rather than an established cause.

Preserve existing Phase 2 artifacts. Never overwrite them.

All new artifacts belong under:

```text
experiments/phase3/
runs/phase3/
```

---

# 3. Milestone 0 — Resolve remaining experimental-integrity issues

Complete these corrections as the first part of Phase 3. Commit them before launching the Phase 3 GPU pilot.

## 3.1 Deterministic execution

The existing `set_seed()` helper must be invoked, not merely defined.

Before constructing any encoder, model, optimizer, scheduler or DataLoader:

- call `random.seed(model_seed)`;
- call `numpy.random.seed(model_seed)`;
- call `torch.manual_seed(model_seed)`;
- call `torch.cuda.manual_seed_all(model_seed)`;
- set `torch.backends.cudnn.deterministic = True`;
- set `torch.backends.cudnn.benchmark = False`;
- use `torch.use_deterministic_algorithms(True)` when supported;
- document any operation that requires a deterministic-algorithm exception.

Separate these seeds in configuration:

```yaml
split_seed: 42
cohort_seed: 4201
training_subset_seed: 7301
model_seed: <per-run seed>
dataloader_seed: <per-run seed>
bootstrap_seed: 9901
```

Do not overload one seed for every purpose.

Create an explicitly seeded `torch.Generator` for the shuffled training DataLoader.

If DataLoaders use `num_workers=0`, remove all claims about worker synchronization.

If workers are enabled:

- pass `worker_init_fn`;
- seed Python and NumPy within each worker;
- record `num_workers`;
- verify deterministic batch order in a focused test.

Add a deterministic smoke test:

1. Start from the same clean Git commit.
2. Run one small configuration twice.
3. Compare the first several batch IDs.
4. Compare initial model hash.
5. Compare epoch losses.
6. Compare final checkpoint parameters.
7. Compare probe predictions.

Define and document numeric tolerances rather than requiring unsafe bitwise CUDA identity.

## 3.2 Fixed paired cohort

Create one persistent Phase 3 evaluation cohort.

Core Phase 3 settings:

```text
scenario: Scenario1b
prediction horizon: k=8
history length: h=4
split unit: split_group_id
holdout type: unseen split-group holdout
```

The cohort must contain target transitions valid for every Phase 3 core configuration.

Store, for every selected evaluation sample:

- transition_id;
- trajectory_id;
- split_group_id;
- context timestep;
- target timestep;
- Red policy;
- Blue policy;
- collection seed;
- future critical-server label;
- current critical-server persistence label.

The same target transition IDs must be used for:

- every representation;
- every aggregation method;
- every model seed;
- every capacity-matched control.

Do not randomly choose a different 2,000-window holdout subset for each seed.

Persist:

```text
experiments/phase3/phase3_cohort.parquet
experiments/phase3/phase3_cohort.json
experiments/phase3/phase3_cohort.sha256
```

The JSON must include:

- cohort-generation configuration;
- number of split groups;
- number of trajectories;
- number of transitions;
- class prevalence;
- prevalence by Red policy;
- prevalence by Blue policy;
- dataset manifest hashes;
- cohort SHA-256.

Add assertions proving cohort equality across all configurations.

## 3.3 Strict oracle joins

Retain the globally unique identity contracts:

```text
trajectory_id =
    red_policy + blue_policy + collection_seed + episode_index

transition_id =
    trajectory_id + step_index

split_group_id =
    collection_seed + episode_index
```

Replace every label lookup equivalent to:

```python
oracle_map.get(transition_id, 0)
```

with a strict lookup that raises an explicit error.

Before every experiment, verify:

- exactly 18 expected shards exist;
- exactly 90,000 transitions exist;
- exactly 1,800 trajectory IDs exist;
- exactly 300 split-group IDs exist;
- transition IDs are globally unique;
- oracle transition IDs are globally unique;
- every transition has exactly one oracle row;
- there are no orphan oracle rows;
- cohort transition IDs exist in both tables;
- labels contain the expected classes.

Never silently interpret a missing label as a negative label.

## 3.4 Persistence baseline

Use the correct terminology:

> Oracle critical-server-label persistence baseline.

The baseline is:

```text
prediction at t+k = critical_server_compromised at t
target at t+k     = critical_server_compromised at t+k
```

Do not describe this as full-state persistence because it predicts one oracle label, not the complete future state.

Save each persistence prediction beside the JEPA probe prediction so paired comparisons can be performed.

## 3.5 Policy-transfer split repair

The same `split_group_id` appears under both Red policies. Therefore, selecting B-line and Meander using only split-group IDs causes overlap or reintroduction of the excluded policy.

Define policy-transfer data by globally unique trajectories plus explicit policy predicates:

```text
bline_to_meander:
    training trajectories: red_policy == bline
    testing trajectories: red_policy == meander

meander_to_bline:
    training trajectories: red_policy == meander
    testing trajectories: red_policy == bline
```

Verify:

- zero trajectory overlap;
- correct Red policy in each side;
- no accidental selection of the opposite policy through shared split groups;
- Blue-policy composition is reported;
- collection-seed composition is reported.

Do not call the primary Phase 3 holdout OOD. Reserve OOD or policy transfer for these explicit sidecars.

## 3.6 Validation gate

Repair the obsolete split-test imports and update tests for the new identity model.

Before architecture work proceeds, require:

- complete pytest collection;
- all identity tests passing;
- all oracle-join tests passing;
- split-group disjointness tests passing;
- cohort-equality tests passing;
- persistence fixture tests passing;
- deterministic smoke test passing;
- Ruff passing for Phase 3 changed files;
- mypy passing for Phase 3 changed files.

Do not weaken assertions to obtain a green test suite.

Commit Milestone 0 before the GPU pilot.

---

# 4. Milestone 1 — Typed context interface

Replace implicit `tensor.dim()` branching with an explicit representation contract.

Introduce a structure equivalent to:

```python
@dataclass
class ContextTokens:
    tokens: torch.Tensor
    padding_mask: torch.Tensor
    visibility_mask: torch.Tensor | None
    time_ids: torch.Tensor
    entity_ids: torch.Tensor
    token_type_ids: torch.Tensor
    global_token: torch.Tensor | None
    metadata: dict[str, Any]
```

Required shapes:

```text
tokens:          [B, L, D]
padding_mask:    [B, L]
visibility_mask: [B, L] or None
time_ids:        [B, L]
entity_ids:      [B, L]
token_type_ids:  [B, L]
global_token:    [B, D] or None
```

Token types must distinguish:

- temporal flat token;
- feature token;
- host token;
- subnet token;
- global token;
- action token;
- target query.

Requirements:

- Every representation returns a deliberate typed object.
- CyberJEPA must not infer representation semantics from tensor rank.
- CyberJEPA must not silently mean-pool structured tokens.
- Aggregation is an explicit configured component.
- Masks propagate into attention.
- Invalid or padded tokens cannot influence aggregation.
- The selected aggregation mode is stored in the resolved run configuration.
- Incompatible representation/aggregator combinations raise clear errors.
- Existing flat behavior remains available as a control.

Add unit tests for:

- shapes;
- masks;
- metadata;
- device movement;
- batch size one;
- variable sequence lengths;
- unknown hosts;
- entirely masked optional entities;
- deterministic output in evaluation mode.

---

# 5. Milestone 2 — Correct structured input semantics

## 5.1 Feature-token representation

Feature tokens are the primary causal representation because they can preserve all 52 ChallengeWrapper values without inventing additional semantics.

For each feature token, include:

- projected scalar value;
- feature index embedding;
- verified host association where applicable;
- feature-type embedding;
- relative timestep embedding;
- visibility/validity embedding.

Every declared embedding must be used.

Add an audit before learned projection proving:

- exactly 52 input values become 52 feature-token values;
- token values equal their corresponding flat-vector values;
- no value is dropped;
- no value is duplicated unintentionally;
- feature ordering is deterministic;
- reconstruction of the original 52-value ordering is possible.

Do not call feature tokens semantic if their feature meaning has not been verified.

## 5.2 Host-token representation

Do not interpret arbitrary pairs of the four ChallengeWrapper host values through `argmax`.

Use one of these verified approaches:

Preferred:

- construct canonical host fields from BlueTable/raw observation semantics.

Acceptable fallback:

- represent the four ChallengeWrapper values as four explicit host-local fields without renaming them as independent activity and compromise logits.

Each host token must represent:

- host identity;
- subnet identity;
- four verified host-local values or canonical activity/compromise fields;
- visibility/known-host state;
- relative timestep.

Unknown and unobserved values must be explicit.

Add tests using hand-constructed CybORG states demonstrating:

- known clean host;
- unknown host;
- scanned host;
- exploited host;
- privileged compromise where available;
- visibility changes;
- correct host and subnet identity.

## 5.3 Hierarchical representation

Use the actual Scenario1b topology.

Stage-one subnet aggregation must enforce:

```text
Enterprise subnet -> Enterprise hosts only
Operational subnet -> Operational hosts only
User subnet -> User hosts only
```

Apply the registered subnet mask in attention.

Add perturbation tests:

- modifying a User host must not change the Enterprise subnet token before global aggregation;
- modifying an Enterprise host must not change the Operational subnet token before global aggregation;
- masked unknown hosts must not contribute as observed evidence;
- modifying one subnet may change the final global token after global aggregation.

Do not report hierarchical topology preservation until these tests pass.

---

# 6. Milestone 3 — Explicit aggregation interventions

Implement all aggregation methods behind one protocol.

Use an interface equivalent to:

```python
class ContextAggregator(Protocol):
    def forward(
        self,
        context: ContextTokens,
    ) -> AggregatedContext:
        ...
```

## 6.1 Legacy last-step mean

Preserve the existing baseline:

```text
[B,T,N,D]
    -> select final timestep
    -> mask-aware mean over N entities
    -> [B,D]
```

Name it explicitly:

```text
legacy_last_step_mean
```

This is the controlled compression baseline.

Do not call it structure-preserving.

## 6.2 Learned-query aggregation

Use a learned query token that attends across all valid temporal/entity tokens:

```text
[B,T,N,D]
    -> [B,T×N,D]
    -> learned query cross-attention
    -> [B,D]
```

Requirements:

- time IDs remain available;
- entity IDs remain available;
- token types remain available;
- visibility and padding masks are applied;
- attention covers all timesteps;
- attention weights are saved for diagnostics;
- no entity mean is applied after cross-attention.

## 6.3 Token-preserving predictor

The predictor must consume the structured token memory directly:

```text
context memory: [B,L,D]
action tokens:  [B,k,D]
target query:   [B,1,D]
```

Recommended flow:

```text
context tokens remain encoder memory
        +
ordered structured action tokens
        +
horizon-conditioned target query
        ↓
cross-attention/Transformer decoder
        ↓
predicted target latent [B,D]
```

Requirements:

- do not flatten `L×D` into an enormous single vector;
- do not mean-pool context before prediction;
- include horizon identity;
- include target-granularity identity;
- preserve action order;
- preserve context time identity;
- expose attention weights where practical;
- support masks;
- return one network-level predicted latent during the Phase 3 core experiment.

The action sequence must remain a first-class model input. Verify that changing action targets changes predictor output.

---

# 7. Target-encoder correction

Remove the artificial repetition:

```text
O_(t+k), O_(t+k), O_(t+k), O_(t+k)
```

Introduce a supported single-frame target path:

```text
target O_(t+k)
    -> target tokenizer with T=1
    -> target aggregation
    -> stop-gradient target latent
```

Requirements:

- online and target encoders use equivalent tokenization semantics;
- target encoder is a complete EMA copy;
- target parameters have `requires_grad=False`;
- target encoder remains in evaluation mode during training;
- target dropout is inactive;
- optimizer excludes target parameters;
- complete tokenizer and encoder parameters receive EMA updates;
- target construction is identical across aggregators within a representation family.

Add tests proving:

- target output is deterministic;
- target encoder receives no gradients;
- target encoder is not changed by optimizer steps;
- EMA changes target parameters;
- target input contains one actual future observation, not repeated history;
- checkpoint reload preserves online and target outputs.

---

# 8. Capacity-control design

Aggregation variants may add trainable parameters. Report separate counts for:

- representation tokenizer;
- context encoder;
- aggregator;
- action encoder;
- predictor;
- target encoder;
- total trainable parameters;
- total non-trainable EMA parameters.

Within one representation family, if a new aggregation path changes total trainable parameters by more than 10%, create a supplementary capacity-matched legacy control.

The matched control must use a legitimate residual MLP or additional predictor block with approximately equivalent trainable capacity.

Do not add unused dummy parameters.

The capacity-matched control answers:

> Did performance improve because structure was preserved, or simply because parameters were added?

Do not mix supplementary capacity controls into the core configuration count. Report them separately.

---

# 9. Phase 3 experimental matrix

Use:

```text
prediction horizon: k=8
history length: h=4
hidden dimension: D=64
model seeds: 1001, 2003, 3005, 4007, 5009
fixed cohort: identical across all runs
```

Keep constant:

- dataset;
- train/validation/holdout groups;
- target cohort;
- batch size;
- optimizer;
- learning rate;
- weight decay;
- scheduler;
- epoch limit;
- early-stopping rule;
- EMA schedule;
- probe configuration;
- target label;
- statistical procedure.

## 9.1 One-seed pilot

Run seed 1001 for:

1. `flat_h4_control`
2. `feature_legacy_mean`
3. `feature_query_pool`
4. `feature_token_predictor`

The pilot must verify:

- finite train and validation losses;
- no NaN or Inf parameters;
- correct mask behavior;
- target encoder remains frozen;
- deterministic checkpoint reload;
- fixed cohort hash matches;
- predictions are not constant;
- action shuffle changes predictor output;
- time shuffle changes at least one time-sensitive output;
- per-sample predictions are saved;
- attention artifacts are generated;
- GPU memory usage is acceptable.

Do not launch the final matrix until the pilot passes.

## 9.2 Core experiment

Run:

| Configuration | Representation | Context boundary |
|---|---|---|
| `flat_h4_control` | Flat | Existing learned global/REG-token pathway |
| `feature_legacy_mean` | Feature | Final-timestep entity mean |
| `feature_query_pool` | Feature | Learned query across temporal-feature tokens |
| `feature_token_predictor` | Feature | Predictor attends directly to all feature tokens |
| `host_legacy_mean` | Canonical host | Final-timestep host mean |
| `host_query_pool` | Canonical host | Learned query across temporal-host tokens |
| `host_token_predictor` | Canonical host | Predictor attends directly to host tokens |
| `hierarchical_current` | Host/subnet | Corrected current global-token pathway |
| `hierarchical_masked_predictor` | Host/subnet | Masked host/subnet tokens retained into predictor |

Total:

```text
9 configurations × 5 model seeds = 45 core runs
```

Do not introduce hidden-dimension sweeps in the core Phase 3 matrix.

Do not automatically continue failed runs with partial metrics.

Each run state must be one of:

- pending;
- running;
- completed;
- failed;
- interrupted;
- invalid.

A run is completed only when every required artifact exists.

---

# 10. Diagnostics

Measure latent geometry at:

1. Encoder token output before aggregation.
2. Context representation after aggregation.
3. Predicted future latent.
4. EMA target latent.

Record:

- effective rank;
- effective-rank fraction;
- median per-dimension standard deviation;
- near-constant dimension fraction;
- singular-value spectrum;
- PC1/PC5/PC10 explained variance;
- mean token-to-token cosine similarity;
- temporal token similarity;
- entity token similarity;
- attention entropy;
- attention mass per timestep;
- attention mass per entity;
- fraction of padded/masked tokens;
- gradient norm by subsystem.

Replace the single boolean collapse narrative with separate warnings:

```text
low_rank_warning
near_constant_warning
constant_prediction_warning
probe_failure
training_instability
```

Retain the old `is_collapsed` metric only for historical comparability.

Do not automatically reject a model solely because effective-rank fraction is below 10%. Flat h4/h8 previously demonstrated that low rank can coexist with useful probe performance.

---

# 11. Evaluation metrics

## 11.1 Primary metric

Frozen linear-probe Macro F1 for:

```text
critical_server_compromised at t+8
```

Evaluate on the fixed unseen split-group holdout.

## 11.2 Secondary metrics

Report:

- AUROC;
- balanced accuracy;
- precision and recall per class;
- Brier score;
- calibration curve/error;
- JEPA validation loss;
- oracle-label persistence Macro F1;
- paired delta over persistence;
- changed-target-only Macro F1;
- unchanged-target Macro F1;
- effective rank;
- inference latency;
- peak GPU memory.

Break down primary metrics by:

- Red policy;
- Blue policy;
- collection seed;
- split group.

Do not select the winner using the holdout repeatedly during development. Use validation for model/checkpoint selection and reveal holdout metrics only after run completion.

---

# 12. Sensitivity controls

For the final flat control and the strongest validation-selected structured model, run evaluation-time controls:

1. Ordered history.
2. Shuffled temporal order.
3. Reversed temporal order.
4. Shuffled entity identities.
5. Shuffled action order.
6. Actions replaced with Sleep/no-op.
7. Action target host replaced.
8. Current observation only.
9. Context tokens zeroed.
10. Visibility masks removed only as an explicitly labelled diagnostic.

For each control, calculate paired change in:

- probe F1;
- AUROC;
- predicted latent cosine similarity;
- predicted latent L2 distance;
- JEPA target loss.

Do not conclude that chronological order matters merely because longer history helps. Chronological dependence requires degradation under the temporal-order controls.

Do not conclude that actions matter merely because an action encoder exists. Action dependence requires measurable output or metric degradation under action controls.

---

# 13. Statistical analysis

Save per-sample predictions containing:

- run ID;
- configuration;
- model seed;
- transition ID;
- trajectory ID;
- split-group ID;
- Red policy;
- Blue policy;
- true target label;
- current persistence label;
- probe probability;
- probe prediction;
- sensitivity-control predictions where applicable.

Use paired split-group bootstrap:

- resample split-group IDs;
- include all trajectories and windows associated with each selected group;
- preserve within-group dependence;
- use at least 5,000 bootstrap replicates;
- compute the metric for each paired configuration on the same resample.

Required paired comparisons:

```text
feature_query_pool
    minus feature_legacy_mean

feature_token_predictor
    minus feature_legacy_mean

host_query_pool
    minus host_legacy_mean

host_token_predictor
    minus host_legacy_mean

hierarchical_masked_predictor
    minus hierarchical_current

best structured model
    minus flat_h4_control

each candidate
    minus oracle-label persistence
```

Report:

- mean paired difference;
- median paired difference;
- 95% confidence interval;
- corrected p-value;
- effect size;
- per-seed direction.

Use Holm correction for the five primary aggregation comparisons.

Do not call an improvement statistically significant unless the corrected paired analysis supports it.

Three model-seed standard deviations are not a substitute for group-level paired inference.

---

# 14. Decision rules

## 14.1 Structure-preservation hypothesis supported

Support the hypothesis only if at least one structure-preserving configuration:

- improves over its own legacy representation;
- has a paired 95% interval excluding zero after correction;
- improves changed-target performance;
- retains action sensitivity;
- reproduces in at least four of five seeds;
- is not explained solely by additional parameter count.

## 14.2 Mean-pooling bottleneck supported

Support the mean-pooling explanation only if replacing legacy mean pooling—while retaining the same feature/host tokenizer, data, target and training configuration—produces the required improvement.

## 14.3 Mean-pooling hypothesis rejected

Reject the hypothesis if learned-query and token-preserving variants fail to improve over their corresponding legacy models.

## 14.4 Structured model competitive with flat

Declare a structured model competitive only if:

- its paired Macro F1 difference from flat is within 0.02; and
- it offers another verified advantage such as changed-target accuracy, policy-transfer robustness, interpretability or efficiency.

## 14.5 Flat remains superior

If corrected structure-preserving models remain substantially below flat, report:

> Under the tested Phase 3 interfaces, preserving structured tokens did not close the performance gap with the flat temporal Cyber-JEPA representation.

Do not immediately invent a new rescue mechanism.

## 14.6 Inconclusive result

Use “inconclusive” when:

- seed directions disagree;
- paired confidence intervals include practically meaningful positive and negative differences;
- training instability affects multiple runs;
- cohort or implementation validity is uncertain.

---

# 15. Policy-transfer sidecar

After the primary matrix is complete, select using validation results:

- flat h4 control;
- best feature configuration;
- best host or hierarchical configuration.

Run:

```text
B-line training -> Meander testing
Meander training -> B-line testing
```

Keep policy-transfer results separate from primary holdout results.

Report:

- exact train/test trajectory sets;
- zero-overlap audit;
- class prevalence;
- performance degradation from primary holdout;
- calibration shift;
- effective-rank shift;
- action-sensitivity shift.

Do not use policy-transfer results to retrospectively select the primary model.

---

# 16. Artifact contract

Create:

```text
experiments/phase3/
    PHASE3_PROTOCOL.md
    PHASE3_REPORT.md
    PHASE3_CLAIM_LEDGER.md
    phase3_results.json
    phase3_paired_statistics.json
    phase3_cohort.parquet
    phase3_cohort.json
    phase3_cohort.sha256
    parameter_accounting.csv
    run_manifest.json
    attention_diagnostics/
    figures/
    tables/
```

Each run must contain:

```text
runs/phase3/<run_id>/
    resolved_config.json
    git_commit.txt
    git_status.txt
    environment.json
    dataset_manifest_hashes.json
    cohort_hash.txt
    seeds.json
    history.json
    best.pt
    last.pt
    metrics.json
    predictions.parquet
    latent_diagnostics.json
    attention_diagnostics.npz
    stdout.log
    failure.json
```

Refuse to launch a run from a dirty worktree.

The run manifest must track:

- run ID;
- configuration ID;
- seed;
- state;
- start/end time;
- Git commit;
- dataset hashes;
- cohort hash;
- checkpoint path;
- metrics path;
- failure reason.

Resume only when Git commit, dataset hashes, cohort hash and resolved configuration match.

---

# 17. Paper-drafting documentation

Create or update:

```text
docs/paper/
    README.md
    01_literature_survey.md
    02_problem_formulation.md
    03_methodology.md
    04_experimental_protocol.md
    05_results_considerations.md
    06_threats_to_validity.md
    07_discussion_and_design_log.md
    08_evidence_ledger.md
    09_paper_outline.md
    references.bib
```

Phase 3 methodology must explain:

- why tokenization and aggregation are separate research variables;
- why feature tokens are the primary information-preserving causal test;
- why host semantics require stronger validation;
- how learned-query pooling differs from token-preserving prediction;
- why common cohorts and paired statistics are required;
- why effective rank is diagnostic rather than sufficient proof of usefulness.

Update the literature survey with verified primary sources on:

- JEPA latent prediction;
- structured/tabular JEPA;
- learned pooling;
- Set Transformers;
- Perceiver-style latent queries;
- cross-attention world models;
- graph/hierarchical cyber representations;
- partial observability;
- CybORG/CAGE defender agents.

For every literature source include:

- citation;
- approach;
- input structure;
- aggregation mechanism;
- temporal treatment;
- action treatment;
- evaluation method;
- relevant takeaway;
- limitation;
- implication for Phase 3;
- what the paper does not prove.

Do not fabricate citations.

The evidence ledger must use:

- Proposed
- Implemented
- Unit-tested
- Integration-tested
- Experimentally supported
- Rejected
- Inconclusive

Every paper claim must reference:

- Git commit;
- dataset hash;
- cohort hash;
- run IDs;
- statistical artifact.

Do not write result values into the paper documents until the corresponding artifact exists.

---

# 18. Required implementation sequence

Execute in this order:

## Milestone 0 — Experimental foundation

- deterministic execution;
- fixed cohort;
- strict joins;
- persistence correction;
- policy-transfer repair;
- test-suite repair;
- commit foundation.

## Milestone 1 — Typed representation contract

- ContextTokens;
- explicit masks and identities;
- aggregation protocol;
- compatibility tests.

## Milestone 2 — Structured semantic correction

- feature-token exactness;
- host semantics;
- visibility handling;
- subnet topology masks.

## Milestone 3 — Aggregation architectures

- legacy mean;
- learned-query aggregation;
- token-preserving predictor;
- corrected single-frame target encoder;
- capacity accounting.

## Milestone 4 — One-seed pilot

- four pilot configurations;
- deterministic and behavioral validation;
- memory/performance verification.

## Milestone 5 — Core experiment

- 45 core runs;
- complete artifacts;
- no invalid partial results.

## Milestone 6 — Controls and statistics

- sensitivity controls;
- paired group bootstrap;
- corrected comparisons;
- capacity controls where required.

## Milestone 7 — Policy transfer and paper documentation

- limited policy-transfer sidecars;
- report;
- evidence ledger;
- paper-methodology updates.

Do not stop after writing architecture code. Phase 3 includes implementation, validation, experiment execution, statistical analysis and documentation.

---

# 19. Completion gates

Phase 3 is complete only when:

1. Full pytest passes.
2. Phase 3 changed files pass Ruff.
3. Phase 3 changed files pass mypy.
4. Dataset integrity reports 90,000 one-to-one transition/oracle rows.
5. The fixed cohort exists and has a stable hash.
6. Every configuration uses the same cohort.
7. The deterministic duplicate smoke run passes.
8. The one-seed pilot passes.
9. All 45 core runs reach valid terminal states.
10. Every completed run contains all required artifacts.
11. Per-sample predictions exist.
12. Paired split-group statistics are complete.
13. Capacity differences are reported and controlled.
14. Sensitivity controls are complete.
15. Claims follow the decision rules.
16. Paper documentation and evidence ledger are updated.
17. Final artifacts are committed from a clean worktree.

---

# 20. Final handoff format

Return a complete handoff containing:

1. Direct Phase 3 verdict.
2. Architecture changes.
3. Dataset and cohort integrity evidence.
4. Exact cohort hash.
5. Reproducibility settings.
6. Git commits.
7. Test, Ruff and mypy results.
8. Pilot results.
9. Core run completion table.
10. Parameter-accounting table.
11. Primary and secondary metrics.
12. Paired statistical comparisons.
13. Sensitivity-control results.
14. Policy-transfer results.
15. Latent and attention diagnostics.
16. Supported, rejected and inconclusive claims.
17. Remaining limitations.
18. Recommended Phase 4 question.

Do not state that structured representations were rescued merely because JEPA loss decreased or attention maps appear interpretable.

The conclusion must be determined by:

- paired holdout probe performance;
- changed-target performance;
- action and temporal sensitivity;
- capacity controls;
- reproducibility across seeds;
- group-level statistical evidence.
```