# Humanlike AI plan for this codebase

This build is an RL + memory engine. To push it toward humanlike behavior, you need three things that are not “more parameters”:

1) a persistent self-state that survives long horizons and episode boundaries correctly
2) a world model that supports internal simulation (not just policy logits)
3) a training curriculum that forces planning, language, and social inference, then measures it

Below is a concrete roadmap that plugs into the current `src/infinity_dual_hybrid` stack.

## 0. What you already have

- A backbone (`ssm_backbone.py` and optional `neocortex_backbone.py`) that can carry state through time.
- Dual-tier parametric memory (`miras.py`) for fast + slow weights.
- A PPO trainer that already supports dynamic entropy and recurrent rollouts.
- A “humanlike” hook in `agent.py` that expects a router returning `action` and `entropy_coef`.

The missing piece is: your “humanlike” router must be real and stable, and then you must give it tasks where deliberation matters.

## 1. Make the agent internally consistent

Hard rules you enforce with tests:

- **Episode reset contract**: `agent.reset_episode()` must reset all recurrent state, Miras state (if configured), and humanlike state.
- **Batch shape contract**: policy/value outputs always match `[B, ...]` even during rollout concatenation.
- **Determinism contract**: with fixed seeds, the same checkpoint produces the same evaluation trace.

Add tests:
- memory changes within an episode and resets across episodes
- rollout storage and replayer reproduce action/logprob/value tensors exactly

## 2. Upgrade the “humanlike” layer from heuristics to control

Right now you can do:
- stateful self-latent z (done)
- dynamic exploration coefficient (done)
- cheap counterfactual selection using a proxy model (done, but proxy)

Next upgrades (in order):

### 2.1 Pause = deliberation, not “stall”
In RL environments you can’t stop time, so “pause” becomes:
- reduce entropy (less random exploration)
- run counterfactual selection (more computation per step)
- optionally increase value-head weight (stabilize)

### 2.2 Replace proxy model with a learned world model
Add:
- a learned transition model `p(h_{t+1} | h_t, a_t)` in latent space
- a reward model `r(h_t, a_t)`
- a termination model

Then your counterfactual simulator becomes real: short rollouts in latent space.

File targets:
- `src/infinity_dual_hybrid/world_model.py` (new)
- integrate into `agent.py` to produce `model_step_fn` based on world model

### 2.3 Add an explicit “self-consistency” objective
Humanlike behavior looks coherent because it preserves a stable internal narrative.
Implement a cheap regularizer:
- predict next z from current z and action
- penalize abrupt z jumps unless the environment shifts (detected by prediction error)

## 3. Curriculum that forces humanlike capabilities

If you only train CartPole you only get CartPole tricks. You need tasks where planning and memory are mandatory.

Phase A: memory and control
- DelayedCue (already present)
- T-Maze / Key-Door (add)
- Regime shift (already present)

Phase B: language grounding (minimal)
- instruction-following toy env: text observation + discrete actions
- “referential” tasks where the cue is a sentence

Phase C: social inference (toy)
- multi-agent gridworld with hidden goals, require belief tracking

Each phase produces a validation report with: success rate, sample efficiency, memory usage, and ablations (no Miras / no LTM / no router / no world model).

## 4. Humanlike evaluation metrics you can actually measure

Avoid vibes. Measure:

- **Consistency**: same prompt/state → same plan in evaluation (low entropy)
- **Deliberation gain**: performance difference when counterfactual is enabled vs disabled
- **Long-horizon recall**: performance vs cue delay length
- **Robustness**: performance under regime shift, observation noise
- **Compression**: how much memory you need for the same return

## 5. Concrete implementation sequence (fastest path)

1. Fix env dependency + wrappers (already done in this version).
2. Make `humanlike_core` stable and importable (already done).
3. Add `world_model.py` with a simple GRU latent dynamics model.
4. Wire `world_model` into the router’s `model_step_fn`.
5. Add `scripts/train_delayedcue_worldmodel.py`:
   - pretrain world model from rollouts
   - then train policy with model-based lookahead enabled
6. Add validation script that runs all tasks and writes a single JSON report.

## 6. What “most accurate humanlike” means in practice

If you mean *humanlike conversation*, RL-on-gym will not get you there.
You will need:

- a language model backbone (or external LLM) for text understanding and generation
- long-term memory + persona continuity + tool use
- a value model that scores helpfulness/honesty/consistency
- training data from real dialogues + preference optimization

This repo can still be the control core:
- world model + planner for “thinking”
- memory system for continuity
- router for compute allocation (fast response vs deeper reasoning)

But the “voice” comes from an LLM.

