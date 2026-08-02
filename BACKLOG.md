# Backlog

Deferred work items. Roadmap and phase history live in `docs/OPTIMISATION_PLAN.md`;
the scoring model itself is documented in `docs/SCORING.md`.

---

## Per-item fungibility override within a group

**Priority:** low · **Size:** substantial

Fungibility degree is currently **per-group**. `assign_fungible_group()` in
`allocator/categorizer.py` unpacks `(degree, prefixes, *_rest)` and hands the same
degree to every item matching the group, which `group_concentration_penalty_for_box()`
in `allocator/strategies/_scoring.py` then applies as a linear multiplier on the
group's excess penalty.

That flattens real differences inside a group. Two savoy cabbages are closer
substitutes than a savoy and a red; a cup mushroom and a flat mushroom are nearer
each other than either is to a portobello. One number per group cannot express this.

**Proposal:** treat the group degree as a *default* that individual items may
override — a per-item degree, falling back to the group's when unset. Sketch:

```
"cabbage": [0.7, ["Cabbage"], "portioned", {"Cabbage - Savoy": 0.9}]
```

**Why it's substantial:**

- `Item.fungible_degree` is populated once at build time in `allocator/allocator.py`,
  so the override has to resolve during item construction, not at scoring time.
- The group penalty currently reads a single degree per group key
  (`groups[key] = (capped_qty, degree)`, first-item-wins). Per-item degrees mean
  deciding how a group of mixed-degree items combines — min, mean, or
  weight each item's contribution separately. That's a modelling decision, not
  just a plumbing one.
- The ILP has its own linearised view of concentration and would need the same
  treatment to stay consistent with the scalar path.
- Adds a new class of tunable parameters, expanding the Optuna search space.

**Prerequisite:** decide the combination rule before writing code — it changes what
the penalty means, not just how it's computed.

---

## Group allowances for new groups are untuned

**Priority:** medium (fold into the next retune) · **Size:** small

Ten groups added 2026-08-02 (`watermelon`, `pumpkin`, `mushroom`, `kale`,
`cabbage`, `capsicum`, `cauliflower`, `celery`, `cooking_greens`, `mild_allium`)
plus allowances backfilled for three previously inert ones (`lemon`, `lime`,
`orange`) carry hand-set `group_allowances` derived by analogy to comparable
existing groups — 39 numbers Optuna has never seen. Add them to the search space
at the next retune.

## Periodic roster-vs-config audit

**Priority:** medium · **Size:** small

Item names drift in the DB while config prefixes don't follow. A fungible-group
miss is **silent** — `assign_fungible_group()` returns `(None, 0.0)` with no
warning, unlike `assign_classification()` which logs a fallback. The 2026-08-02
audit found five items silently ungrouped this way (`Tomatoes - Cherry`,
`Lettuce - Baby Cos`, `Orange - Navels`, `Strawberries 500g Punnet`, plus the
Hawkes potato rename).

Worth a small script that runs current `offer_parts` for the produce categories
through `assign_fungible_group()` / `assign_classification()` and reports:
items hitting the classification fallback; items sharing a group's vocabulary
but not assigned to it; config prefixes matching nothing live; and ungrouped
clusters sharing a leading token. Seasonal absences (stone fruit in winter)
produce expected noise and need filtering, not fixing.
