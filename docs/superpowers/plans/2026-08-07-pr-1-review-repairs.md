# PR #1 Review Repairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all confirmed review findings in PR #1 while retaining its frozen-config and two-category feature-matrix contracts.

**Architecture:** The diagnostics requirements manifest becomes the only owner of distribution floors, with lazy bootstrap validation in pytest. Feature guards read the same import-time bindings as extraction, including allowance inputs owned by `_scoring`. Unsupported categories remain schema errors inside the pure extractor but are converted to per-record `[SKIP]` results at the batch boundary.

**Tech Stack:** Python 3.10, pytest 9, `importlib.metadata`, `packaging.version`, SHA-256/JSON configuration guards.

## Global Constraints

- Keep diagnostics dependencies out of production `requirements.txt`.
- Declare `packaging>=22` in `requirements-diagnostics.txt`; retain every existing floor unchanged.
- Ordinary pytest startup and collection must not import `packaging.version`.
- `--strict-diagnostics-deps` must reject absent/below-floor `packaging` and below-floor pytest before collection.
- Unknown names passed to `require_dep()` remain import-only checks.
- Preserve import-time-frozen configuration behavior; do not replace it with live config accessors.
- Hash exactly 13 named effective inputs and snapshot exactly 14 keys, including both category IDs.
- Catch only `UnsupportedCategoryError` at the extraction batch boundary; unrelated exceptions propagate.
- Preserve the DB-free import contract of `allocator.box_features`.
- Use synthetic fixtures only; tests perform no DB or network operations.

---

### Task 1: Manifest-Owned Diagnostic Dependency Gating

**Files:**
- Modify: `requirements-diagnostics.txt`
- Modify: `tests/conftest.py:11-18,258-325`
- Modify: `tests/test_box_features.py:28-172`

**Interfaces:**
- Consumes: `requirements-diagnostics.txt` lines in the exact form `distribution>=minimum`.
- Produces:
  ```python
  def _load_diagnostic_dependencies(path: Path) -> dict[str, tuple[str, str]]
  def require_dep(name: str):
      """Import a module and enforce its manifest floor when declared."""
  ```
- `_DIAGNOSTIC_DEPENDENCIES` is derived once from the manifest. The only exceptional module mapping is `scikit-learn` to `sklearn`.

- [ ] **Step 1: Add regression tests for manifest parsing and lazy version imports**

Replace `test_declared_dependency_floors_and_distribution_names_are_exact` with behavior tests that use a temporary manifest:

```python
def test_dependency_floors_are_loaded_from_the_manifest(tmp_path):
    from tests.conftest import _load_diagnostic_dependencies

    manifest = tmp_path / "requirements-diagnostics.txt"
    manifest.write_text(
        "# diagnostic stack\n"
        "example-lib>=1.2.3\n"
        "scikit-learn>=9.8.7\n"
    )

    assert _load_diagnostic_dependencies(manifest) == {
        "example_lib": ("example-lib", "1.2.3"),
        "sklearn": ("scikit-learn", "9.8.7"),
    }


def test_unknown_dependency_does_not_import_packaging(monkeypatch):
    from tests import conftest

    real_import = conftest.importlib.import_module
    imported = []

    def recording_import(name):
        imported.append(name)
        return real_import(name)

    monkeypatch.setattr(conftest.importlib, "import_module", recording_import)
    assert require_dep("json").loads("{}") == {}
    assert "packaging.version" not in imported
```

Add `sitecustomize.py` support to `_collect_module_scope_dependency()` so subprocesses can override distribution metadata before pytest imports the project plugin:

```python
def _collect_module_scope_dependency(
    tmp_path, *pytest_args, dependency="a_module_that_does_not_exist_xyz",
    version_overrides=None, absent_distributions=(),
):
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "import json, os\n"
        "from importlib import metadata\n"
        "_real_version = metadata.version\n"
        "_overrides = json.loads(os.environ.get('DIAGNOSTIC_VERSION_OVERRIDES', '{}'))\n"
        "_absent = set(json.loads(os.environ.get('DIAGNOSTIC_ABSENT_DISTRIBUTIONS', '[]')))\n"
        "def _version(name):\n"
        "    if name in _absent:\n"
        "        raise metadata.PackageNotFoundError(name)\n"
        "    return _overrides.get(name, _real_version(name))\n"
        "metadata.version = _version\n"
    )
    test_module = tmp_path / "test_diagnostic_dependency.py"
    test_module.write_text(
        "import pytest\n"
        "from tests.conftest import require_dep\n"
        "pytestmark = pytest.mark.diagnostics(reason='synthetic')\n"
        f"require_dep({dependency!r})\n"
        "def test_reached(): pass\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(_PROJECT_ROOT)))
    env["DIAGNOSTIC_VERSION_OVERRIDES"] = json.dumps(version_overrides or {})
    env["DIAGNOSTIC_ABSENT_DISTRIBUTIONS"] = json.dumps(absent_distributions)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-c", str(_PROJECT_ROOT / "pyproject.toml"),
         "-p", "tests.conftest", *pytest_args, str(test_module)],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
```

Add subprocess assertions:

```python
def test_plain_pytest_skips_when_packaging_distribution_is_absent(tmp_path):
    proc = _collect_module_scope_dependency(
        tmp_path, dependency="packaging", absent_distributions=["packaging"]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 skipped" in proc.stdout


@pytest.mark.parametrize("distribution, found", [
    ("packaging", "21.9"),
    ("pytest", "8.4.2"),
])
def test_strict_mode_rejects_below_floor_bootstrap_dependencies(
    tmp_path, distribution, found,
):
    proc = _collect_module_scope_dependency(
        tmp_path,
        "-m", "diagnostics", "--strict-diagnostics-deps",
        dependency="json",
        version_overrides={distribution: found},
    )
    assert proc.returncode != 0
    output = proc.stdout + proc.stderr
    assert f"{distribution}>=" in output
    assert f"found {found}" in output
```

Before writing these tests, name their mutations: hardcoded floors should fail the temporary-manifest test; a top-level `Version` import should fail the import-recording test; removing strict bootstrap validation should make each strict subprocess unexpectedly exit zero.

- [ ] **Step 2: Run the dependency tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_box_features.py -v -k "dependency or packaging or bootstrap"
```

Expected failures: `_load_diagnostic_dependencies` is absent; the current hardcoded dictionary ignores the temporary manifest; strict startup does not validate bootstrap floors.

- [ ] **Step 3: Make the manifest authoritative and add lazy bootstrap validation**

Add `packaging>=22` immediately after `pytest>=9.0.0` in `requirements-diagnostics.txt`.

Remove `from packaging.version import Version` from module scope. Replace the hardcoded dependency dictionary with:

```python
_MODULE_NAME_OVERRIDES = {"scikit-learn": "sklearn"}
_BOOTSTRAP_DEPENDENCIES = ("packaging", "pytest")


def _load_diagnostic_dependencies(
    path: Path = _PROJECT_ROOT / "requirements-diagnostics.txt",
) -> dict[str, tuple[str, str]]:
    dependencies = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.partition("#")[0].strip()
        if not line:
            continue
        distribution, separator, minimum = line.partition(">=")
        distribution, minimum = distribution.strip(), minimum.strip()
        if not separator or not distribution or not minimum:
            raise ValueError(
                f"{path}:{line_number}: expected distribution>=minimum"
            )
        module = _MODULE_NAME_OVERRIDES.get(
            distribution, distribution.replace("-", "_")
        )
        dependencies[module] = (distribution, minimum)
    return dependencies


_DIAGNOSTIC_DEPENDENCIES = _load_diagnostic_dependencies()
```

Factor the common policy and comparison:

```python
def _unavailable(message: str, cause: BaseException | None = None):
    if _STRICT:
        raise ImportError(message) from cause
    pytest.skip(message, allow_module_level=True)


def _require_version(distribution: str, minimum: str) -> None:
    try:
        installed = importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError as exc:
        _unavailable(f"required diagnostic distribution {distribution!r} is absent", exc)
    try:
        version_module = importlib.import_module("packaging.version")
    except ImportError as exc:
        _unavailable("required diagnostic dependency 'packaging' is not importable", exc)
    if version_module.Version(installed) < version_module.Version(minimum):
        _unavailable(
            f"required diagnostic dependency {distribution}>={minimum}; found {installed}"
        )
```

Update `require_dep()` to return immediately for unknown names and otherwise call `_require_version(distribution, minimum)`. In `pytest_configure()` set `_STRICT` first, then validate bootstrap modules only in strict mode:

```python
def pytest_configure(config):
    global _STRICT
    _STRICT = config.getoption("--strict-diagnostics-deps")
    if _STRICT:
        for name in _BOOTSTRAP_DEPENDENCIES:
            require_dep(name)
```

Validation order is `packaging`, then `pytest`: an absent packaging distribution is reported without trying to compare pytest first, while an installed below-floor packaging can still supply `Version` to compare its own version.

- [ ] **Step 4: Run dependency tests and verify GREEN**

Run:

```bash
python3 -m pytest tests/test_box_features.py -v -k "dependency or packaging or bootstrap"
```

Expected: all selected tests pass, with subprocess outputs containing the declared floors from the manifest.

- [ ] **Step 5: Commit the dependency-gating slice**

```bash
git add requirements-diagnostics.txt tests/conftest.py tests/test_box_features.py
git commit -m "fix: derive diagnostic floors from manifest"
```

---

### Task 2: Effective Configuration Identity

**Files:**
- Modify: `allocator/box_features.py:12-29,302-358`
- Modify: `tests/test_box_features.py:184-253,706-875`

**Interfaces:**
- Consumes: frozen local bindings in `allocator.box_features`; allowance bindings in `allocator.strategies._scoring`.
- Produces:
  ```python
  def config_hash() -> str       # 16 hex characters over 13 named inputs
  def config_snapshot() -> dict  # exact 14-key JSON-round-trippable object
  ```
- Snapshot adds integer values `category_fruit` and `category_vegetables`.

- [ ] **Step 1: Write failing identity-owner and category tests**

Update the exact key test to include `category_fruit` and `category_vegetables`. Replace mutation tests that patch `allocator.config` with tests against the effective owners:

```python
@pytest.mark.parametrize("name", ["CATEGORY_FRUIT", "CATEGORY_VEGETABLES"])
def test_category_id_changes_invalidate_both_identity_contracts(monkeypatch, name):
    import allocator.box_features as box_features

    before_hash = box_features.config_hash()
    before_snapshot = box_features.config_snapshot()
    monkeypatch.setattr(box_features, name, getattr(box_features, name) + 1000)
    after_snapshot = box_features.config_snapshot()

    assert box_features.config_hash() != before_hash
    assert after_snapshot != before_snapshot
    assert after_snapshot[
        "category_fruit" if name == "CATEGORY_FRUIT" else "category_vegetables"
    ] == getattr(box_features, name)


def test_live_config_and_categorizer_reassignment_do_not_change_frozen_guards(
    monkeypatch,
):
    import allocator.box_features as box_features
    import allocator.categorizer as categorizer
    import allocator.config as config

    before = (box_features.config_hash(), box_features.config_snapshot())
    monkeypatch.setattr(config, "CATEGORY_FRUIT", config.CATEGORY_FRUIT + 1000)
    monkeypatch.setattr(config, "BOX_TIERS", {"changed": {"price": 1, "target_value": 1}})
    monkeypatch.setattr(categorizer, "DEFAULT_CLASSIFICATION", ("changed",) * 4)
    assert (box_features.config_hash(), box_features.config_snapshot()) == before
```

Add one behavioral test per `_scoring` owner. Each test must assert both the extracted allowance and the identity change. For example, quantity classes:

```python
def test_quantity_class_owner_changes_allowance_hash_and_snapshot(monkeypatch):
    import allocator.box_features as box_features
    import allocator.strategies._scoring as scoring

    before_hash = box_features.config_hash()
    before_snapshot = box_features.config_snapshot()
    classes = {name: dict(values) for name, values in scoring.QUANTITY_CLASSES.items()}
    classes["snack_piece"]["small"] += 7
    monkeypatch.setattr(scoring, "QUANTITY_CLASSES", classes)

    record = box_features.extract_box_features(
        "x", {1: 1}, _item_lookup(), "small", _available_tags(), 1
    )
    assert record["item_quantities"][0][2] == classes["snack_piece"]["small"]
    assert box_features.config_hash() != before_hash
    assert box_features.config_snapshot() != before_snapshot
```

Use the existing apple fixture to change `FUNGIBLE_GROUPS["apple"]` to a different quantity class. Use an ungrouped snacking fixture priced at the boundary to change `QTY_CLASS_PRICE_THRESHOLDS["snacking_max"]`. Do not assert only on guard text: each mutation must change `item_quantities` as well.

Before writing, name the mutations: reading `_cfg.CATEGORY_FRUIT` should fail the category-owner test; reading live `_cfg`/`_categorizer` should fail the frozen-owner test; hashing config copies instead of `_scoring` values should fail the paired allowance tests.

- [ ] **Step 2: Run identity tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_box_features.py -v -k "config_hash or config_snapshot or owner"
```

Expected failures: category IDs are missing, live-module reassignment changes guards, and `_scoring` mutations are absent from guard inputs.

- [ ] **Step 3: Align guard inputs with their effective owners**

In `allocator.box_features`, remove `import allocator.config as _cfg`, remove `import allocator.categorizer as _categorizer`, and import these additional local bindings from config:

```python
BOX_TARGET_PCT,
VALUE_PENALTY_EXPONENT,
VALUE_SWEET_FROM,
VALUE_SWEET_TO,
```

Import the scoring module and call the helper through it:

```python
from allocator.strategies import _scoring

# in extract_box_features()
item_allowance = _scoring._resolve_item_allowance_from_lookup(info, tier)
```

Replace `_HASH_INPUTS` and live lookups with an explicit effective-input function:

```python
def _hash_inputs() -> dict:
    return {
        "BOX_TIERS": BOX_TIERS,
        "GROUP_ALLOWANCES": GROUP_ALLOWANCES,
        "ITEM_CLASSIFICATIONS": ITEM_CLASSIFICATIONS,
        "FUNGIBLE_GROUPS": _scoring.FUNGIBLE_GROUPS,
        "CLASSIFICATION_FALLBACK": CLASSIFICATION_FALLBACK,
        "QUANTITY_CLASSES": _scoring.QUANTITY_CLASSES,
        "QTY_CLASS_PRICE_THRESHOLDS": _scoring.QTY_CLASS_PRICE_THRESHOLDS,
        "VALUE_SWEET_FROM": VALUE_SWEET_FROM,
        "VALUE_SWEET_TO": VALUE_SWEET_TO,
        "VALUE_PENALTY_EXPONENT": VALUE_PENALTY_EXPONENT,
        "CATEGORY_FRUIT": CATEGORY_FRUIT,
        "CATEGORY_VEGETABLES": CATEGORY_VEGETABLES,
        "DEFAULT_CLASSIFICATION": DEFAULT_CLASSIFICATION,
    }


def config_hash() -> str:
    return _digest(_hash_inputs())
```

Build the snapshot from the same owners:

```python
return {
    "box_tiers": {
        tier: {"price": entry["price"], "target_value": entry["target_value"]}
        for tier, entry in BOX_TIERS.items()
    },
    "box_target_pct": BOX_TARGET_PCT,
    "category_fruit": CATEGORY_FRUIT,
    "category_vegetables": CATEGORY_VEGETABLES,
    "value_sweet_from": VALUE_SWEET_FROM,
    "value_sweet_to": VALUE_SWEET_TO,
    "value_penalty_exponent": VALUE_PENALTY_EXPONENT,
    "group_allowances": GROUP_ALLOWANCES,
    "quantity_classes": _scoring.QUANTITY_CLASSES,
    "qty_class_price_thresholds": _scoring.QTY_CLASS_PRICE_THRESHOLDS,
    "item_classifications_hash": _digest(ITEM_CLASSIFICATIONS),
    "fungible_groups_hash": _digest(_scoring.FUNGIBLE_GROUPS),
    "classification_fallback_hash": _digest(CLASSIFICATION_FALLBACK),
    "default_classification_hash": _digest(DEFAULT_CLASSIFICATION),
}
```

Read `FUNGIBLE_GROUPS`, `QUANTITY_CLASSES`, and `QTY_CLASS_PRICE_THRESHOLDS` only through `_scoring`; do not add direct imports for them.

- [ ] **Step 4: Run identity tests and verify GREEN**

Run:

```bash
python3 -m pytest tests/test_box_features.py -v -k "config_hash or config_snapshot or owner"
```

Expected: all selected tests pass; the snapshot contains exactly 14 keys and round-trips through JSON.

- [ ] **Step 5: Commit the identity slice**

```bash
git add allocator/box_features.py tests/test_box_features.py
git commit -m "fix: hash effective box feature configuration"
```

---

### Task 3: Per-Record Unsupported-Category Skipping

**Files:**
- Modify: `allocator/box_features.py:32-44,176-188`
- Modify: `scripts/extract_features.py:28-51,54-164,246-252`
- Modify: `tests/test_box_features.py:535-566` and add batch-boundary tests

**Interfaces:**
- Produces:
  ```python
  class UnsupportedCategoryError(ValueError): ...
  def _extract_or_skip(*args, **kwargs) -> dict | None
  ```
- `_extract_or_skip` prints `  [SKIP] {error}` only for `UnsupportedCategoryError`; it returns `None` for that record.

- [ ] **Step 1: Write failing exception and batch-continuation tests**

Strengthen the direct extractor test:

```python
def test_a_third_category_raises_compatible_specific_error():
    from allocator.box_features import UnsupportedCategoryError, extract_box_features

    lookup = _item_lookup()
    lookup[4] = {**lookup[1], "name": "Mystery Herb", "price": 500,
                 "category_id": 987654, "fungible_group": None,
                 "fungible_degree": 0.0}
    with pytest.raises(UnsupportedCategoryError, match="neither CATEGORY_FRUIT") as exc:
        extract_box_features(
            "x", {1: 3, 3: 2, 4: 1}, lookup, "small", _available_tags(), 1
        )
    assert isinstance(exc.value, ValueError)
```

Exercise the real wrapper twice to prove a later record is still extractable:

```python
def test_batch_boundary_skips_unsupported_record_and_allows_next(capsys):
    from scripts.extract_features import _extract_or_skip

    lookup = _item_lookup()
    lookup[4] = {**lookup[1], "category_id": 987654,
                 "fungible_group": None, "fungible_degree": 0.0}
    skipped = _extract_or_skip(
        "bad", {4: 1}, lookup, "small", _available_tags(), 10
    )
    accepted = _extract_or_skip(
        "good", {1: 1}, lookup, "small", _available_tags(), 10
    )

    assert skipped is None
    assert accepted["box_name"] == "good"
    assert "[SKIP]" in capsys.readouterr().out
```

Prove synthetic generation continues after one rejected candidate by monkeypatching only the extraction boundary, returning a real-shaped sentinel for the next named candidate and `None` for the rest:

```python
def test_synthetic_generation_continues_after_unsupported_candidate(monkeypatch):
    import scripts.extract_features as script
    from allocator.box_features import UnsupportedCategoryError

    def selective_extract(box_name, *args, **kwargs):
        if box_name == "synth_mono_small":
            raise UnsupportedCategoryError("unsupported synthetic")
        if box_name == "synth_random_small":
            return {"box_name": box_name, "source": "synth_random"}
        return None

    monkeypatch.setattr(script, "extract_box_features", selective_extract)
    features = script.generate_synthetic_boxes(1, _item_lookup(), _available_tags())
    assert features == [{"box_name": "synth_random_small", "source": "synth_random"}]
```

Prove exception selectivity:

```python
def test_batch_boundary_does_not_swallow_unrelated_value_error(monkeypatch):
    import scripts.extract_features as script

    def broken(*args, **kwargs):
        raise ValueError("unrelated")

    monkeypatch.setattr(script, "extract_box_features", broken)
    with pytest.raises(ValueError, match="unrelated"):
        script._extract_or_skip("x", {}, {}, "small", {}, 1)
```

Before writing, name the mutations: raising base `ValueError` should fail the specific-error test; catching all `ValueError` should fail exception selectivity; leaving any synthetic call on the raw extractor should abort the continuation test when that name is selected.

- [ ] **Step 2: Run category-boundary tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_box_features.py -v -k "third_category or unsupported or batch_boundary or synthetic_generation"
```

Expected failures: `UnsupportedCategoryError` and `_extract_or_skip` do not exist; synthetic generation propagates the injected error.

- [ ] **Step 3: Add the specific error and central batch wrapper**

In `allocator.box_features.py`, add before `extract_box_features`:

```python
class UnsupportedCategoryError(ValueError):
    """A resolved item cannot fit the two-category feature schema."""
```

Raise `UnsupportedCategoryError` at the existing guard without changing its detailed message.

In `scripts/extract_features.py`, import both symbols and add:

```python
from allocator.box_features import UnsupportedCategoryError, extract_box_features


def _extract_or_skip(*args, **kwargs) -> dict | None:
    try:
        return extract_box_features(*args, **kwargs)
    except UnsupportedCategoryError as exc:
        print(f"  [SKIP] {exc}")
        return None
```

Replace all five synthetic calls and the manual call with `_extract_or_skip`. Remove unused `CATEGORY_FRUIT`, `CATEGORY_VEGETABLES`, and `GROUP_ALLOWANCES` imports. Keep `BOX_TIERS`, `FUNGIBLE_GROUPS`, and `QUANTITY_CLASSES`, which synthetic generation uses.

- [ ] **Step 4: Run category-boundary tests and verify GREEN**

Run:

```bash
python3 -m pytest tests/test_box_features.py -v -k "third_category or unsupported or batch_boundary or synthetic_generation"
```

Expected: all selected tests pass; captured output contains `[SKIP]`; unrelated `ValueError` still propagates.

- [ ] **Step 5: Run the complete box-feature module**

Run:

```bash
python3 -m pytest tests/test_box_features.py -v
```

Expected: all tests pass with no unexpected warnings.

- [ ] **Step 6: Commit the category-boundary slice**

```bash
git add allocator/box_features.py scripts/extract_features.py tests/test_box_features.py
git commit -m "fix: skip unsupported category feature records"
```

---

### Task 4: Current and Authoritative Documentation

**Files:**
- Modify in PR worktree: `CLAUDE.md:44-77,163-166`
- Modify in planning workspace: `/home/mark/jointly/Mystery-Manager/docs/superpowers/specs/2026-08-03-ebm-ordinal-diagnostics-design.md`
- Modify in planning workspace: `/home/mark/jointly/Mystery-Manager/docs/superpowers/plans/2026-08-04-diagnostics-foundation.md`

**Interfaces:**
- The PR documentation describes only commands that exist now and the exact 14-key snapshot.
- The planning documents enumerate the same 9-package manifest, 13-input hash, and 14-key snapshot as implementation.

- [ ] **Step 1: Correct current repository documentation**

In `CLAUDE.md`, keep the install command but remove the nonexistent script invocation:

```bash
pip install --target .venv-diagnostics/lib -r requirements-diagnostics.txt
```

Keep the strict diagnostic pytest command unchanged. Change the key-module description from “exact 12-key `config_snapshot()`” to “exact 14-key `config_snapshot()`”.

- [ ] **Step 2: Synchronize the parent diagnostics design in the planning workspace**

In the Part 4 dependency block, insert:

```text
packaging>=22
```

immediately after `pytest>=9.0.0`, and document both as bootstrap dependencies validated at strict pytest startup.

In the schema-guard section, add `CATEGORY_FRUIT` and `CATEGORY_VEGETABLES` to the enumerated hash inputs and change every “eleven named inputs”/“11-input” statement to 13.

In the snapshot JSON object, add:

```json
"category_fruit": 0,
"category_vegetables": 0,
```

after `box_target_pct`, and change every exact twelve-key statement to fourteen keys. Replace the claim that `packaging.version.Version` is merely “available with pytest” with the explicit `packaging>=22` manifest/bootstrap contract.

- [ ] **Step 3: Synchronize the foundation plan in the planning workspace**

Apply the same manifest, hash-input, snapshot-key, code-example, test-expectation, and prose-count changes throughout `2026-08-04-diagnostics-foundation.md`. Its Task 1 dependency file example must list `packaging>=22`; its Task 5 interfaces must say 13 inputs and 14 keys; its `_HASH_INPUTS` and snapshot examples must include both category IDs and effective-owner semantics.

- [ ] **Step 4: Verify documentation consistency**

Run:

```bash
rg -n "ebm_diagnostic.py|12-key|twelve keys|eleven named inputs|11-input|packaging.version.Version \(available with pytest\)" CLAUDE.md
rg -n "pytest>=9.0.0|packaging>=22|CATEGORY_FRUIT|CATEGORY_VEGETABLES|category_fruit|category_vegetables|13-input|14-key|fourteen keys" /home/mark/jointly/Mystery-Manager/docs/superpowers/specs/2026-08-03-ebm-ordinal-diagnostics-design.md /home/mark/jointly/Mystery-Manager/docs/superpowers/plans/2026-08-04-diagnostics-foundation.md
git diff --check
```

Expected: the first command returns no matches in `CLAUDE.md`; the planning documents show `packaging>=22`, both category IDs, 13-input hash wording, and 14-key snapshot wording. Review numeric matches such as dataset counts separately; do not mechanically replace unrelated numbers 11, 12, 13, or 14.

- [ ] **Step 5: Commit PR documentation and record planning-workspace state separately**

In the PR worktree:

```bash
git add CLAUDE.md
git commit -m "docs: correct diagnostics foundation commands"
```

The two authoritative companion files are intentionally outside PR #1 and ignored by this repository. Preserve their edits in the planning workspace; do not force-add them to the PR branch. Report their exact paths in the final handoff so they are not mistaken for PR changes.

---

### Task 5: Final Verification and Review-Finding Audit

**Files:**
- Verify only; no new implementation files.

**Interfaces:**
- Produces fresh evidence that each of the seven review findings is resolved or intentionally handled.

- [ ] **Step 1: Run focused diagnostics-foundation tests**

```bash
python3 -m pytest tests/test_box_features.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Run the complete suite**

```bash
python3 -m pytest -q
```

Expected: zero failures and zero collection errors.

- [ ] **Step 3: Run static and import checks**

```bash
python3 -m py_compile allocator/box_features.py scripts/extract_features.py tests/conftest.py tests/test_box_features.py
python3 -c "import allocator.box_features; print('DB-free import passed')"
git diff main...HEAD --check
git status --short
```

Expected: compilation and DB-free import exit zero; diff check emits nothing; status contains only the separately managed planning-document edits outside this worktree, not uncommitted PR files.

- [ ] **Step 4: Audit each review item against the resulting diff**

Confirm all seven outcomes explicitly:

1. no top-level `packaging` import and `packaging>=22` is declared;
2. `CLAUDE.md` names no nonexistent diagnostic script;
3. both category IDs participate in hash and snapshot;
4. guards read the same frozen owners as feature production;
5. unsupported categories skip one record while unrelated errors propagate;
6. dependency floors occur only in the manifest, with code deriving them;
7. the three unused script imports are removed.

- [ ] **Step 5: Request code review before integration**

Use `superpowers:requesting-code-review` on `main...HEAD`. Address any confirmed findings with `superpowers:receiving-code-review`, then repeat Steps 1–4 before reporting completion.
