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
- Modify: `tests/conftest.py` diagnostics dependency gating section
- Modify: `tests/test_box_features.py` dependency-gating helpers and tests

Tasks in this plan are strictly sequential. Symbol names, not inherited line
numbers, identify edit locations after earlier tasks shift the files.

**Interfaces:**
- Consumes: `requirements-diagnostics.txt` lines in the exact form `distribution>=minimum`.
- Produces:
  ```python
  _MODULE_NAME_OVERRIDES = {"scikit-learn": "sklearn"}
  _BOOTSTRAP_DEPENDENCIES = ("packaging", "pytest")
  _RUNNING_VERSION_MODULES = {"pytest"}
  _DIAGNOSTIC_REQUIREMENT: re.Pattern

  def _load_diagnostic_dependencies(
      path: Path = _PROJECT_ROOT / "requirements-diagnostics.txt",
      *, strict: bool = False,
  ) -> dict[str, tuple[str, str]]
  def _unavailable(message: str, cause: BaseException | None = None): ...
  def _require_version(
      distribution: str, minimum: str, *, installed: str | None = None,
  ) -> None
  def _running_version(name: str, module) -> str | None
  def _validate_bootstrap_dependencies(
      dependencies: dict[str, tuple[str, str]],
  ) -> None
  def require_dep(name: str):
      """Import a module and enforce its manifest floor when declared."""
  ```
- `_DIAGNOSTIC_DEPENDENCIES` is derived once from the manifest. The only exceptional module mapping is `scikit-learn` to `sklearn`.
- Because pytest becomes a declared manifest entry, `require_dep("pytest")`
  changes from an import-only check to a floor-enforcing check against the
  running module's `__version__`.

- [ ] **Step 1: Add regression tests for manifest parsing and lazy version imports**

Replace `test_declared_dependency_floors_and_distribution_names_are_exact`
with behavior tests that use a temporary manifest while retaining explicit
bootstrap-presence coverage:

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


def test_production_manifest_declares_every_diagnostic_module():
    from tests.conftest import _DIAGNOSTIC_DEPENDENCIES

    assert set(_DIAGNOSTIC_DEPENDENCIES) == {
        "pytest", "packaging", "interpret", "statsmodels", "sklearn",
        "numpy", "pandas", "numexpr", "bottleneck",
    }
    assert _DIAGNOSTIC_DEPENDENCIES["sklearn"][0] == "scikit-learn"


def test_manifest_parser_is_tolerant_for_plain_collection_but_strict_on_demand(
    tmp_path,
):
    from tests.conftest import _load_diagnostic_dependencies

    manifest = tmp_path / "requirements-diagnostics.txt"
    manifest.write_text(
        "valid>=1.0\n"
        "conditional>=2; python_version >= '3.10'\n"
        "pinned==3\n"
    )

    assert _load_diagnostic_dependencies(manifest) == {
        "valid": ("valid", "1.0")
    }
    with pytest.raises(ValueError, match="expected distribution>=minimum"):
        _load_diagnostic_dependencies(manifest, strict=True)


def test_missing_manifest_is_empty_in_plain_mode_and_errors_in_strict_mode(
    tmp_path,
):
    from tests.conftest import _load_diagnostic_dependencies

    missing = tmp_path / "missing-requirements.txt"
    assert _load_diagnostic_dependencies(missing) == {}
    with pytest.raises(ValueError, match="missing-requirements.txt.*does not exist"):
        _load_diagnostic_dependencies(missing, strict=True)


def test_bootstrap_dependency_must_be_present_in_manifest():
    from tests.conftest import _validate_bootstrap_dependencies

    with pytest.raises(pytest.UsageError, match="packaging.*manifest"):
        _validate_bootstrap_dependencies({
            "pytest": ("pytest", "9.0.0"),
        })
```

Add a fresh-interpreter regression that imports pytest first, removes its
already-loaded `packaging.version`, blocks any re-import, and executes the root
conftest as a module. This fails if a top-level
`from packaging.version import Version` returns:

```python
def test_conftest_does_not_import_packaging_version_at_module_scope():
    code = r'''\
import importlib.abc
import runpy
import sys
import packaging
import pytest

sys.modules.pop("packaging.version", None)
if hasattr(packaging, "version"):
    delattr(packaging, "version")

class BlockPackagingVersion(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "packaging.version":
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockPackagingVersion())
runpy.run_path("tests/conftest.py", run_name="conftest_import_probe")
'''
    provisioned_paths = [
        _PROJECT_ROOT / "scoring_config.json",
        _PROJECT_ROOT / "identifiers.json",
    ]
    absent_before = {path for path in provisioned_paths if not path.exists()}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
    finally:
        for path in absent_before:
            if path.exists():
                path.unlink()
    assert proc.returncode == 0, proc.stdout + proc.stderr
```

Add `sitecustomize.py` support to `_collect_module_scope_dependency()` so subprocesses can override distribution metadata before pytest imports the project plugin:

```python
def _collect_module_scope_dependency(
    tmp_path, *pytest_args, dependency="a_module_that_does_not_exist_xyz",
    version_overrides=None, absent_distributions=(), running_pytest_version=None,
):
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "import json, os\n"
        "from importlib import metadata\n"
        "_real_version = metadata.version\n"
        "_overrides = json.loads(os.environ.get('DIAGNOSTIC_VERSION_OVERRIDES', '{}'))\n"
        "_absent = set(json.loads(os.environ.get('DIAGNOSTIC_ABSENT_DISTRIBUTIONS', '[]')))\n"
        "_running_pytest = os.environ.get('DIAGNOSTIC_RUNNING_PYTEST_VERSION')\n"
        "if _running_pytest:\n"
        "    import pytest\n"
        "    pytest.__version__ = _running_pytest\n"
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
        "def test_unreachable():\n"
        "    raise AssertionError('dependency gate did not stop collection')\n"
    )
    (tmp_path / "test_sentinel.py").write_text(
        "import pytest\n"
        "pytestmark = pytest.mark.diagnostics(reason='synthetic')\n"
        "def test_pytest_completed_collection(): pass\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(_PROJECT_ROOT)))
    env["DIAGNOSTIC_VERSION_OVERRIDES"] = json.dumps(version_overrides or {})
    env["DIAGNOSTIC_ABSENT_DISTRIBUTIONS"] = json.dumps(absent_distributions)
    if running_pytest_version is not None:
        env["DIAGNOSTIC_RUNNING_PYTEST_VERSION"] = running_pytest_version
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-c", str(_PROJECT_ROOT / "pyproject.toml"),
         "-p", "tests.conftest", *pytest_args, str(tmp_path)],
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


def test_strict_mode_rejects_absent_packaging_distribution(tmp_path):
    proc = _collect_module_scope_dependency(
        tmp_path,
        "-m", "diagnostics", "--strict-diagnostics-deps",
        dependency="json",
        absent_distributions=["packaging"],
    )
    assert proc.returncode == pytest.ExitCode.USAGE_ERROR
    assert "packaging" in proc.stdout + proc.stderr
    assert "absent" in proc.stdout + proc.stderr


def test_strict_mode_rejects_below_floor_packaging(tmp_path):
    proc = _collect_module_scope_dependency(
        tmp_path,
        "-m", "diagnostics", "--strict-diagnostics-deps",
        dependency="json",
        version_overrides={"packaging": "21.9"},
    )
    assert proc.returncode == pytest.ExitCode.USAGE_ERROR
    output = proc.stdout + proc.stderr
    assert "packaging>=22" in output
    assert "found 21.9" in output


def test_strict_mode_checks_the_running_pytest_version(tmp_path):
    proc = _collect_module_scope_dependency(
        tmp_path,
        "-m", "diagnostics", "--strict-diagnostics-deps",
        dependency="json",
        version_overrides={"pytest": "99.0.0"},
        running_pytest_version="8.4.2",
    )
    assert proc.returncode == pytest.ExitCode.USAGE_ERROR
    output = proc.stdout + proc.stderr
    assert "pytest>=9.0.0" in output
    assert "found 8.4.2" in output
```

Preserving `test_sentinel.py` and targeting `str(tmp_path)` keeps the existing
`test_marker_keyword_selection_does_not_arm_dependency_strictness` expectation
of `1 passed, 1 skipped`. Before writing these tests, name their mutations:
hardcoded floors fail the temporary-manifest test; a top-level `Version` import
fails the fresh-interpreter blocker; deleting either bootstrap manifest line
fails the presence test; using distribution metadata for pytest fails the
running-version subprocess; removing strict bootstrap validation makes strict
subprocesses unexpectedly exit zero.

- [ ] **Step 2: Run the dependency tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_box_features.py -v -k "require_dep or dependency or packaging or bootstrap or conftest or marker_keyword"
```

Expected failures: `_load_diagnostic_dependencies` and
`_validate_bootstrap_dependencies` are absent; the current hardcoded dictionary
ignores the temporary manifest; strict startup does not validate bootstrap
floors.

- [ ] **Step 3: Make the manifest authoritative and add lazy bootstrap validation**

Add `packaging>=22` immediately after `pytest>=9.0.0` in `requirements-diagnostics.txt`.

Remove `from packaging.version import Version` from module scope, add `import re`,
and replace the hardcoded dependency dictionary with:

```python
_MODULE_NAME_OVERRIDES = {"scikit-learn": "sklearn"}
_BOOTSTRAP_DEPENDENCIES = ("packaging", "pytest")
_RUNNING_VERSION_MODULES = {"pytest"}
_DIAGNOSTIC_REQUIREMENT = re.compile(
    r"^(?P<distribution>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r">=(?P<minimum>[0-9]+(?:\.[0-9]+)*)$"
)


def _load_diagnostic_dependencies(
    path: Path = _PROJECT_ROOT / "requirements-diagnostics.txt",
    *,
    strict: bool = False,
) -> dict[str, tuple[str, str]]:
    dependencies = {}
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError as exc:
        if strict:
            raise ValueError(f"{path} does not exist") from exc
        return {}
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.partition("#")[0].strip()
        if not line:
            continue
        match = _DIAGNOSTIC_REQUIREMENT.fullmatch(line)
        if match is None:
            if strict:
                raise ValueError(
                    f"{path}:{line_number}: expected distribution>=minimum"
                )
            continue
        distribution = match.group("distribution")
        minimum = match.group("minimum")
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


def _require_version(
    distribution: str,
    minimum: str,
    *,
    installed: str | None = None,
) -> None:
    if installed is None:
        try:
            installed = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            _unavailable(
                f"required diagnostic distribution {distribution!r} is absent", exc
            )
    try:
        from packaging.version import Version
    except ImportError as exc:
        _unavailable("required diagnostic dependency 'packaging' is not importable", exc)
    if Version(installed) < Version(minimum):
        _unavailable(
            f"required diagnostic dependency {distribution}>={minimum}; found {installed}"
        )


def _running_version(name: str, module) -> str | None:
    return module.__version__ if name in _RUNNING_VERSION_MODULES else None
```

Update `require_dep()` with the complete flow below. It returns immediately for
unknown names and passes `module.__version__` only when `name == "pytest"`; all
other modules use distribution metadata:

```python
def require_dep(name: str):
    """Import `name` and enforce its manifest floor when one is declared."""
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        _unavailable(
            f"required diagnostic dependency {name!r} is not importable", exc
        )

    requirement = _DIAGNOSTIC_DEPENDENCIES.get(name)
    if requirement is None:
        return module
    distribution, minimum = requirement
    installed = _running_version(name, module)
    _require_version(distribution, minimum, installed=installed)
    return module
```

The lazy `Version` import uses an import statement, not
`conftest.importlib.import_module`, preserving the three existing tests that
monkeypatch that function wholesale.

Add strict bootstrap validation that accepts an explicit dependency mapping for
direct testing and converts startup failures into concise pytest usage errors:

```python
def _validate_bootstrap_dependencies(
    dependencies: dict[str, tuple[str, str]],
) -> None:
    for name in _BOOTSTRAP_DEPENDENCIES:
        requirement = dependencies.get(name)
        if requirement is None:
            raise pytest.UsageError(
                f"bootstrap dependency {name!r} is missing from requirements-diagnostics.txt"
            )
        try:
            module = importlib.import_module(name)
            distribution, minimum = requirement
            installed = _running_version(name, module)
            _require_version(distribution, minimum, installed=installed)
        except ImportError as exc:
            raise pytest.UsageError(str(exc)) from exc


def pytest_configure(config):
    global _STRICT
    _STRICT = config.getoption("--strict-diagnostics-deps")
    if _STRICT:
        try:
            dependencies = _load_diagnostic_dependencies(
                _PROJECT_ROOT / "requirements-diagnostics.txt", strict=True
            )
        except ValueError as exc:
            raise pytest.UsageError(str(exc)) from exc
        _validate_bootstrap_dependencies(dependencies)
```

Validation order is `packaging`, then `pytest`: an absent packaging distribution
is reported without trying to compare pytest first, while an installed
below-floor packaging can still supply `Version` to compare its own version.
Strict parsing turns unsupported requirement syntax into a clean usage error;
ordinary collection ignores such lines instead of losing the entire suite.

- [ ] **Step 4: Run dependency tests and verify GREEN**

Run:

```bash
python3 -m pytest tests/test_box_features.py -v -k "require_dep or dependency or packaging or bootstrap or conftest or marker_keyword"
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
- Modify: `allocator/box_features.py` imports, extraction helper call, and identity contracts
- Modify: `tests/test_box_features.py` synthetic-config fixture and identity tests

**Interfaces:**
- Consumes: frozen local bindings in `allocator.box_features`; allowance bindings in `allocator.strategies._scoring`.
- Produces:
  ```python
  def config_hash() -> str       # 16 hex characters over 13 named inputs
  def config_snapshot() -> dict  # exact 14-key JSON-round-trippable object
  ```
- Snapshot adds integer values `category_fruit` and `category_vegetables`.

- [ ] **Step 1: Write failing identity-owner and category tests**

First extend `_use_synthetic_scoring_config` so every new direct binding receives
synthetic data and no real ignored config value can reach test output:

```python
monkeypatch.setattr(
    box_features, "BOX_TARGET_PCT", _TEST_ONLY_CONFIG["box_target_pct"]
)
monkeypatch.setattr(box_features, "VALUE_SWEET_FROM", value_band["sweet_from"])
monkeypatch.setattr(box_features, "VALUE_SWEET_TO", value_band["sweet_to"])
monkeypatch.setattr(
    box_features, "VALUE_PENALTY_EXPONENT", value_band["penalty_exponent"]
)
```

Keep the existing `allocator.config` fixture patches because config-loader tests
still use them; the new local patches are additions, not replacements.

The owner change invalidates these existing tests as written, so replace each
explicitly rather than discovering them after GREEN:

- `test_config_hash_changes_when_any_pinned_input_changes` (10 parameters)
- `test_config_hash_changes_with_canonical_default_classification`
- `test_config_hash_includes_target_percentage_through_box_tiers`
- `test_config_snapshot_changes_with_classification_fallback`
- `test_config_snapshot_changes_with_default_classification`

Seven former config-module hash inputs now belong directly to
`allocator.box_features`; cover every one with:

```python
@pytest.mark.parametrize("name", [
    "BOX_TIERS",
    "GROUP_ALLOWANCES",
    "ITEM_CLASSIFICATIONS",
    "CLASSIFICATION_FALLBACK",
    "VALUE_SWEET_FROM",
    "VALUE_SWEET_TO",
    "VALUE_PENALTY_EXPONENT",
])
def test_local_config_owner_changes_hash(monkeypatch, name):
    import allocator.box_features as box_features

    before = box_features.config_hash()
    current = getattr(box_features, name)
    if isinstance(current, (int, float)):
        mutated = current + 1
    else:
        sample = next(iter(current))
        probe = max(current) + 1 if isinstance(sample, int) else "zzz_probe"
        mutated = {**current, probe: 1}
    monkeypatch.setattr(box_features, name, mutated)
    assert box_features.config_hash() != before


def test_default_classification_owner_changes_hash_and_snapshot(monkeypatch):
    import allocator.box_features as box_features

    before_hash = box_features.config_hash()
    before_snapshot = box_features.config_snapshot()
    monkeypatch.setattr(
        box_features,
        "DEFAULT_CLASSIFICATION",
        ("reserved_default", "cooking", "green", "round"),
    )
    assert box_features.config_hash() != before_hash
    assert (
        box_features.config_snapshot()["default_classification_hash"]
        != before_snapshot["default_classification_hash"]
    )


def test_box_tier_owner_carries_target_percentage_into_hash(monkeypatch):
    import allocator.box_features as box_features

    before = box_features.config_hash()
    tiers = {tier: dict(values) for tier, values in box_features.BOX_TIERS.items()}
    tiers["small"]["target_value"] += 1
    monkeypatch.setattr(box_features, "BOX_TIERS", tiers)
    assert box_features.config_hash() != before


def test_classification_fallback_owner_changes_snapshot(monkeypatch):
    import allocator.box_features as box_features

    before = box_features.config_snapshot()
    fallback = dict(box_features.CLASSIFICATION_FALLBACK)
    category_id = next(iter(fallback))
    fallback[category_id] = ("reserved_fallback", *fallback[category_id][1:])
    monkeypatch.setattr(box_features, "CLASSIFICATION_FALLBACK", fallback)
    assert (
        box_features.config_snapshot()["classification_fallback_hash"]
        != before["classification_fallback_hash"]
    )
```

The box-tier test deliberately overlaps the parameterized owner test because it
documents the indirect `BOX_TARGET_PCT -> BOX_TIERS.target_value -> hash`
contract. Update the exact snapshot-key test to include `category_fruit` and
`category_vegetables`, then add category-owner tests:

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


def test_fungible_group_owner_changes_allowance_hash_and_snapshot(monkeypatch):
    import allocator.box_features as box_features
    import allocator.strategies._scoring as scoring

    before_hash = box_features.config_hash()
    before_snapshot = box_features.config_snapshot()
    groups = dict(scoring.FUNGIBLE_GROUPS)
    degree, prefixes, _quantity_class = groups["apple"]
    groups["apple"] = (degree, prefixes, "cooking_piece")
    monkeypatch.setattr(scoring, "FUNGIBLE_GROUPS", groups)

    record = box_features.extract_box_features(
        "x", {1: 1}, _item_lookup(), "small", _available_tags(), 1
    )
    assert record["item_quantities"][0][2] == scoring.QUANTITY_CLASSES[
        "cooking_piece"
    ]["small"]
    assert box_features.config_hash() != before_hash
    assert box_features.config_snapshot() != before_snapshot


def test_price_threshold_owner_changes_allowance_hash_and_snapshot(monkeypatch):
    import allocator.box_features as box_features
    import allocator.strategies._scoring as scoring

    lookup = _item_lookup()
    original_threshold = scoring.QTY_CLASS_PRICE_THRESHOLDS["snacking_max"]
    lookup[4] = {
        **lookup[1],
        "price": original_threshold - 1,
        "fungible_group": None,
        "fungible_degree": 0.0,
    }
    before_hash = box_features.config_hash()
    before_snapshot = box_features.config_snapshot()
    before_record = box_features.extract_box_features(
        "x", {4: 1}, lookup, "small", _available_tags(), 1
    )

    thresholds = dict(scoring.QTY_CLASS_PRICE_THRESHOLDS)
    thresholds["snacking_max"] -= 1
    monkeypatch.setattr(scoring, "QTY_CLASS_PRICE_THRESHOLDS", thresholds)
    after_record = box_features.extract_box_features(
        "x", {4: 1}, lookup, "small", _available_tags(), 1
    )

    assert before_record["item_quantities"][0][2] == scoring.QUANTITY_CLASSES[
        "snack_piece"
    ]["small"]
    assert after_record["item_quantities"][0][2] == scoring.QUANTITY_CLASSES[
        "cooking_piece"
    ]["small"]
    assert box_features.config_hash() != before_hash
    assert box_features.config_snapshot() != before_snapshot
```

The price-threshold test explicitly creates item 4 as an ungrouped snacking
fixture one cent below the original boundary, then lowers the boundary by one so
the comparison changes from `snack_piece` to `cooking_piece`. Do not assert only
on guard text: each mutation changes `item_quantities` as well.

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
    """Stamp over thirteen effective schema/scoring inputs.

    BOX_TARGET_PCT is represented by BOX_TIERS' derived target_value entries;
    config_snapshot() carries the frozen percentage directly.
    """
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

Adding the two category IDs intentionally changes the hash value. No migration
is required: repository search confirms nothing outside
`allocator.box_features` and its tests reads `config_hash()` yet, and no stored
feature file currently carries that stamp.

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
- Modify: `allocator/box_features.py` exception type and category guard
- Modify: `scripts/extract_features.py` imports, extraction wrapper, and all batch call sites
- Modify: `tests/test_box_features.py` unsupported-category and batch-boundary tests

**Interfaces:**
- Produces:
  ```python
  class UnsupportedCategoryError(ValueError): ...
  def _extract_or_skip(
      box_name: str,
      allocations: dict[int, int],
      item_lookup: dict[int, dict],
      tier: str,
      available_tags: dict[str, set[str]],
      offer_id: int,
      source: str = "manual",
      preference: str | None = None,
  ) -> dict | None
  ```
- `_extract_or_skip` prints `  [SKIP] {error}` only for
  `UnsupportedCategoryError`. It also preserves the extractor's existing `None`
  for an empty/unresolved box. Both outcomes are intentionally absent from the
  emitted feature list, so `n_manual` continues to count emitted manual records,
  not attempted boxes.
- Importing `scripts.extract_features` for its pure batch helpers must not import
  `compare` or `allocator.db`; DB-dependent imports move inside `main()`.

- [ ] **Step 1: Write failing exception and batch-continuation tests**

First add a fresh-process import boundary test so the three following tests do
not depend on gitignored `queries.json`:

```python
def test_extract_features_batch_helpers_import_without_database():
    code = (
        "import sys\n"
        "import scripts.extract_features\n"
        "assert 'compare' not in sys.modules\n"
        "assert 'allocator.db' not in sys.modules\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PROJECT_ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
```

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
python3 -m pytest tests/test_box_features.py -v -k "third_category or unsupported or batch_boundary or synthetic_generation or batch_helpers"
```

Expected failures: importing the script loads `allocator.db` (or fails when
`queries.json` is absent); `UnsupportedCategoryError` and `_extract_or_skip` do
not exist; synthetic generation propagates the injected error.

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


def _extract_or_skip(
    box_name: str,
    allocations: dict[int, int],
    item_lookup: dict[int, dict],
    tier: str,
    available_tags: dict[str, set[str]],
    offer_id: int,
    source: str = "manual",
    preference: str | None = None,
) -> dict | None:
    try:
        return extract_box_features(
            box_name,
            allocations,
            item_lookup,
            tier,
            available_tags,
            offer_id,
            source=source,
            preference=preference,
        )
    except UnsupportedCategoryError as exc:
        print(f"  [SKIP] {exc}")
        return None
```

Remove the module-scope `compare` and `allocator.db` imports. Import only the
names actually used by `main()` at the top of that function:

```python
def main():
    import argparse

    from compare import (
        build_item_lookup,
        compute_available_tags,
        load_historical_csv,
        load_summary,
        read_xlsx_pack_overrides,
    )
    from allocator.db import fetch_mystery_box_buyers
```

Do not retain the already-unused `_discover_cleaned_offer_ids` or `_offer_tier`
imports.

Replace all five synthetic calls and the manual call with `_extract_or_skip`.
Keep only `BOX_TIERS`, `DONATION_IDENTIFIERS`, `SKIP_COLUMN_IDENTIFIERS`, and
`STAFF_IDENTIFIERS` in the config import block. Remove the five import-only
names `CATEGORY_FRUIT`, `CATEGORY_VEGETABLES`, `FUNGIBLE_GROUPS`,
`GROUP_ALLOWANCES`, and `QUANTITY_CLASSES`.

- [ ] **Step 4: Run category-boundary tests and verify GREEN**

Run:

```bash
python3 -m pytest tests/test_box_features.py -v -k "third_category or unsupported or batch_boundary or synthetic_generation or batch_helpers"
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

### Task 4: PR Documentation

**Files:**
- Modify: `CLAUDE.md` diagnostics and key-modules sections

**Interfaces:**
- The PR documentation describes only commands that exist now and the exact 14-key snapshot.

- [ ] **Step 1: Correct current repository documentation**

In `CLAUDE.md`, keep the install command but remove the nonexistent script invocation:

```bash
pip install --target .venv-diagnostics/lib -r requirements-diagnostics.txt
```

Keep the strict diagnostic pytest command unchanged. Replace the single pytest
floor paragraph with:

```markdown
`pytest>=9.0.0` and `packaging>=22` are bootstrap dependencies for diagnostic
tests, not production runtime dependencies. Under
`--strict-diagnostics-deps`, pytest validates `packaging` and the running
pytest's `__version__` against those floors before collection. Other diagnostic
libraries remain demand-checked by `require_dep()` when their test modules load.
```

Update the plain-versus-strict explanation to state that malformed bootstrap
requirements or missing/below-floor bootstrap dependencies fail startup with a
concise pytest usage error. Change the key-module description from “exact
12-key `config_snapshot()`” to “exact 14-key `config_snapshot()`”.

- [ ] **Step 2: Verify and commit PR documentation**

Run:

```bash
rg -n "ebm_diagnostic.py|12-key|pytest>=9.0.0.*only" CLAUDE.md
git diff --check
```

Expected: the search returns no matches and the diff check emits nothing. Then
commit only the PR documentation:

```bash
git add CLAUDE.md
git commit -m "docs: correct diagnostics foundation commands"
```

---

### Task 5: Authoritative Planning-Workspace Documentation

**Files:**
- Modify: `/home/mark/jointly/Mystery-Manager/docs/superpowers/specs/2026-08-03-ebm-ordinal-diagnostics-design.md`
- Modify: `/home/mark/jointly/Mystery-Manager/docs/superpowers/plans/2026-08-04-diagnostics-foundation.md`

**Interfaces:**
- These ignored planning-workspace documents enumerate the same 9-package
  manifest, 13-input hash, and 14-key snapshot as implementation.
- This task makes no PR-worktree edits and creates no PR commit.

- [ ] **Step 1: Synchronize the parent diagnostics design**

In the Part 4 dependency block, insert:

```text
packaging>=22
```

immediately after `pytest>=9.0.0`, and document both as bootstrap dependencies validated at strict pytest startup.
State that pytest's floor is checked against the running module's
`pytest.__version__`, not distribution metadata, and that missing/below-floor or
strictly malformed bootstrap requirements raise `pytest.UsageError` before
collection.

In the schema-guard section, add `CATEGORY_FRUIT` and `CATEGORY_VEGETABLES` to the enumerated hash inputs and change every “eleven named inputs”/“11-input” statement to 13.

In the snapshot JSON object, add:

```json
"category_fruit": 0,
"category_vegetables": 0,
```

after `box_target_pct`, and change every exact twelve-key statement to fourteen keys. Replace the claim that `packaging.version.Version` is merely “available with pytest” with the explicit `packaging>=22` manifest/bootstrap contract.

- [ ] **Step 2: Synchronize the foundation plan**

Apply the same manifest, hash-input, snapshot-key, code-example, test-expectation, and prose-count changes throughout `2026-08-04-diagnostics-foundation.md`. Its Task 1 dependency file example must list `packaging>=22`; its Task 5 interfaces must say 13 inputs and 14 keys; its `_HASH_INPUTS` and snapshot examples must include both category IDs and effective-owner semantics.
Its dependency-gating example must preserve the sentinel collection test, use
the running pytest version, lazily import `Version`, enforce bootstrap manifest
membership, and specify `pytest.UsageError` for strict startup failures.

- [ ] **Step 3: Verify planning-document consistency**

Run:

```bash
rg -n "pytest>=9.0.0|packaging>=22|CATEGORY_FRUIT|CATEGORY_VEGETABLES|category_fruit|category_vegetables|13-input|14-key|fourteen keys" /home/mark/jointly/Mystery-Manager/docs/superpowers/specs/2026-08-03-ebm-ordinal-diagnostics-design.md /home/mark/jointly/Mystery-Manager/docs/superpowers/plans/2026-08-04-diagnostics-foundation.md
```

Expected: the planning documents show `packaging>=22`, both category IDs,
13-input hash wording, and 14-key snapshot wording. Review numeric matches such
as dataset counts separately; do not mechanically replace unrelated numbers 11,
12, 13, or 14.

- [ ] **Step 4: Record planning-workspace state separately**

The two authoritative companion files are intentionally outside PR #1 and
ignored by this repository. Preserve their edits in the planning workspace; do
not force-add them to the PR branch. Report their exact paths in the final
handoff so they are not mistaken for PR changes.

---

### Task 6: Final Verification and Review-Finding Audit

**Files:**
- Verify only; no new implementation files.

**Interfaces:**
- Produces fresh evidence that each review finding is resolved or intentionally handled.

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
BOX_PRICE_SMALL=2000 BOX_PRICE_MEDIUM=3500 BOX_PRICE_LARGE=5000 python3 -c "import allocator.box_features; print('DB-free import passed')"
BOX_PRICE_SMALL=2000 BOX_PRICE_MEDIUM=3500 BOX_PRICE_LARGE=5000 python3 -c "import sys, scripts.extract_features; assert 'compare' not in sys.modules and 'allocator.db' not in sys.modules; print('batch-helper import passed')"
git diff main...HEAD --check
git status --short
```

Expected: the two imports use the test suite's synthetic box-price preconditions
(`BOX_PRICE_SMALL=2000`, `BOX_PRICE_MEDIUM=3500`, and
`BOX_PRICE_LARGE=5000`) and both pass without loading `compare` or
`allocator.db`; compilation exits zero; diff check emits nothing; PR-worktree
status is empty. The separately managed planning documents are
gitignored files in another checkout and therefore cannot appear in this status;
their contents are verified by Task 5's explicit `rg` command and reported by
absolute path in the final handoff.

- [ ] **Step 4: Audit each review item against the resulting diff**

Confirm all seven outcomes explicitly:

1. no top-level `packaging` import and `packaging>=22` is declared;
2. `CLAUDE.md` names no nonexistent diagnostic script;
3. both category IDs participate in hash and snapshot;
4. guards read the same frozen owners as feature production;
5. unsupported categories skip one record while unrelated errors propagate;
6. dependency floors occur only in the manifest, with code deriving them;
7. all five import-only config names are removed from the script, including the
   three named by the original review.
8. importing the script's batch helpers does not load DB configuration.

- [ ] **Step 5: Request code review before integration**

Use `superpowers:requesting-code-review` on `main...HEAD`. Address any confirmed findings with `superpowers:receiving-code-review`, then repeat Steps 1–4 before reporting completion.
