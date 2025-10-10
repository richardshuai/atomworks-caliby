# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in the AtomWorks repository.

## Table of Contents
1. [Critical Setup Requirements](#critical-setup-requirements)
2. [Mandatory Git Workflow](#mandatory-git-workflow)
3. [Feature Development Workflow](#feature-development-workflow)
4. [Coding Practices](#coding-practices)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation Guidelines](#documentation-guidelines)
7. [Pre-PR Checklist](#pre-pr-checklist)

---

## Critical Setup Requirements

### Environment Setup

**CRITICAL**: Before running **ANY** commands, activate the Python environment:

```bash
# ALWAYS ACTIVATE THE PYTHON ENVIRONMENT FIRST!
source .venv/bin/activate
```

If `.venv` does not exist:
```bash
# Create virtual environment
uv venv --python 3.12

# Install dependencies
make install
```

### Test Suite Setup

Download test data (required for running tests):
```bash
atomworks setup tests
```

Run all tests:
```bash
PDB_MIRROR_PATH=tests/data/pdb pytest -n 15 tests -m "not very_slow"
```

When developing actively, focus on the most relevant tests rather than the entire test suite.

---

## Mandatory Git Workflow

### Core Rules

1. **ALWAYS create a feature branch BEFORE making changes**
   ```bash
   git checkout -b feat/[feature-name]  # or fix/[bug-name], refactor/[name], docs/[name]
   ```

2. **Commit changes REGULARLY during development**
   - After completing each major step
   - When switching between different files/features
   - Before running tests or builds
   - Use meaningful commit messages with type prefix
   - Keep messages brief and informative; avoid bloat

3. **NEVER work directly on main/dev branch**
   - All changes must go through feature branches
   - Create pull requests for review

### Commit Message Format

```bash
git commit -m "type: Brief description

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Common types**: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `style`

**Examples**:
- `feat: Add MSA generation pipeline`
- `fix: Handle missing residue annotations in parser`
- `refactor: Extract common atom selection logic`
- `docs: Update docstrings for transform base class`

### Pull Request Target Branches

**CRITICAL**: PRs should target the branch they were created from unless explicitly specified otherwise.

**Branch Strategy:**
- Feature branches typically merge back to `dev`
- If explicitly told to target `staging`, follow special staging rules below

### AtomWorks-Specific: Merging to Staging

**IMPORTANT**: When creating PRs to `staging` branch, certain files MUST be excluded.

**NEVER include in staging PRs:**
- `CLAUDE.md` (all CLAUDE.md files anywhere in repo)
- `.ipd/` directory and all contents
- `.env` file
- `src/atomworks/ml/databases/` directory and all contents

**How to exclude files when targeting staging:**

1. **Check what you're about to commit:**
   ```bash
   git status
   git diff --staged
   ```

2. **If forbidden files are present, unstage them:**
   ```bash
   # Unstage specific files
   git restore --staged CLAUDE.md
   git restore --staged .ipd/
   git restore --staged .env
   git restore --staged src/atomworks/ml/databases/
   ```

3. **Verify before pushing:**
   ```bash
   git diff --staged --name-only | grep -E "(CLAUDE\.md|\.ipd/|\.env|src/atomworks/ml/databases/)" && echo "⚠️  WARNING: Forbidden files staged for staging!" || echo "✓ Clean for staging"
   ```

4. **Alternative: Create staging-specific branch from clean state:**
   ```bash
   # Create new branch excluding forbidden files
   git checkout -b feat/feature-name-staging dev
   git cherry-pick <commits>  # Pick only relevant commits
   # Manually ensure forbidden files aren't included
   ```

**Before creating PR to staging, verify:**
```bash
# List all changed files in the PR
git diff origin/staging...HEAD --name-only

# Should NOT contain:
# - CLAUDE.md
# - .ipd/*
# - .env
# - src/atomworks/ml/databases/*
```

### Critical Rule

**If you complete a task without proper Git commits = TASK INCOMPLETE**

---

## Feature Development Workflow

Follow this structured workflow for ALL feature development:

### Phase 1: Understanding & Planning

1. **Read and understand the request**
   - Ask clarifying questions if requirements are ambiguous
   - Identify edge cases and potential issues upfront

2. **Survey the codebase**
   - Search for related existing code
   - Identify patterns to follow
   - Find similar implementations to use as reference
   - Look through this CLAUDE.md file, and other CLAUDE.md files in the codebase, to discover relevant guidelines

3. **Create a task plan**
   - Use TodoWrite to create a detailed todo list
   - Break down complex tasks into 3-7 actionable steps
   - Include testing and documentation in the plan, adhering to my testing and documentation guidelines detailed below

4. **Explain your approach**
   - Describe your implementation strategy briefly
   - Highlight any design decisions or tradeoffs
   - Get user confirmation if making architectural changes

5. **Collate into a design document**
    - Combine your task plan, codebase survey, and approach strategy into a thorough design document
    - **IMPORTANT**: Save the design document as a markdown file for future reference, with an informative name
    - Compare your proposed design document against the best-practices for testing, documentation, and code detailed in this file to ensure that they comply

### Phase 2: Implementation

5. **Create feature branch**
   ```bash
   git checkout -b feat/descriptive-name
   ```

6. **Implement incrementally**
   - Work through todo list one item at a time
   - Mark items as in_progress → completed as you go
   - Commit after each major step (not just at the end)

7. **Follow coding best practices** (see Coding Practices section)
   - DRY: Extract common logic
   - YAGNI: Don't add unrequested features
   - Prefer functions over classes

8. **Write tests as you go** (not at the end)
   - Add tests alongside implementation
   - Ensure tests pass before moving to next step

### Phase 3: Quality Assurance

9. **Run the full test suite**
   ```bash
   make format  # Format code first
   PDB_MIRROR_PATH=tests/data/pdb pytest tests -m "not very_slow" -n auto
   ```

10. **Self-review using Pre-PR Checklist** (see section below)
    - Review all changes systematically
    - Fix any issues found

11. **Final commit and PR**
    - Commit any final fixes
    - Create pull request with meaningful description

### Example Workflow

```
User: "Add a transform to filter atoms by B-factor"

Claude:
1. [Plan] I'll create a new transform class that filters atoms based on B-factor threshold.

   Todo list:
   - Survey existing filter transforms for patterns
   - Implement BFactorFilter transform class
   - Add comprehensive docstring
   - Write parameterized tests
   - Update documentation

2. [Search] Looking for similar filter implementations...
   Found: SelectionTransform, RemoveWaters - will follow this pattern

3. [Implement] Creating feature branch...
   git checkout -b feat/bfactor-filter

4. [Code] Writing BFactorFilter class...
   [Commits after completing class]

5. [Test] Adding tests...
   [Commits after tests pass]

6. [Review] Running pre-PR checklist...
   - Docstrings: ✓ Google style with examples
   - Tests: ✓ Parameterized end-to-end test
   - Lint: ✓ No issues
   - Full test suite: ✓ All passing

7. [Done] Creating PR...
```

---

## Coding Practices

### Core Principles

**IMPORTANT**: These are fundamental to our codebase. Retroactively review all code against these principles.

#### 1. DRY (Don't Repeat Yourself)

Extract common operations into shared functions or utilities.

**Bad Example**:
```python
# In file A
def process_chain_a(chain):
    atoms = chain[chain.element != "H"]
    atoms = atoms[atoms.b_factor < 50]
    return atoms

# In file B
def process_chain_b(chain):
    atoms = chain[chain.element != "H"]
    atoms = atoms[chain.b_factor < 50]
    return atoms
```

**Good Example**:
```python
# In utils/atom_array.py
def filter_atoms(
    atoms: AtomArray,
    remove_hydrogen: bool = True,
    max_bfactor: float | None = None
) -> AtomArray:
    """Filter atoms by element and B-factor."""
    if remove_hydrogen:
        atoms = atoms[atoms.element != "H"]
    if max_bfactor is not None:
        atoms = atoms[atoms.b_factor < max_bfactor]
    return atoms

# In both files
result = filter_atoms(chain, remove_hydrogen=True, max_bfactor=50)
```

#### 2. Prefer Functions Over Classes

Use classes only when you need state or complex inheritance. Otherwise, use functions.

**Bad Example** (unnecessary class):
```python
class CoordinateCalculator:
    def calculate_centroid(self, atoms: AtomArray) -> np.ndarray:
        return atoms.coord.mean(axis=0)

    def calculate_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(a - b)
```

**Good Example** (simple functions):
```python
def calculate_centroid(atoms: AtomArray) -> np.ndarray:
    """Calculate the geometric center of atoms."""
    return atoms.coord.mean(axis=0)

def calculate_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(a - b)
```

**When to use classes**: Transforms, parsers with state, complex builders.

#### 3. YAGNI (You Ain't Gonna Need It)

Don't speculatively add features. Prioritize simplicity and brevity.

**Bad Example**:
```python
# User asked to filter by B-factor
def filter_atoms(
    atoms: AtomArray,
    bfactor_threshold: float | None = None,
    occupancy_threshold: float | None = None,  # Not requested
    charge_filter: str | None = None,  # Not requested
    custom_filter_fn: Callable | None = None,  # Not requested
) -> AtomArray:
    # ... complex implementation with many branches
```

**Good Example**:
```python
# User asked to filter by B-factor
def filter_by_bfactor(atoms: AtomArray, threshold: float) -> AtomArray:
    """Filter atoms with B-factor below threshold."""
    return atoms[atoms.b_factor < threshold]

# Later, if needed, add filter_by_occupancy as separate function
```

### Additional Best Practices

#### Type Hints

Always use type hints for function signatures. For PyTorch tensors, use `jaxtyping` to specify shapes.

**Standard type hints:**
```python
def parse_structure(path: str, remove_waters: bool = True) -> dict[str, AtomArray]:
    """Parse structure from file."""
    ...
```

**Use jaxtyping for PyTorch tensors:**
```python
from jaxtyping import Float, Int
import torch

def compute_distances(
    coords: Float[torch.Tensor, "n_atoms 3"]
) -> Float[torch.Tensor, "n_atoms n_atoms"]:
    """Compute pairwise distances between atoms."""
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    return torch.norm(diff, dim=-1)

def encode_sequence(
    seq_ids: Int[torch.Tensor, "batch seq_len"],
    embeddings: Float[torch.Tensor, "vocab_size embed_dim"]
) -> Float[torch.Tensor, "batch seq_len embed_dim"]:
    """Embed sequence IDs."""
    return embeddings[seq_ids]
```

**When to use jaxtyping:**
- ✅ Public APIs that accept/return tensors
- ✅ Transform classes working with tensor features
- ✅ Model forward passes and complex tensor operations
- ❌ Simple internal utilities where shape is obvious
- ❌ When shapes are truly dynamic and can't be specified

**jaxtyping conventions:**
- Use descriptive dimension names: `"n_atoms 3"`, `"batch seq_len embed_dim"`
- Use `...` for variable dimensions: `"batch ... 3"` for batch of coordinates
- Multiple possible shapes: `Float[torch.Tensor, "n 3"] | Float[torch.Tensor, "n m 3"]`

#### Error Handling

**Philosophy: Trust Python's error handling. Don't litter code with defensive checks.**

Most of the time, let Python's natural errors occur. Only add explicit error handling when:
1. **User-facing functions** need clear error messages (CLI commands, top-level APIs)
2. **Data integrity** issues that would cause silent corruption
3. **Resource cleanup** is required (file handles, connections)

**Default approach - Let it fail naturally:**
```python
def get_chain(assembly: AtomArray, chain_id: str) -> AtomArray:
    """Get chain by ID."""
    # Let numpy/biotite raise natural errors if chain_id doesn't exist
    return assembly[assembly.chain_id == chain_id]
```

**When to add checks - User-facing APIs with unclear errors:**
```python
def parse_structure(path: str) -> dict[str, AtomArray]:
    """Parse structure from file.

    Raises:
      FileNotFoundError: If path does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    return _parse_internal(path)  # Let internal errors propagate
```

**Use assertions for invariants (catches programmer errors):**
```python
def crop_atoms(atoms: AtomArray, center_idx: int, radius: float) -> AtomArray:
    """Crop atoms within radius of center."""
    assert radius > 0, "Radius must be positive"
    assert 0 <= center_idx < len(atoms), "Invalid center index"

    distances = np.linalg.norm(atoms.coord - atoms.coord[center_idx], axis=1)
    return atoms[distances <= radius]
```

**Guidelines:**
- Don't validate input types - type hints + tests handle this
- Don't check for None unless it's truly ambiguous
- Don't wrap operations in try/except unless you're adding value
- Document exceptions only when you explicitly raise them
- Use early returns to avoid nesting, but don't add defensive checks "just in case"

#### Naming Conventions
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`
- Use descriptive names; avoid abbreviations unless standard (e.g., `pdb_id`, `msa`)

---

## Testing Guidelines

### Philosophy

**Quality over quantity. Avoid test bloat at all costs.**

### Core Principles

1. **Focus on end-to-end tests** over implementation details
2. **1-2 good tests >> 3-4 bad tests** - use parameterization
3. **Don't test trivial functionality** - skip getters/setters
4. **Test behavior, not implementation** - avoid testing internal state

### Test Patterns

#### Parameterized End-to-End Tests (Preferred)

```python
@pytest.mark.parametrize("pdb_id", ["1a8o", "6lyz", "7bv2"])
def test_parse_workflow(pdb_id: str):
    """Test parsing workflow on diverse structures."""
    result = parse(get_pdb_path(pdb_id), remove_waters=True)
    assert result is not None
    assert len(result["asym_unit"]) > 0
    assert "chain_id" in result["asym_unit"].get_annotation_categories()
```

#### Property-Based Tests for Transforms

```python
@pytest.mark.parametrize(
    "threshold,expected_max",
    [(30.0, 30.0), (50.0, 50.0), (100.0, 100.0)]
)
def test_bfactor_filter(threshold: float, expected_max: float):
    """Test B-factor filtering maintains invariants."""
    atoms = cached_parse("1a8o")
    filtered = filter_by_bfactor(atoms, threshold)

    assert filtered.b_factor.max() <= expected_max
    assert len(filtered) <= len(atoms)
```

### What to Test

**DO write tests for:**
- Complex parsing logic with edge cases
- Transforms that modify structures/annotations
- Regression cases (previously buggy behavior)
- Integration of multiple components

**DON'T write tests for:**
- Simple data classes or POJOs
- Obvious delegation/wrapper functions
- Code already covered by integration tests
- Getters/setters without logic

### Test Organization

- Group related tests in same file
- Use fixtures for shared test data
- Name descriptively: `test_<what>_<scenario>`
- Mark slow tests: `@pytest.mark.slow`
- Use `cached_parse()` for reusing structures

---

## Documentation Guidelines

### Quick Reference

**Primary Goals:**
- **Concise**: Short while being clear
- **No redundancy**: Don't repeat types or obvious behavior
- **Sphinx-first**: Use reStructuredText roles/directives
- **Google sections**: Use `Args:`, `Returns:`, etc.

### Essential Rules

1. **One-line summary**: Imperative, ends with period
   ```python
   """Filter atoms by B-factor threshold."""
   ```

2. **Args section**: Omit types when annotated
   ```python
   def fn(a: int, path: str | None = None):
       """Do something.

       Args:
         a: Number of items to process.
         path: Optional file path. Defaults to ``None``.
       """
   ```

3. **Returns section**: Omit if None or obvious from summary
   ```python
   def compute() -> dict[str, int]:
       """Compute counts.

       Returns:
         Mapping of names to counts.
       """
   ```

4. **Examples section**: Strongly encouraged
   ```python
   def square(x: int) -> int:
       """Return the square of ``x``.

       Examples:
         >>> square(3)
         9
       """
   ```

5. **Raises section**: Only unusual/important exceptions
   ```python
   Raises:
     ValueError: If threshold is negative.
   ```

### Section Ordering

1. Summary (required)
2. Optional elaboration
3. Args
4. Returns/Yields
5. Raises
6. Examples
7. References
8. See Also

### Code References

Use Sphinx roles liberally:
- Functions: `:py:func:`~package.module.fn``
- Classes: `:py:class:`~package.module.Class``
- Methods: `:py:meth:`~package.module.Class.method``

### Quick Templates

**Simple function**:
```python
def fn(x: int) -> int:
    """Return the square of ``x``.

    Examples:
      >>> fn(3)
      9
    """
```

**Complex function**:
```python
def load(path: str, *, strict: bool = False) -> dict:
    """Load a configuration from ``path``.

    Args:
      path: File path to load.
      strict: Validate schema strictly. Defaults to ``False``.

    Returns:
      Mapping representing the configuration.

    Raises:
      FileNotFoundError: If the file does not exist.

    Examples:
      >>> cfg = load("config.yaml", strict=True)
    """
```

**Class**:
```python
class Cache:
    """In-memory cache with time-based eviction."""

    def __init__(self, ttl: float, capacity: int = 1024):
        """Initialize the cache.

        Args:
          ttl: Time-to-live in seconds.
          capacity: Max entries. Defaults to ``1024``.
        """
```

---

## Pre-PR Checklist

**CRITICAL**: Complete this checklist before opening ANY pull request.

### 1. Code Quality

- [ ] **DRY principle**: No repeated code; common logic extracted
- [ ] **YAGNI principle**: No speculative features; only requested functionality
- [ ] **Functions over classes**: Used functions unless state/inheritance needed
- [ ] **Type hints**: All function signatures have complete type annotations
- [ ] **Error handling**: Informative error messages; specific exceptions
- [ ] **Naming**: Descriptive names following conventions (snake_case, PascalCase)

### 2. Documentation

- [ ] **Docstrings present**: All public functions/classes/methods documented
- [ ] **Google style**: Proper section headers (`Args:`, `Returns:`, etc.)
- [ ] **Concise**: No redundancy; types omitted when annotated
- [ ] **Examples included**: Non-trivial functions have usage examples
- [ ] **Cross-references**: Used Sphinx roles for code references
- [ ] **Raises documented**: Unusual exceptions documented in docstring

### 3. Testing

- [ ] **Tests written**: New functionality has tests
- [ ] **End-to-end focus**: Tests cover full workflows, not just helpers
- [ ] **Parameterized**: Used `@pytest.mark.parametrize` for multiple cases
- [ ] **No trivial tests**: Didn't test getters/setters/obvious wrappers
- [ ] **Tests pass locally**: All new tests pass
- [ ] **Full suite passes**: Entire test suite passes (not just new tests)
  ```bash
  make format
  PDB_MIRROR_PATH=tests/data/pdb pytest tests -m "not very_slow"
  ```

### 4. Code Formatting

- [ ] **Linted**: Code passes ruff checks
  ```bash
  make format
  ruff check src tests
  ```
- [ ] **Formatted**: Code is auto-formatted
  ```bash
  ruff format .
  ```

### 5. Git Workflow

- [ ] **Feature branch**: Work done on feature branch (not main/dev)
- [ ] **Regular commits**: Multiple meaningful commits (not one giant commit)
- [ ] **Commit messages**: Proper format with type prefix
- [ ] **No debug code**: Removed print statements, breakpoints, TODOs

### 6. Final Review

- [ ] **Read all diffs**: Reviewed every changed line
- [ ] **No unintended changes**: No accidental formatting changes in unrelated files
- [ ] **Imports clean**: No unused imports or wildcard imports
- [ ] **Comments removed**: Removed commented-out code blocks
- [ ] **Error messages tested**: Verified error paths work correctly

### 7. Pull Request

- [ ] **Correct target branch**: PR targets the branch it was created from (typically `dev`)
- [ ] **Staging exclusions verified**: If targeting `staging`, confirmed no forbidden files included:
  - No `CLAUDE.md` files
  - No `.ipd/` directory
  - No `.env` file
  - No `src/atomworks/ml/databases/` directory
- [ ] **Descriptive title**: Clear summary of changes
- [ ] **Summary section**: 1-3 bullet points of what changed
- [ ] **Test plan**: List of how changes were tested

### 8. Cleanup Before PR

**IMPORTANT**: Clean up temporary files before opening the pull request.

**Keep in working directory (do NOT commit):**
- Original design document/task plan (markdown file created during planning phase)
- `SUMMARY.md`: Brief summary of what was accomplished, any important decisions, gotchas
- Development/exploration test files that shouldn't be in the codebase
- Backup files (`.bak`, etc.)

**Delete completely:**
- All other temporary files created during development
- Intermediate test outputs
- Temporary scripts
- Debug files

**Example cleanup:**
```bash
# Keep these in working directory (already gitignored or don't add):
# - design_bfactor_filter.md (original plan)
# - SUMMARY.md (what was done)
# - test_exploration.py (dev tests)

# Delete temporary files:
rm temp_output.txt debug_coords.npy intermediate_results.pkl

# Verify only intended files are staged:
git status
git diff --staged
```

### Checklist Usage

Before running PR creation, explicitly state:

```
Running Pre-PR Checklist:
✓ Code Quality - DRY, YAGNI, functions over classes
✓ Documentation - Google style docstrings with examples
✓ Testing - End-to-end parameterized tests, full suite passes
✓ Formatting - Linted and formatted with ruff
✓ Git - Feature branch, regular commits
✓ Review - All diffs reviewed, no debug code
✓ PR Target - Targeting correct branch (dev/staging), staging exclusions verified
✓ Cleanup - Temporary files deleted, design doc and summary kept

Ready to create PR.
```

---

## Project Architecture

### Package Structure
- `src/atomworks/`: Main package
  - `io/`: Data I/O operations (parsing, conversion, manipulation)
    - `parser.py`: Main entry point with `parse()` function
    - `utils/`: Utilities for atom arrays, assemblies, chains
    - `tools/`: Tools for FASTA, RDKit integration, inference
  - `ml/`: Machine learning components
    - `datasets/`: Dataset wrappers and parsers
    - `transforms/`: Feature engineering and data transforms
    - `pipelines/`: Pre-built ML pipelines (AF3, RF2AA)
    - `preprocessing/`: Data preprocessing utilities
- `src/atomworks_cli/`: Command-line interface using Typer

### Key Concepts
- **AtomArray**: Core data structure from biotite, extended via monkey patching
- **PN Units**: "Polymer XOR Non-polymer units" - unified representation
- **Transforms**: Composable data processing maintaining AtomArray representation
- **Pipelines**: Pre-built transform chains for specific models

### Development Environment
- Python 3.12 preferred (3.11+ required)
- `ruff` for linting and formatting
- `pytest` with coverage, parallel execution, benchmarking
- Environment variables in `.env` file; load via `load_dotenv`

---

## Quick Command Reference

```bash
# Environment
source .venv/bin/activate          # ALWAYS run first

# Installation
make install                       # Install all dependencies

# Development
make format                        # Format and lint code
ruff format .                      # Format only
ruff check --fix src tests        # Lint and fix

# Testing
make test                          # Run tests with coverage
make parallel_test                 # Run tests in parallel
pytest tests/io/                   # Run specific test directory
make benchmark                     # Run speed benchmarks

# Git workflow
git checkout -b feat/feature-name  # Create feature branch
git add src/atomworks/io/new_file.py
git commit -m "feat: Add new parser

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```
