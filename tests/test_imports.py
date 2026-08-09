"""
Import resolution and package collision test.

Asserts that external CybORG resolves inside installed dependencies (site-packages)
and that local project code resolves under cyber_jepa in src/.
"""

import inspect
from pathlib import Path

import CybORG
import cyber_jepa


def test_cyborg_import_resolution():
    """Assert CybORG resolves to installed dependency, never project repo."""
    cyborg_file = Path(inspect.getfile(CybORG)).resolve()
    project_root = Path(__file__).resolve().parent.parent

    # Local cyborg directory must not exist
    local_cyborg_dir = project_root / "cyborg"
    assert not local_cyborg_dir.exists(), (
        f"Local package directory {local_cyborg_dir} still exists. "
        "It must be removed to prevent case-insensitive package collision with CybORG."
    )

    # CybORG must resolve inside site-packages
    assert "site-packages" in [p.name.lower() for p in cyborg_file.parents], (
        f"CybORG resolved outside site-packages: {cyborg_file}"
    )


def test_cyber_jepa_import_resolution():
    """Assert cyber_jepa resolves to src/cyber_jepa."""
    jepa_file = Path(inspect.getfile(cyber_jepa)).resolve()
    project_root = Path(__file__).resolve().parent.parent
    expected_path = (project_root / "src" / "cyber_jepa" / "__init__.py").resolve()

    assert jepa_file == expected_path, (
        f"cyber_jepa resolved to {jepa_file}, expected {expected_path}"
    )
