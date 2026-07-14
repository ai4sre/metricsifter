import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = (ROOT / ".github/workflows/ci.yaml").read_text(encoding="utf-8")
PUBLISH_WORKFLOW = (ROOT / ".github/workflows/publish.yaml").read_text(encoding="utf-8")


def _job(workflow: str, name: str) -> str:
    match = re.search(rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [\w-]+:\n|\Z)", workflow)
    assert match is not None, f"workflow job {name!r} is missing"
    return match.group(1)


def _assert_in_order(text: str, *needles: str) -> None:
    positions = [text.find(needle) for needle in needles]
    assert all(position >= 0 for position in positions), f"missing workflow step from {needles!r}"
    assert positions == sorted(positions), f"workflow steps are out of order: {needles!r}"


def test_ci_runs_for_pushes_and_pull_requests() -> None:
    trigger = CI_WORKFLOW.split("\njobs:", maxsplit=1)[0]

    assert re.search(r"(?m)^  push:\s*$", trigger)
    assert re.search(r"(?m)^  pull_request:\s*$", trigger)


def test_publish_runs_only_for_version_tags() -> None:
    trigger = PUBLISH_WORKFLOW.split("\njobs:", maxsplit=1)[0]

    assert re.search(r'(?m)^      - ["\']v\*["\']\s*$', trigger)
    assert "pull_request:" not in trigger


def test_ci_quality_job_uses_locked_dev_environment() -> None:
    quality = _job(CI_WORKFLOW, "quality")

    assert "uv sync --locked --extra dev" in quality
    assert "uv lock --check" in quality
    assert "ruff check ." in quality
    assert "black --check ." in quality


def test_ci_runs_the_full_test_suite_on_all_supported_python_versions() -> None:
    test = _job(CI_WORKFLOW, "test")

    for version in ("3.10", "3.11", "3.12", "3.13", "3.14"):
        assert version in test
    assert "uv sync --locked --extra dev" in test
    assert "pytest -s -vv tests" in test
    assert "if: matrix.python-version" not in test


def test_ci_packages_only_after_quality_and_tests_and_smoke_tests_the_wheel() -> None:
    package = _job(CI_WORKFLOW, "package")

    assert re.search(r"needs:\s*\[?[^\n]*quality[^\n]*test", package)
    assert "*.whl" in package
    assert "mktemp -d" in package
    _assert_in_order(
        package,
        "uv build",
        "uv venv",
        "uv pip install",
        'cd "$SMOKE_DIR"',
        "import metricsifter",
        'bin/metricsifter" --help',
    )


def test_publish_validates_the_tag_and_package_before_uploading_artifacts() -> None:
    build = _job(PUBLISH_WORKFLOW, "validate-and-build")

    assert "uv sync --locked --extra dev" in build
    assert "uv lock --check" in build
    _assert_in_order(
        build,
        "pytest -s -vv tests",
        "ruff check .",
        "black --check .",
        "GITHUB_REF_NAME",
        "pyproject.toml",
        "uv build",
        "uv venv",
        "uv pip install",
        'cd "$SMOKE_DIR"',
        "import metricsifter",
        'bin/metricsifter" --help',
        "actions/upload-artifact",
    )


def test_publish_job_uses_only_the_validated_artifact() -> None:
    publish = _job(PUBLISH_WORKFLOW, "publish-to-pypi")

    assert "needs: validate-and-build" in publish
    _assert_in_order(publish, "actions/download-artifact", "pypa/gh-action-pypi-publish")
    assert "metricsifter_pypi" in publish
    assert "id-token: write" in publish
