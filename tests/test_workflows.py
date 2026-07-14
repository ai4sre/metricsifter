import re
from pathlib import Path

import pytest

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


def _assert_unconditional_test_matrix(test_job: str) -> None:
    conditional_lines = [line for line in test_job.splitlines() if line.strip().startswith("if:")]
    assert not conditional_lines, f"test matrix contains conditional steps: {conditional_lines!r}"


def _assert_wheel_smoke(job: str) -> None:
    assert "*.whl" in job
    assert "mktemp -d" in job
    _assert_in_order(
        job,
        "uv build",
        "uv venv",
        "uv pip install",
        'cd "$SMOKE_DIR"',
        "import metricsifter",
        "metricsifter.__version__",
        'bin/metricsifter" --help',
    )


def _swap(text: str, first: str, second: str) -> str:
    sentinel = "__WORKFLOW_TEST_SENTINEL__"
    assert sentinel not in text
    return text.replace(first, sentinel, 1).replace(second, first, 1).replace(sentinel, second, 1)


def test_ci_runs_for_pushes_and_pull_requests() -> None:
    trigger = CI_WORKFLOW.split("\njobs:", maxsplit=1)[0]

    assert re.search(r"(?m)^  push:\s*$", trigger)
    assert re.search(r"(?m)^  pull_request:\s*$", trigger)


def test_publish_runs_only_for_version_tags() -> None:
    trigger = PUBLISH_WORKFLOW.split("\njobs:", maxsplit=1)[0]
    push = re.search(r"(?ms)^  push:\s*\n(.*?)(?=^  [\w-]+:|\Z)", trigger)

    assert push is not None
    assert re.findall(r"(?m)^    ([\w-]+):", push.group(1)) == ["tags"]
    assert re.findall(r'(?m)^      - ["\']?([^"\'\s]+)["\']?\s*$', push.group(1)) == ["v*"]
    assert "branches" not in push.group(1)
    assert "pull_request:" not in trigger


def test_ci_quality_job_uses_locked_dev_environment() -> None:
    quality = _job(CI_WORKFLOW, "quality")

    _assert_in_order(
        quality,
        "uv sync --locked --extra dev",
        "uv lock --check",
        "ruff check .",
        "black --check .",
    )


def test_ci_runs_the_full_test_suite_on_all_supported_python_versions() -> None:
    test = _job(CI_WORKFLOW, "test")

    for version in ("3.10", "3.11", "3.12", "3.13", "3.14"):
        assert version in test
    assert "uv sync --locked --extra dev" in test
    assert "pytest -s -vv tests" in test
    _assert_unconditional_test_matrix(test)


def test_ci_packages_only_after_quality_and_tests_and_smoke_tests_the_wheel() -> None:
    package = _job(CI_WORKFLOW, "package")

    assert re.search(r"needs:\s*\[?[^\n]*quality[^\n]*test", package)
    _assert_wheel_smoke(package)


def test_publish_validates_the_tag_and_package_before_uploading_artifacts() -> None:
    build = _job(PUBLISH_WORKFLOW, "validate-and-build")

    _assert_in_order(
        build,
        "uv sync --locked --extra dev",
        "pytest -s -vv tests",
        "ruff check .",
        "black --check .",
        "uv lock --check",
        "GITHUB_REF_NAME",
        "pyproject.toml",
        "uv build",
        "uv venv",
        "uv pip install",
        'cd "$SMOKE_DIR"',
        "import metricsifter",
        "metricsifter.__version__",
        'bin/metricsifter" --help',
        "actions/upload-artifact",
    )


def test_publish_job_uses_only_the_validated_artifact() -> None:
    publish = _job(PUBLISH_WORKFLOW, "publish-to-pypi")

    assert "needs: validate-and-build" in publish
    _assert_in_order(publish, "actions/download-artifact", "pypa/gh-action-pypi-publish")
    assert "metricsifter_pypi" in publish
    assert "id-token: write" in publish


def test_workflow_guards_reject_previous_regressions() -> None:
    quality = _job(CI_WORKFLOW, "quality")
    old_quality_order = _swap(quality, "uv sync --locked --extra dev", "uv lock --check")
    with pytest.raises(AssertionError):
        _assert_in_order(old_quality_order, "uv sync --locked --extra dev", "uv lock --check")

    publish = _job(PUBLISH_WORKFLOW, "validate-and-build")
    old_publish_order = _swap(publish, "uv sync --locked --extra dev", "uv lock --check")
    with pytest.raises(AssertionError):
        _assert_in_order(
            old_publish_order,
            "uv sync --locked --extra dev",
            "pytest -s -vv tests",
            "black --check .",
            "uv lock --check",
        )

    test = _job(CI_WORKFLOW, "test")
    conditional_test = test.replace(
        "      - name: Test\n",
        "      - name: Test\n        if: ${{ matrix.python-version == '3.11' }}\n",
    )
    with pytest.raises(AssertionError):
        _assert_unconditional_test_matrix(conditional_test)

    package = _job(CI_WORKFLOW, "package")
    without_installed_version = package.replace("metricsifter.__version__", "metricsifter")
    with pytest.raises(AssertionError):
        _assert_wheel_smoke(without_installed_version)
