"""Static checks for experiment wrappers with optional dependencies.

The experiment modules cannot be imported in the core test environment because
PyRCA is optional. Parsing their syntax still catches unresolved runtime names
without making PyRCA a prerequisite for the standard test suite.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse(relative_path: str) -> ast.Module:
    return ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))


def _import_sources(tree: ast.Module) -> dict[str, str]:
    sources: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                sources[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                sources[alias.asname or alias.name] = node.module
    return sources


def test_rcd_source_imports_every_runtime_dependency_from_supported_modules():
    imports = _import_sources(_parse("experiments/localization/rcd.py"))

    expected = {
        "defaultdict": "collections",
        "joblib": "joblib",
        "RCD": "pyrca.analyzers.rcd",
        "RCDConfig": "pyrca.analyzers.rcd",
        "gsq": "pyrca.thirdparty.causallearn.utils.cit",
        "zscore": "scipy.stats",
        "threadpool_limits": "threadpoolctl",
    }
    assert expected.items() <= imports.items()


def test_unknown_building_step_error_references_the_requested_step():
    tree = _parse("experiments/localization/pyrca.py")
    run_rca = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_rca")

    matching_errors = []
    for node in ast.walk(run_rca):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        if not isinstance(node.exc.func, ast.Name) or node.exc.func.id != "ValueError" or not node.exc.args:
            continue
        message = node.exc.args[0]
        if not isinstance(message, ast.JoinedStr):
            continue
        constants = "".join(value.value for value in message.values if isinstance(value, ast.Constant))
        names = {
            value.value.id
            for value in message.values
            if isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name)
        }
        if "Model " in constants and "not supported." in constants and "building_step" in names:
            matching_errors.append(node)

    assert len(matching_errors) == 1
