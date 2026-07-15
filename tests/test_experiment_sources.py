"""Regression checks for experiment wrappers and their optional dependencies.

PyRCA remains optional for the standard test suite, so most source checks parse
the wrapper and the import check supplies only the external API surface it needs.
"""

import ast
import importlib
import logging
from pathlib import Path
import sys
from types import ModuleType

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def _project_configuration() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)


def _lock_configuration() -> dict:
    with (ROOT / "uv.lock").open("rb") as file:
        return tomllib.load(file)


def _stub_package(monkeypatch, name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _stub_module(monkeypatch, name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


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


def test_warning_suppression_uses_python_310_compatible_api():
    tree = _parse("experiments/localization/pyrca.py")

    incompatible_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "catch_warnings"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "warnings"
        and {keyword.arg for keyword in node.keywords} & {"action", "category"}
    ]

    assert incompatible_calls == []


def test_experiment_dependencies_are_limited_to_supported_python_versions():
    experiments = _project_configuration()["project"]["optional-dependencies"]["experiments"]

    for value in experiments:
        requirement = Requirement(value)
        assert requirement.marker is not None, value
        assert requirement.marker.evaluate({"python_version": "3.10"}), value
        assert requirement.marker.evaluate({"python_version": "3.11"}), value
        assert not requirement.marker.evaluate({"python_version": "3.12"}), value


def test_experiment_sklearn_constraint_is_compatible_with_pinned_pyrca():
    experiments = _project_configuration()["project"]["optional-dependencies"]["experiments"]
    requirement = Requirement(
        next(requirement for requirement in experiments if requirement.startswith("scikit-learn"))
    )

    assert Version("1.1.3") in requirement.specifier
    assert Version("1.2") not in requirement.specifier


def test_experiment_numpy_constraint_preserves_legacy_binary_abi():
    experiments = _project_configuration()["project"]["optional-dependencies"]["experiments"]
    requirement = Requirement(next(requirement for requirement in experiments if requirement.startswith("numpy")))

    assert Version("1.26.4") in requirement.specifier
    assert Version("2.0") not in requirement.specifier


def test_uv_declares_dev_and_experiments_extras_as_conflicting():
    conflicts = _project_configuration()["tool"]["uv"]["conflicts"]
    conflict_sets = [{item["extra"] for item in conflict} for conflict in conflicts]

    assert {"dev", "experiments"} in conflict_sets


def test_lock_metadata_matches_optional_dependency_groups():
    project_extras = _project_configuration()["project"]["optional-dependencies"]
    locked_project = next(package for package in _lock_configuration()["package"] if package["name"] == "metricsifter")
    locked_requirements = locked_project["metadata"]["requires-dist"]

    for extra in ("dev", "experiments"):
        project_names = {Requirement(value).name for value in project_extras[extra]}
        locked_extra = [item for item in locked_requirements if f"extra == '{extra}'" in item.get("marker", "")]
        assert {item["name"] for item in locked_extra} == project_names

    locked_experiments = [item for item in locked_requirements if "extra == 'experiments'" in item.get("marker", "")]
    assert all("python_full_version < '3.12'" in item["marker"] for item in locked_experiments)
    locked_sklearn = next(item for item in locked_experiments if item["name"] == "scikit-learn")
    assert Version("1.1.3") in SpecifierSet(locked_sklearn["specifier"])
    assert Version("1.2") not in SpecifierSet(locked_sklearn["specifier"])
    locked_numpy = next(item for item in locked_experiments if item["name"] == "numpy")
    assert Version("1.26.4") in SpecifierSet(locked_numpy["specifier"])
    assert Version("2.0") not in SpecifierSet(locked_numpy["specifier"])


def test_experiment_requirements_pin_pyrca_and_install_only_the_experiment_extra():
    requirements = (ROOT / "experiments/requirements.txt").read_text(encoding="utf-8").splitlines()

    assert '-e ".[experiments]"' in requirements
    assert "sfr-pyrca @ git+https://github.com/salesforce/PyRCA@d85512b" in requirements
    assert all("dev,experiments" not in requirement for requirement in requirements)


def test_pyrca_wrapper_imports_without_internal_logger(monkeypatch):
    class Dependency:
        pass

    networkx = _stub_module(monkeypatch, "networkx")
    networkx.exception = Dependency()
    _stub_module(monkeypatch, "threadpoolctl", threadpool_limits=lambda *_args, **_kwargs: None)
    _stub_package(monkeypatch, "priorknowledge")
    _stub_module(monkeypatch, "priorknowledge.base", PriorKnowledge=Dependency)
    _stub_module(
        monkeypatch,
        "priorknowledge.call_graph",
        get_forbits=lambda *_args, **_kwargs: [],
        prepare_init_graph=lambda *_args, **_kwargs: None,
    )
    _stub_package(monkeypatch, "pyrca")
    _stub_package(monkeypatch, "pyrca.analyzers")
    _stub_module(
        monkeypatch,
        "pyrca.analyzers.epsilon_diagnosis",
        EpsilonDiagnosis=Dependency,
        EpsilonDiagnosisConfig=Dependency,
    )
    _stub_module(monkeypatch, "pyrca.analyzers.ht", HT=Dependency, HTConfig=Dependency)
    _stub_package(monkeypatch, "pyrca.graphs")
    _stub_package(monkeypatch, "pyrca.graphs.causal")
    _stub_module(monkeypatch, "pyrca.graphs.causal.lingam", LiNGAM=Dependency, LiNGAMConfig=Dependency)
    _stub_module(monkeypatch, "pyrca.graphs.causal.pc", PC=Dependency, PCConfig=Dependency)
    _stub_module(monkeypatch, "experiments.localization.rcd", run_rcd=lambda *_args, **_kwargs: [])

    module_name = "experiments.localization.pyrca"
    sys.modules.pop(module_name, None)
    try:
        module = importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)

    assert isinstance(module.logger, logging.Logger)
