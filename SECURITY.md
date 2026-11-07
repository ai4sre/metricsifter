# Security policy

MetricSifter is a library and a command-line tool that analyses time series
handed to it by its caller: DataFrames, CSV files (`metricsifter run`) and
parsed Prometheus `query_range` responses. It needs no privileges, makes no
network requests and executes nothing from its input. Issues that let crafted
input do more than produce a wrong selection or a clean error are treated as
vulnerabilities: code execution, reading or writing files other than the ones
named on the command line, or unbounded resource use out of proportion to the
input size. So are weaknesses in how releases are built and published to PyPI.

Out of scope: the research code under `experiments/` (it reproduces the paper
with pinned, older dependencies and is not part of the package), and
vulnerabilities in dependencies themselves (please report those upstream; a
dependency update is welcome as a regular pull request).

## Supported versions

Security fixes are made on the `main` branch and released in the next
version. Only the latest release on PyPI is supported.

## Reporting a vulnerability

Please do **not** open a public issue, pull request or discussion.

Report the vulnerability privately with GitHub's
[private vulnerability reporting](https://github.com/ai4sre/metricsifter/security/advisories/new)
(the "Report a vulnerability" button of the Security tab). If that is not
possible, contact the maintainer privately via the contact information on
https://github.com/yuuki.

Please include the MetricSifter version (`python -c "import metricsifter;
print(metricsifter.__version__)"`), the Python version, the input that
triggers the issue (or a script that builds it), the `Sifter` parameters or
command line, and the impact. You should receive an answer within a few
weeks; MetricSifter is maintained on a best-effort basis.

## Verifying releases

Releases are published from a `v*` tag by `.github/workflows/publish.yaml`
through PyPI Trusted Publishing, which also uploads PEP 740 attestations
linking each distribution to that workflow run. They are shown on the
package's PyPI page.
