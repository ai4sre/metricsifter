"""Tests for the argparse-based CLI (driving main(argv) directly, no subprocess)."""

import json

import pandas as pd
import pytest

from metricsifter import cli
from tests.conftest import make_synthetic


@pytest.fixture
def input_csv(tmp_path):
    path = tmp_path / "input.csv"
    make_synthetic().to_csv(path, index=True)
    return path


class TestRun:
    def test_writes_output_csv(self, tmp_path, input_csv):
        out = tmp_path / "out.csv"
        code = cli.main(["run", str(input_csv), "--output", str(out), "--index-col", "0", "--n-jobs", "1"])

        assert code == cli.EXIT_OK
        assert out.exists()
        result_df = pd.read_csv(out, index_col=0)
        assert set(result_df.columns) == {"failure_0", "failure_1", "failure_2"}

    def test_writes_report_json(self, tmp_path, input_csv):
        out = tmp_path / "out.csv"
        report = tmp_path / "report.json"
        code = cli.main(
            ["run", str(input_csv), "--output", str(out), "--report", str(report), "--index-col", "0", "--n-jobs", "1"]
        )

        assert code == cli.EXIT_OK
        payload = json.loads(report.read_text())
        assert set(payload["selected_metrics"]) == {"failure_0", "failure_1", "failure_2"}
        assert set(payload.keys()) == {
            "selected_metrics",
            "excluded",
            "metric_to_change_points",
            "metric_to_change_times",
            "segments",
            "selected_segment",
            "penalty_tuning",
            "bandwidth_tuning",
        }

    def test_stdout_when_no_output(self, capsys, input_csv):
        code = cli.main(["run", str(input_csv), "--index-col", "0", "--n-jobs", "1"])
        assert code == cli.EXIT_OK
        captured = capsys.readouterr()
        assert "failure_0" in captured.out

    def test_default_treats_every_column_as_metric(self, tmp_path):
        """Without --index-col no column may be swallowed as the row index."""
        out = tmp_path / "out.csv"
        path = tmp_path / "plain.csv"
        make_synthetic().to_csv(path, index=False)
        code = cli.main(["run", str(path), "--output", str(out), "--n-jobs", "1"])

        assert code == cli.EXIT_OK
        result_df = pd.read_csv(out, index_col=0)
        assert set(result_df.columns) == {"failure_0", "failure_1", "failure_2"}

    def test_parse_dates_flag(self, tmp_path):
        data = make_synthetic(as_datetime=True)
        path = tmp_path / "dt.csv"
        data.to_csv(path)
        report = tmp_path / "report.json"
        out = tmp_path / "out.csv"
        code = cli.main(
            [
                "run",
                str(path),
                "--output",
                str(out),
                "--report",
                str(report),
                "--index-col",
                "0",
                "--parse-dates",
                "--n-jobs",
                "1",
            ]
        )
        assert code == cli.EXIT_OK
        payload = json.loads(report.read_text())
        # DatetimeIndex input yields wall-clock change times in the report.
        assert payload["metric_to_change_times"] is not None


class TestInputErrors:
    def test_missing_file_returns_exit_2(self, tmp_path):
        code = cli.main(["run", str(tmp_path / "nope.csv")])
        assert code == cli.EXIT_INPUT_ERROR

    def test_empty_file_returns_exit_2(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        code = cli.main(["run", str(empty)])
        assert code == cli.EXIT_INPUT_ERROR

    def test_bad_arguments_exit_2(self):
        # argparse exits with code 2 on unknown args.
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["run"])  # missing required INPUT
        assert excinfo.value.code == 2
