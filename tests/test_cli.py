"""Tests for CLI interface."""
import json
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock


@pytest.fixture
def cli_runner():
    return CliRunner()


class TestCliPredict:
    def test_predict_command_exists(self, cli_runner):
        from hn_simulator.cli import main
        result = cli_runner.invoke(main, ["predict", "--help"])
        assert result.exit_code == 0
        assert "title" in result.output.lower()

    def test_predict_requires_title(self, cli_runner):
        from hn_simulator.cli import main
        result = cli_runner.invoke(main, ["predict"])
        assert result.exit_code != 0

    @patch("hn_simulator.cli.HNSimulator")
    def test_predict_outputs_json(self, mock_sim_cls, cli_runner):
        from hn_simulator.cli import main
        mock_sim = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "predicted_score": 42.5,
            "predicted_comments": 15.0,
            "reception_label": "hot",
            "confidence": 0.75,
            "label_distribution": {},
            "simulated_comments": [],
            "similar_stories": [],
        }
        mock_sim.simulate.return_value = mock_result
        mock_sim_cls.return_value = mock_sim

        result = cli_runner.invoke(main, [
            "predict",
            "--title", "Show HN: My Project",
            "--description", "A cool project",
            "--no-comments",
            "--json",
        ])
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["predicted_score"] == 42.5

    @patch("hn_simulator.cli.HNSimulator")
    def test_predict_human_readable_output(self, mock_sim_cls, cli_runner):
        from hn_simulator.cli import main
        mock_sim = MagicMock()
        mock_result = MagicMock()
        mock_result.predicted_score = 42.5
        mock_result.predicted_comments = 15.0
        mock_result.reception_label = "hot"
        mock_result.confidence = 0.75
        mock_result.label_distribution = {"flop": 0.05, "moderate": 0.15, "hot": 0.55, "viral": 0.25}
        mock_result.simulated_comments = []
        mock_result.similar_stories = []
        mock_result.percentile = None
        mock_result.expected_score = None
        mock_result.shap_features = []
        mock_result.time_recommendation = ""
        mock_sim.simulate.return_value = mock_result
        mock_sim_cls.return_value = mock_sim

        result = cli_runner.invoke(main, [
            "predict",
            "--title", "Show HN: My Project",
            "--no-comments",
        ])
        assert result.exit_code == 0
        assert "42" in result.output
        assert "HOT" in result.output
        assert "75%" in result.output


class TestCliTrain:
    def test_train_command_exists(self, cli_runner):
        from hn_simulator.cli import main
        result = cli_runner.invoke(main, ["train", "--help"])
        assert result.exit_code == 0

    def test_train_has_sample_size_option(self, cli_runner):
        from hn_simulator.cli import main
        result = cli_runner.invoke(main, ["train", "--help"])
        assert result.exit_code == 0
        assert "sample-size" in result.output

    def test_train_has_min_score_option(self, cli_runner):
        from hn_simulator.cli import main
        result = cli_runner.invoke(main, ["train", "--help"])
        assert result.exit_code == 0
        assert "min-score" in result.output


class TestCliFetch:
    def test_fetch_command_exists(self, cli_runner):
        from hn_simulator.cli import main
        result = cli_runner.invoke(main, ["fetch", "--help"])
        assert result.exit_code == 0

    def test_fetch_has_sample_size_option(self, cli_runner):
        from hn_simulator.cli import main
        result = cli_runner.invoke(main, ["fetch", "--help"])
        assert result.exit_code == 0
        assert "sample-size" in result.output


class TestCliBuildIndex:
    def test_build_index_command_exists(self, cli_runner):
        from hn_simulator.cli import main
        result = cli_runner.invoke(main, ["build-index", "--help"])
        assert result.exit_code == 0

    def test_build_index_has_sample_size_option(self, cli_runner):
        from hn_simulator.cli import main
        result = cli_runner.invoke(main, ["build-index", "--help"])
        assert result.exit_code == 0
        assert "sample-size" in result.output


class TestCliBacktest:
    def test_backtest_command_exists(self, cli_runner):
        from hn_simulator.cli import main
        result = cli_runner.invoke(main, ["backtest", "--help"])
        assert result.exit_code == 0

    def test_backtest_has_features_dir_option(self, cli_runner):
        from hn_simulator.cli import main
        result = cli_runner.invoke(main, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "features-dir" in result.output


class TestCliSuggestLoop:
    def test_suggest_loop_command_exists(self, cli_runner):
        from hn_simulator.cli import main
        result = cli_runner.invoke(main, ["suggest-loop", "--help"])
        assert result.exit_code == 0

    def test_suggest_loop_requires_title(self, cli_runner):
        from hn_simulator.cli import main
        result = cli_runner.invoke(main, ["suggest-loop"])
        assert result.exit_code != 0

    def test_suggest_loop_has_max_iterations_option(self, cli_runner):
        from hn_simulator.cli import main
        result = cli_runner.invoke(main, ["suggest-loop", "--help"])
        assert result.exit_code == 0
        assert "max-iterations" in result.output
