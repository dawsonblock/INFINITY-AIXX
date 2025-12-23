"""Logger smoke test.

Imports kept inside the test so pytest collection doesn't pull in torch/tensorboard.
"""


def test_logger_writes_files(tmp_path):
    from infinity_dual_hybrid.logger import UnifiedLogger, LoggerConfig

    lg = UnifiedLogger(
        LoggerConfig(
            run_dir=str(tmp_path),
            use_console=False,
            use_csv=True,
            use_jsonl=True,
            use_tensorboard=False,
        )
    )
    lg.log({"a": 1, "b": 2.5})
    lg.close()

    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "metrics.jsonl").exists()
