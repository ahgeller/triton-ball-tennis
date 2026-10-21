"""tennis_tracker.paths: the model assets are spelled once and every consumer agrees with it."""
from pathlib import Path

from tennis_tracker import paths
from tennis_tracker.config import Config


def test_assets_sit_under_the_repository_models_folder():
    assert paths.ROOT == Path(__file__).resolve().parents[1]
    assert paths.MODELS == paths.ROOT / "models"
    for asset in (paths.GRIDTRACKNET_WEIGHTS, paths.PLAYER_ENGINE, paths.COURT_ENGINE, paths.BOUNCE_MODEL):
        assert asset.parent == paths.MODELS and asset.is_absolute()


def test_config_defaults_and_the_workspace_use_the_same_files():
    from finetune.workspace import REPO_WEIGHTS
    import evaluate_archive

    cfg = Config()
    assert (cfg.model_path, cfg.player_model_path, cfg.court_model_path) == (
        str(paths.GRIDTRACKNET_WEIGHTS), str(paths.PLAYER_ENGINE), str(paths.COURT_ENGINE))
    assert REPO_WEIGHTS == paths.GRIDTRACKNET_WEIGHTS
    assert evaluate_archive.MODELS["gridtracknet"] == paths.GRIDTRACKNET_WEIGHTS
