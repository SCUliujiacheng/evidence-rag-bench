from pathlib import Path

from evidence_rag_bench.config import get_settings


def test_settings_resolve_project_paths(tmp_path: Path) -> None:
    settings = get_settings(tmp_path)

    assert settings.project_root == tmp_path.resolve()
    assert settings.artifacts_dir == tmp_path / "artifacts"
    assert settings.corpus_dir == tmp_path / "data" / "corpus"
    assert settings.eval_dir == tmp_path / "data" / "eval"
