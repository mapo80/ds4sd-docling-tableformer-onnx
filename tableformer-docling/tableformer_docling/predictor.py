"""Lightweight wrapper around Docling's TableFormer components."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, MutableMapping, Optional, Sequence

import numpy as np
from huggingface_hub import snapshot_download

from docling_ibm_models.tableformer.common import read_config
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor

_LOGGER = logging.getLogger(__name__)


class TableFormerDocling:
    """Utility class to run Docling's TableFormer model outside Docling."""

    def __init__(
        self,
        *,
        artifacts_dir: Optional[Path] = None,
        repo_id: str = "ds4sd/docling-models",
        revision: str = "main",
        variant: str = "fast",
        device: str = "cpu",
        num_threads: int = 4,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.variant = variant
        self.device = device
        self.num_threads = num_threads

        if artifacts_dir is None:
            artifacts_dir = Path(__file__).resolve().parent.parent / "artifacts"
        self.artifacts_dir = Path(artifacts_dir)

        self._predictor: Optional[TFPredictor] = None
        self._config: Optional[MutableMapping[str, object]] = None

    @property
    def predictor(self) -> TFPredictor:
        if self._predictor is None:
            self._load_predictor()
        assert self._predictor is not None
        return self._predictor

    def _download_artifacts(self) -> Path:
        target_dir = self.artifacts_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=self.repo_id,
            revision=self.revision,
            allow_patterns=[f"model_artifacts/tableformer/{self.variant}/*"],
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        return target_dir / "model_artifacts" / "tableformer" / self.variant

    def _load_predictor(self) -> None:
        model_dir = self.artifacts_dir / "model_artifacts" / "tableformer" / self.variant
        if not model_dir.exists():
            _LOGGER.info("Downloading TableFormer artifacts to %s", model_dir)
            model_dir = self._download_artifacts()

        config_path = model_dir / "tm_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing configuration file: {config_path}")

        config = read_config(config_path)
        config.setdefault("model", {})
        config["model"]["save_dir"] = str(model_dir)

        self._config = config
        self._predictor = TFPredictor(
            config,
            device=self.device,
            num_threads=self.num_threads,
        )

    def predict_page(
        self,
        image: np.ndarray,
        table_bboxes: Optional[Sequence[Sequence[float]]] = None,
        *,
        tokens: Optional[Iterable[MutableMapping[str, object]]] = None,
        do_cell_matching: bool = True,
        correct_overlapping_cells: bool = False,
    ) -> List[MutableMapping[str, object]]:
        """Run inference on a page image."""

        if image.ndim != 3:
            raise ValueError("Expected an RGB image array with shape (H, W, C).")

        height, width = image.shape[:2]
        if table_bboxes is None:
            table_bboxes = [[0.0, 0.0, float(width), float(height)]]

        page_input: MutableMapping[str, object] = {
            "image": image,
            "width": float(width),
            "height": float(height),
        }

        if tokens is not None:
            page_input["tokens"] = list(tokens)
        else:
            page_input["tokens"] = []

        effective_matching = do_cell_matching and bool(page_input["tokens"])

        predictor = self.predictor
        predictions = predictor.multi_table_predict(
            page_input,
            [list(bbox) for bbox in table_bboxes],
            do_matching=effective_matching,
            correct_overlapping_cells=correct_overlapping_cells,
        )
        return predictions

    def get_config(self) -> MutableMapping[str, object]:
        if self._config is None:
            self._load_predictor()
        assert self._config is not None
        return self._config

