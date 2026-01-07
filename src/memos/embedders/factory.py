from typing import Any, ClassVar

from memos.configs.embedder import EmbedderConfigFactory
from memos.embedders.ark import ArkEmbedder
from memos.embedders.base import BaseEmbedder
from memos.embedders.ollama import OllamaEmbedder
from memos.embedders.sentence_transformer import SenTranEmbedder
from memos.embedders.universal_api import UniversalAPIEmbedder
from memos.log import get_logger
from memos.memos_tools.singleton import singleton_factory


logger = get_logger(__name__)


class EmbedderFactory(BaseEmbedder):
    """Factory class for creating embedder instances."""

    backend_to_class: ClassVar[dict[str, Any]] = {
        "ollama": OllamaEmbedder,
        "sentence_transformer": SenTranEmbedder,
        "ark": ArkEmbedder,
        "universal_api": UniversalAPIEmbedder,
    }

    @classmethod
    @singleton_factory()
    def from_config(cls, config_factory: EmbedderConfigFactory) -> BaseEmbedder:
        backend = config_factory.backend
        if backend not in cls.backend_to_class:
            raise ValueError(f"Invalid backend: {backend}")
        embedder_class = cls.backend_to_class[backend]

        # Log selected embedder config (sanitized) to help diagnose runtime behavior.
        try:
            cfg = config_factory.config
            cfg_dict = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)
            # Remove secrets
            cfg_dict.pop("api_key", None)
            cfg_dict.pop("headers_extra", None)
            logger.info(
                "Embedder initialized: backend=%s, config=%s",
                backend,
                {k: cfg_dict.get(k) for k in ("model_name_or_path", "api_base", "base_url", "provider") if k in cfg_dict},
            )
        except Exception:
            # Never fail initialization due to logging.
            pass

        return embedder_class(config_factory.config)
