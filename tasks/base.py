from typing import Any, Dict, Tuple, Protocol
import mlx.core as mx
from models import trm

class Task(Protocol):
    """
    Abstract base class/Protocol for a Task.
    """

    def get_dataset(self, batch_size: int, **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Returns (train_loader, val_loader, meta).
        """
        ...

    def get_model_config(self, meta: Dict[str, Any]) -> trm.ModelConfig:
        """
        Returns the model configuration based on dataset metadata.
        """
        ...

    def loss_fn(self, model_outputs: Dict[str, mx.array], batch: Dict[str, mx.array], carry: Dict[str, Any]) -> Tuple[mx.array, mx.array, Dict[str, Any]]:
        """
        Computes loss and metrics.
        Returns:
            loss: scalar mx.array
            correct_count: scalar mx.array (number of correct predictions)
            stats: dict of additional statistics to log
        """
        ...
