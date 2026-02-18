from abc import ABC, abstractmethod

import mlx.core as mx
import mlx.nn as nn


def _named_modules(module: nn.Module, prefix: str = ""):
    """Recursively yield (name, module) pairs, similar to PyTorch's named_modules()."""
    yield prefix, module
    for name, child in module.children().items():
        if isinstance(child, nn.Module):
            full_name = f"{prefix}.{name}" if prefix else name
            yield from _named_modules(child, full_name)
        elif isinstance(child, list):
            for i, item in enumerate(child):
                if isinstance(item, nn.Module):
                    full_name = f"{prefix}.{name}.{i}" if prefix else f"{name}.{i}"
                    yield from _named_modules(item, full_name)


def _get_stateful_modules(module: nn.Module) -> list[tuple[str, "StatefulModule"]]:
    """Get a cached list of (name, module) pairs for all StatefulModules.

    This avoids walking the module tree on every call to init_states/increment_steps.
    The list is computed once and cached on the module.
    """
    cache_attr = "_cached_stateful_modules"
    cached = getattr(module, cache_attr, None)
    if cached is not None:
        return cached
    result = [
        (name, mod)
        for name, mod in _named_modules(module)
        if isinstance(mod, StatefulModule)
    ]
    # Cache on the module object (bypass nn.Module __setattr__)
    object.__setattr__(module, cache_attr, result)
    return result


def init_states(
    model: nn.Module, batch_size: int, sequence_length: int
) -> dict[str, dict[str, mx.array]]:
    result = {}
    for module_name, module in _get_stateful_modules(model):
        module_state = module.init_state(batch_size, sequence_length=sequence_length)
        result[module_name] = module_state
    return result


def increment_steps(
    module: nn.Module, model_state: dict[str, dict[str, mx.array]], increment: int = 1
):
    for module_name, mod in _get_stateful_modules(module):
        mod.increment_step(model_state[module_name], increment)


class StatefulModule(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self._module_absolute_name = None

    @abstractmethod
    def init_state(self, batch_size: int, sequence_length: int):
        """Initialize the state."""
        raise NotImplementedError

    def increment_step(self, state: dict, increment: int = 1):
        pass

    def get_state(self, model_state: dict[str, dict[str, mx.array]]) -> dict[str, mx.array]:
        """Get the state for this module from the model state."""
        return model_state[self._module_absolute_name]
