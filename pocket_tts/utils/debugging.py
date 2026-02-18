import mlx.core as mx


def to_str(obj):
    if isinstance(obj, mx.array):
        return f"A(s={list(obj.shape)})"
    elif isinstance(obj, (list, tuple)):
        return "[" + ", ".join(to_str(o) for o in obj) + "]"
    elif isinstance(obj, dict):
        return "{" + ", ".join(f"{to_str(k)}: {to_str(v)}" for k, v in obj.items()) + "}"
    else:
        return str(obj)
