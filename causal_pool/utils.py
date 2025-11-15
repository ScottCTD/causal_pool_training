def normalize_model_name(model_name: str) -> str:
    """Normalize model name for file naming."""
    return model_name.split("/")[-1].replace("-", "_").lower()