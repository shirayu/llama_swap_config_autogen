from pathlib import Path

import yaml


def generate_config_template(models_dirs: list[Path] | None = None, binary_path: Path | str = "") -> str:
    template_path = Path(__file__).parent / "template.yaml"

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    try:
        with template_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse template YAML: {e}") from e

    if models_dirs:
        config["models"] = [str(path) for path in models_dirs]

    if binary_path and "macros" in config:
        config["macros"]["binary"] = str(binary_path)

    return yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)


def write_config_template(
    output_path: Path, models_dirs: list[Path] | None = None, binary_path: Path | str = ""
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write(generate_config_template(models_dirs, binary_path))
