"""Command line interface for llama-swap YAML generator."""

import argparse
from pathlib import Path

import yaml

from .config import create_settings_from_config, load_config
from .generator import generate_full_config
from .template import write_config_template
from .validator import validate_yaml_file


def validate_model_dir(path_str: str) -> Path:
    """Validate the existence of model directory."""
    path = Path(path_str)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Model path '{path_str}' does not exist.")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Model path '{path_str}' is not a directory.")
    return path


def validate_binary_file(path_str: str) -> Path:
    """Validate the existence of binary file."""
    path = Path(path_str)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Binary path '{path_str}' does not exist.")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Binary path '{path_str}' is not a file.")
    return path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate LLM proxy configuration",
        prog="llama-swap-config-autogen",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    # init サブコマンド
    init_parser = subparsers.add_parser(
        "init",
        help="Generate config.yaml template",
    )
    init_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default="/dev/stdout",
        help="Output config file path",
    )
    init_parser.add_argument(
        "--model",
        type=validate_model_dir,
        action="append",
        required=True,
        help="Path to models directory (can specify multiple times)",
    )
    init_parser.add_argument(
        "--binary",
        type=validate_binary_file,
        required=True,
        help="Path to llama-server binary (required)",
    )

    # generate サブコマンド
    generate_parser = subparsers.add_parser("generate", help="Generate llama-swap config from config.yaml")
    generate_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration file (required)",
    )
    generate_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (default: print to stdout)",
    )

    # validate サブコマンド
    validate_parser = subparsers.add_parser("validate", help="Validate llama-swap YAML configuration")
    validate_parser.add_argument(
        "yaml_file",
        type=Path,
        help="Path to YAML file to validate (required)",
    )
    validate_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only show errors (suppress warnings and success messages)",
    )

    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()

    # Process subcommands
    if args.command == "init":
        # Process init subcommand

        write_config_template(
            args.output,
            args.model,
            args.binary,
        )
        return

    # Process generate subcommand
    if args.command == "generate":
        config = load_config(args.config)
        settings = create_settings_from_config(config, args.config)
        output_config = generate_full_config(settings, config)

        yaml_output = yaml.dump(
            output_config,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=float("inf"),
        )

        if args.output:
            with args.output.open("w", encoding="utf-8") as f:
                f.write(yaml_output)
        else:
            print(yaml_output, end="")

    # Process validate subcommand
    elif args.command == "validate":
        result = validate_yaml_file(args.yaml_file)

        if not args.quiet:
            print(result.format_report())
        elif not result.is_valid:
            # In quiet mode, only show errors
            for error in result.errors:
                print(f"ERROR: {error}")

        # Exit with error code if validation failed
        if not result.is_valid:
            exit(1)

    else:
        print("Usage:")
        print("  llama-swap-config-autogen init --model /path/to/models --binary /path/to/llama-server")
        print("  llama-swap-config-autogen generate --config config.yaml --output proxy-config.yaml")
        print("  llama-swap-config-autogen validate config.yaml")
        exit(1)


if __name__ == "__main__":
    main()
