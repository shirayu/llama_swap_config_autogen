"""Tests for config generation, including fit-params-based VRAM estimation."""

from pathlib import Path
from unittest.mock import patch

import yaml

from llama_swap_config_autogen.config import create_settings_from_config, load_config
from llama_swap_config_autogen.fit_params import FitParamsCache
from llama_swap_config_autogen.generator import (
    build_capabilities,
    build_model_metadata,
    estimate_vram_gib,
    extract_context_length,
    extract_extra_fit_args,
    extract_ngl,
    generate_full_config,
)
from llama_swap_config_autogen.gguf_metadata import GGUFMetadata, GGUFMetadataCache, MmprojModalities
from llama_swap_config_autogen.models import MacroConfig


def _make_metadata(mtime: float = 0.0, size: int = 1024, context_length: int = 4096) -> GGUFMetadata:
    return GGUFMetadata(
        mtime=mtime,
        size=size,
        context_length=context_length,
    )


# ---------------------------------------------------------------------------
# extract_ngl / extract_context_length
# ---------------------------------------------------------------------------


class TestExtractNgl:
    def test_found(self):
        assert extract_ngl("llama-server -m model.gguf -ngl 99 --port 8080") == 99

    def test_found_long_flag(self):
        assert extract_ngl("llama-server -m model.gguf --n-gpu-layers 25 --port 8080") == 25

    def test_not_found(self):
        assert extract_ngl("llama-server -m model.gguf --port 8080") == 0

    def test_zero(self):
        assert extract_ngl("-ngl 0") == 0


class TestExtractContextLength:
    def test_short_flag(self):
        assert extract_context_length("-c 8192 --port 8080", fallback=4096) == 8192

    def test_long_flag(self):
        assert extract_context_length("--ctx-size 16384", fallback=4096) == 16384

    def test_zero_uses_fallback(self):
        assert extract_context_length("-c 0", fallback=32768) == 32768

    def test_fallback(self):
        assert extract_context_length("no context flag here", fallback=4096) == 4096


class TestExtractExtraFitArgs:
    def test_no_offload_flags(self):
        assert extract_extra_fit_args("-ngl 99 -c 4096") == []

    def test_cpu_moe_flag(self):
        assert extract_extra_fit_args("-ngl 99 --cpu-moe") == ["--cpu-moe"]

    def test_n_cpu_moe_flag(self):
        assert extract_extra_fit_args("-ngl 99 --n-cpu-moe 12") == ["--n-cpu-moe", "12"]

    def test_ot_flag(self):
        assert extract_extra_fit_args("-ngl 99 -ot '.*.ffn_.*_exps.=CPU'") == ["-ot", "'.*.ffn_.*_exps.=CPU'"]


# ---------------------------------------------------------------------------
# estimate_vram_gib / build_model_metadata
# ---------------------------------------------------------------------------


class TestEstimateVramGib:
    def test_returns_none_without_llama_bin(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        metadata_cache = GGUFMetadataCache()
        fit_cache = FitParamsCache()

        result = estimate_vram_gib(model, "-ngl 99 -c 4096", 0, metadata_cache, fit_cache, None, {})
        assert result is None

    def test_delegates_to_fit_params(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        metadata_cache = GGUFMetadataCache()
        fit_cache = FitParamsCache()

        with (
            patch(
                "llama_swap_config_autogen.generator.get_gguf_metadata",
                return_value=_make_metadata(),
            ),
            patch(
                "llama_swap_config_autogen.generator.estimate_vram_gib_via_fit_params",
                return_value=12.5,
            ) as mock_fit,
        ):
            result = estimate_vram_gib(
                model, "-ngl 99 -c 4096 --cache-type-k q8_0", 0, metadata_cache, fit_cache, ["/app/llama"], {}
            )

        assert result == 12.5
        mock_fit.assert_called_once()
        _, kwargs = mock_fit.call_args
        assert kwargs["ngl"] == 99
        assert kwargs["ctx"] == 4096
        assert kwargs["cache_type_k"] == "q8_0"

    def test_returns_none_on_exception(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        metadata_cache = GGUFMetadataCache()
        fit_cache = FitParamsCache()

        with patch(
            "llama_swap_config_autogen.generator.get_gguf_metadata",
            side_effect=RuntimeError("read error"),
        ):
            result = estimate_vram_gib(model, "-ngl 99 -c 4096", 0, metadata_cache, fit_cache, ["/app/llama"], {})
        assert result is None


class TestBuildModelMetadata:
    def test_includes_vram_estimate_when_llama_bin_provided(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        metadata_cache = GGUFMetadataCache()
        fit_cache = FitParamsCache()

        with (
            patch(
                "llama_swap_config_autogen.generator.get_gguf_metadata",
                return_value=_make_metadata(),
            ),
            patch(
                "llama_swap_config_autogen.generator.estimate_vram_gib_via_fit_params",
                return_value=10.0,
            ),
        ):
            metadata, _ = build_model_metadata(
                "model",
                model,
                "-ngl 99 -c 4096",
                metadata_cache,
                fit_params_cache=fit_cache,
                llama_bin=["/app/llama"],
                path_prefix_map={},
                vram_estimation=True,
            )

        assert metadata["estimated_vram_bytes"] == round(10.0 * 1024**3)

    def test_omits_vram_estimate_without_fit_params_cache(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        metadata_cache = GGUFMetadataCache()

        with patch(
            "llama_swap_config_autogen.generator.get_gguf_metadata",
            return_value=_make_metadata(),
        ):
            metadata, _ = build_model_metadata(
                "model",
                model,
                "-ngl 99 -c 4096",
                metadata_cache,
                fit_params_cache=None,
                llama_bin=None,
                vram_estimation=True,
            )

        assert "estimated_vram_bytes" not in metadata


# ---------------------------------------------------------------------------
# Integration: generate_full_config
# ---------------------------------------------------------------------------


def _write_config(
    config_path: Path, models_dir: Path, vram_estimation: bool = False, extra: dict | None = None
) -> None:
    config: dict = {
        "models": [str(models_dir)],
        "vram_estimation": vram_estimation,
        "macros": {
            "binary": "/app/llama-server",
            "default-params": "-ngl 99 --ctx-size 4096",
        },
    }
    if extra:
        config.update(extra)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


class TestGenerateFullConfigIntegration:
    def test_model_metadata_omits_vram_when_llama_bin_not_provided(self, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "llama3"
        model_dir.mkdir(parents=True)
        model_file = model_dir / "llama3-Q4_K_M.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        config_path = tmp_path / "config.yaml"
        _write_config(config_path, models_dir, vram_estimation=True)

        config = load_config(config_path)
        settings = create_settings_from_config(config, config_path)
        result = generate_full_config(settings, config)

        model = result["models"]["llama3:Q4_K_M"]
        assert "estimated_vram_bytes" not in model["metadata"]

    def test_model_metadata_includes_vram_estimate_with_llama_bin(self, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "llama3"
        model_dir.mkdir(parents=True)
        model_file = model_dir / "llama3-Q4_K_M.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        config_path = tmp_path / "config.yaml"
        _write_config(config_path, models_dir, vram_estimation=True)

        config = load_config(config_path)
        settings = create_settings_from_config(config, config_path, llama_bin=["/app/llama"])

        with (
            patch(
                "llama_swap_config_autogen.generator.get_gguf_metadata",
                return_value=_make_metadata(mtime=model_file.stat().st_mtime, size=model_file.stat().st_size),
            ),
            patch(
                "llama_swap_config_autogen.generator.estimate_vram_gib_via_fit_params",
                return_value=8.0,
            ),
        ):
            result = generate_full_config(settings, config)

        model = result["models"]["llama3:Q4_K_M"]
        assert isinstance(model["metadata"]["estimated_vram_bytes"], int)

    def test_path_prefix_map_rewrites_model_path_in_cmd(self, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "llama3"
        model_dir.mkdir(parents=True)
        model_file = model_dir / "llama3-Q4_K_M.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        config_path = tmp_path / "config.yaml"
        _write_config(
            config_path,
            models_dir,
            vram_estimation=False,
            extra={"path_prefix_map": {str(models_dir): "/models"}},
        )

        config = load_config(config_path)
        settings = create_settings_from_config(config, config_path)
        result = generate_full_config(settings, config)

        cmd = result["models"]["llama3:Q4_K_M"]["cmd"]
        assert "/models/llama3/llama3-Q4_K_M.gguf" in cmd
        assert str(model_file) not in cmd

    def test_model_patterns_can_match_generated_model_id(self, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "Gemma-4-12B"
        model_dir.mkdir(parents=True)
        model_file = model_dir / "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
        model_file.write_bytes(b"\x00")

        config_path = tmp_path / "config.yaml"
        _write_config(
            config_path,
            models_dir,
            vram_estimation=False,
            extra={
                "macros": {
                    "binary": "/app/llama-server",
                    "default-params": "--ctx-size 32768",
                    "gemma-12b-xl-params": "--ctx-size 16384",
                },
                "model_patterns": {
                    "gemma-4-12b:Q4_K_XL": "gemma-12b-xl-params",
                },
            },
        )

        config = load_config(config_path)
        settings = create_settings_from_config(config, config_path)
        result = generate_full_config(settings, config)

        assert "${gemma-12b-xl-params}" in result["models"]["gemma-4-12b:Q4_K_XL"]["cmd"]

    def test_model_pattern_can_suppress_base_model(self, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "Gemma-4-12B"
        model_dir.mkdir(parents=True)
        model_file = model_dir / "gemma-4-12B-it-Q4_K_M.gguf"
        model_file.write_bytes(b"\x00")

        config_path = tmp_path / "config.yaml"
        _write_config(
            config_path,
            models_dir,
            vram_estimation=False,
            extra={
                "macros": {
                    "binary": "/app/llama-server",
                    "default-params": "--ctx-size 32768",
                    "gemma-4-base": "--ctx-size 32768",
                    "gemma-4-long": "--ctx-size 49152",
                },
                "model_patterns": {
                    "gemma-4-": {
                        "macro": "gemma-4-base",
                        "emit_base": False,
                    }
                },
                "variants": [
                    {
                        "base_pattern": "gemma-4-",
                        "suffix": " (32k)",
                        "macro": "gemma-4-base",
                    },
                    {
                        "base_pattern": "gemma-4-",
                        "suffix": " (48k)",
                        "macro": "gemma-4-long",
                    },
                ],
            },
        )

        config = load_config(config_path)
        settings = create_settings_from_config(config, config_path)
        result = generate_full_config(settings, config)

        assert "gemma-4-12b:Q4_K_M" not in result["models"]
        assert "gemma-4-12b:Q4_K_M--32k" in result["models"]
        assert "gemma-4-12b:Q4_K_M--48k" in result["models"]

    def test_sidecar_json_merges_into_metadata(self, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "llama3"
        model_dir.mkdir(parents=True)
        model_file = model_dir / "llama3-Q4_K_M.gguf"
        model_file.write_bytes(b"\x00" * 1024)
        sidecar_file = model_dir / "llama3-Q4_K_M.json"
        sidecar_file.write_text('{"notes": "hand curated", "model_family": "custom-family"}', encoding="utf-8")

        config_path = tmp_path / "config.yaml"
        _write_config(config_path, models_dir, vram_estimation=False)

        config = load_config(config_path)
        settings = create_settings_from_config(config, config_path)

        result = generate_full_config(settings, config)

        model = result["models"]["llama3:Q4_K_M"]
        assert model["metadata"]["notes"] == "hand curated"
        assert model["metadata"]["model_family"] == "custom-family"


class TestBuildCapabilities:
    def test_no_mmproj_yields_text_only(self, tmp_path):
        model_file = tmp_path / "model-Q4_K_M.gguf"
        model_file.write_bytes(b"\x00")

        capabilities = build_capabilities("default-params", MacroConfig(), model_file, None, None, mmproj_path=None)

        assert capabilities.in_ == ["text"]

    def test_vision_mmproj_yields_image(self, tmp_path):
        model_file = tmp_path / "model-Q4_K_M.gguf"
        model_file.write_bytes(b"\x00")
        mmproj_file = tmp_path / "mmproj-F16.gguf"
        mmproj_file.write_bytes(b"\x00")

        with patch(
            "llama_swap_config_autogen.generator.read_mmproj_modalities",
            return_value=MmprojModalities(has_vision=True, has_audio=False),
        ):
            capabilities = build_capabilities(
                "default-params", MacroConfig(), model_file, None, None, mmproj_path=mmproj_file
            )

        assert capabilities.in_ == ["text", "image"]

    def test_audio_mmproj_yields_audio_not_image(self, tmp_path):
        model_file = tmp_path / "model-Q4_K_M.gguf"
        model_file.write_bytes(b"\x00")
        mmproj_file = tmp_path / "mmproj-ultravox-F16.gguf"
        mmproj_file.write_bytes(b"\x00")

        with patch(
            "llama_swap_config_autogen.generator.read_mmproj_modalities",
            return_value=MmprojModalities(has_vision=False, has_audio=True),
        ):
            capabilities = build_capabilities(
                "default-params", MacroConfig(), model_file, None, None, mmproj_path=mmproj_file
            )

        assert capabilities.in_ == ["text", "audio"]

    def test_vision_and_audio_mmproj_yields_both(self, tmp_path):
        model_file = tmp_path / "model-Q4_K_M.gguf"
        model_file.write_bytes(b"\x00")
        mmproj_file = tmp_path / "mmproj-F16.gguf"
        mmproj_file.write_bytes(b"\x00")

        with patch(
            "llama_swap_config_autogen.generator.read_mmproj_modalities",
            return_value=MmprojModalities(has_vision=True, has_audio=True),
        ):
            capabilities = build_capabilities(
                "default-params", MacroConfig(), model_file, None, None, mmproj_path=mmproj_file
            )

        assert capabilities.in_ == ["text", "image", "audio"]

    def test_mmproj_without_modality_metadata_falls_back_to_image(self, tmp_path):
        model_file = tmp_path / "model-Q4_K_M.gguf"
        model_file.write_bytes(b"\x00")
        mmproj_file = tmp_path / "mmproj-F16.gguf"
        mmproj_file.write_bytes(b"\x00")

        with patch(
            "llama_swap_config_autogen.generator.read_mmproj_modalities",
            return_value=MmprojModalities(has_vision=False, has_audio=False),
        ):
            capabilities = build_capabilities(
                "default-params", MacroConfig(), model_file, None, None, mmproj_path=mmproj_file
            )

        assert capabilities.in_ == ["text", "image"]
