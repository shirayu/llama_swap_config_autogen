"""Tests for VRAM estimation via llama.cpp's fit-params tool."""

import subprocess
from unittest.mock import patch

from llama_swap_config_autogen.fit_params import (
    FitParamsCache,
    apply_path_prefix_map,
    build_cache_key,
    estimate_vram_gib_via_fit_params,
    parse_fit_print_output,
)


class TestApplyPathPrefixMap:
    def test_no_match_returns_original(self, tmp_path):
        model = tmp_path / "model.gguf"
        assert apply_path_prefix_map(model, {}) == str(model)

    def test_rewrites_matching_prefix(self, tmp_path):
        model = tmp_path / "models" / "foo.gguf"
        result = apply_path_prefix_map(model, {str(tmp_path / "models"): "/models"})
        assert result == "/models/foo.gguf"

    def test_uses_longest_matching_prefix(self, tmp_path):
        model = tmp_path / "models" / "special" / "foo.gguf"
        prefix_map = {
            str(tmp_path / "models"): "/models",
            str(tmp_path / "models" / "special"): "/special-models",
        }
        result = apply_path_prefix_map(model, prefix_map)
        assert result == "/special-models/foo.gguf"


class TestParseFitPrintOutput:
    def test_sums_gpu_device_rows_excluding_host(self):
        output = "CUDA0 14408 79 407 \nHost 721 0 32\n"
        result = parse_fit_print_output(output)
        assert result == (14408 + 79 + 407) / 1024

    def test_sums_multiple_gpu_devices(self):
        output = "CUDA0 1000 100 50 \nCUDA1 2000 200 100 \nHost 500 0 10\n"
        result = parse_fit_print_output(output)
        assert result == (1000 + 100 + 50 + 2000 + 200 + 100) / 1024

    def test_returns_none_for_unparsable_output(self):
        assert parse_fit_print_output("garbage output\nno rows here") is None

    def test_ignores_log_lines_mixed_in(self):
        output = "0.00.110 I some log line here\nCUDA0 100 10 5 \nHost 50 0 2\n"
        result = parse_fit_print_output(output)
        assert result == (100 + 10 + 5) / 1024


class TestBuildCacheKey:
    def test_same_inputs_produce_same_key(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        key1 = build_cache_key(model, ngl=99, ctx=4096, extra_args=())
        key2 = build_cache_key(model, ngl=99, ctx=4096, extra_args=())
        assert key1 == key2

    def test_different_ngl_produces_different_key(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        key1 = build_cache_key(model, ngl=99, ctx=4096, extra_args=())
        key2 = build_cache_key(model, ngl=50, ctx=4096, extra_args=())
        assert key1 != key2

    def test_different_extra_args_produces_different_key(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        key1 = build_cache_key(model, ngl=99, ctx=4096, extra_args=())
        key2 = build_cache_key(model, ngl=99, ctx=4096, extra_args=("--cpu-moe",))
        assert key1 != key2


class TestFitParamsCache:
    def test_get_returns_none_for_unknown_key(self):
        cache = FitParamsCache()
        assert cache.get("unknown") is None

    def test_set_and_get(self):
        cache = FitParamsCache()
        cache.set("key1", 12.5)
        assert cache.get("key1") == 12.5

    def test_save_and_load_roundtrip(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        cache = FitParamsCache()
        cache.set("key1", 12.5)

        with patch("llama_swap_config_autogen.fit_params.CACHE_PATH", cache_file):
            cache.save()
            loaded = FitParamsCache.load()

        assert loaded.get("key1") == 12.5

    def test_load_ignores_corrupt_cache(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        cache_file.write_text("not json", encoding="utf-8")
        with patch("llama_swap_config_autogen.fit_params.CACHE_PATH", cache_file):
            cache = FitParamsCache.load()
        assert cache.entries == {}


class TestEstimateVramGibViaFitParams:
    def test_returns_none_when_subprocess_fails(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        cache = FitParamsCache()

        with patch(
            "llama_swap_config_autogen.fit_params.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "llama"),
        ):
            result = estimate_vram_gib_via_fit_params(
                llama_bin=["/app/llama"],
                path_model=model,
                ngl=99,
                ctx=4096,
                cache_type_k=None,
                cache_type_v=None,
                extra_args=[],
                path_prefix_map={},
                cache=cache,
            )
        assert result is None

    def test_parses_successful_output_and_caches(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        cache = FitParamsCache()

        fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="CUDA0 1000 100 50 \nHost 10 0 1\n")
        with patch("llama_swap_config_autogen.fit_params.subprocess.run", return_value=fake_result) as mock_run:
            result = estimate_vram_gib_via_fit_params(
                llama_bin=["/app/llama"],
                path_model=model,
                ngl=99,
                ctx=4096,
                cache_type_k="q8_0",
                cache_type_v="q8_0",
                extra_args=[],
                path_prefix_map={},
                cache=cache,
            )
        assert result == (1000 + 100 + 50) / 1024
        mock_run.assert_called_once()
        command = mock_run.call_args[0][0]
        assert command[0] == "/app/llama"
        assert "fit-params" in command
        assert "--fit-print" in command

    def test_adds_mmproj_gib_to_result(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        cache = FitParamsCache()

        fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="CUDA0 1000 0 0 \nHost 0 0 0\n")
        with patch("llama_swap_config_autogen.fit_params.subprocess.run", return_value=fake_result):
            result = estimate_vram_gib_via_fit_params(
                llama_bin=["/app/llama"],
                path_model=model,
                ngl=99,
                ctx=4096,
                cache_type_k=None,
                cache_type_v=None,
                extra_args=[],
                path_prefix_map={},
                cache=cache,
                extra_gpu_gib=0.5,
            )
        assert result == 1000 / 1024 + 0.5

    def test_uses_cache_on_second_call(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        cache = FitParamsCache()

        fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="CUDA0 1000 0 0 \nHost 0 0 0\n")
        with patch("llama_swap_config_autogen.fit_params.subprocess.run", return_value=fake_result) as mock_run:
            estimate_vram_gib_via_fit_params(
                llama_bin=["/app/llama"],
                path_model=model,
                ngl=99,
                ctx=4096,
                cache_type_k=None,
                cache_type_v=None,
                extra_args=[],
                path_prefix_map={},
                cache=cache,
            )
            estimate_vram_gib_via_fit_params(
                llama_bin=["/app/llama"],
                path_model=model,
                ngl=99,
                ctx=4096,
                cache_type_k=None,
                cache_type_v=None,
                extra_args=[],
                path_prefix_map={},
                cache=cache,
            )
        mock_run.assert_called_once()

    def test_builds_command_with_path_prefix_and_extra_args(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model = models_dir / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        cache = FitParamsCache()

        fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="CUDA0 1000 0 0 \nHost 0 0 0\n")
        with patch("llama_swap_config_autogen.fit_params.subprocess.run", return_value=fake_result) as mock_run:
            estimate_vram_gib_via_fit_params(
                llama_bin=["podman", "container", "exec", "llama-swap", "/app/llama"],
                path_model=model,
                ngl=99,
                ctx=4096,
                cache_type_k=None,
                cache_type_v=None,
                extra_args=["--n-cpu-moe", "12"],
                path_prefix_map={str(models_dir): "/models"},
                cache=cache,
            )
        command = mock_run.call_args[0][0]
        assert command[:5] == ["podman", "container", "exec", "llama-swap", "/app/llama"]
        assert "/models/model.gguf" in command
        assert "--n-cpu-moe" in command
        assert "12" in command
