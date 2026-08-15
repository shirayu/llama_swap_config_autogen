"""Tests for matches_model_pattern glob/substring matching"""

from llama_swap_config_autogen.generator import matches_model_pattern


class TestMatchesModelPatternSubstring:
    """Patterns without glob metacharacters keep plain substring matching."""

    def test_plain_substring_match(self):
        assert matches_model_pattern("qwen3", "qwen3-30b-instruct:Q4_K_M")

    def test_plain_substring_no_match(self):
        assert not matches_model_pattern("qwen3", "gemma-4-31b:Q4_K_M")

    def test_case_insensitive(self):
        assert matches_model_pattern("QWEN3", "qwen3-30b-instruct:Q4_K_M")

    def test_matches_any_identifier(self):
        assert matches_model_pattern("gemma-4-12b", "gemma-4-12b:Q4_K_XL", "some-file.gguf", "Gemma 4 12B")

    def test_list_of_plain_patterns(self):
        assert matches_model_pattern(["gemma-3", "qwen3"], "qwen3-30b-instruct:Q4_K_M")
        assert not matches_model_pattern(["gemma-3", "gemma-4"], "qwen3-30b-instruct:Q4_K_M")


class TestMatchesModelPatternGlob:
    """Patterns containing *, ?, or [ are matched as globs (fnmatch) against the whole identifier."""

    def test_star_wildcard_matches(self):
        assert matches_model_pattern("*qwen3*-vl-*", "some-id", "qwen3-30b-vl-instruct:Q4_K_M")

    def test_star_wildcard_no_match(self):
        assert not matches_model_pattern("*qwen3*-vl-*", "qwen3-30b-instruct:Q4_K_M")

    def test_star_wildcard_matches_suffix_boundary(self):
        assert matches_model_pattern("*qwen3*-vl*", "qwen3-30b-vl:Q4_K_M")

    def test_glob_requires_full_match_not_substring(self):
        # Without leading/trailing '*' fnmatch requires an exact match, so a
        # bare glob pattern behaves differently from the old substring style.
        assert not matches_model_pattern("qwen3*", "prefix-qwen3-30b:Q4_K_M")
        assert matches_model_pattern("qwen3*", "qwen3-30b:Q4_K_M")

    def test_question_mark_wildcard(self):
        assert matches_model_pattern("gemma-4-3?b", "gemma-4-31b")
        assert not matches_model_pattern("gemma-4-3?b", "gemma-4-312b")

    def test_bracket_wildcard(self):
        assert matches_model_pattern("gemma-4-[0-9][0-9]b", "gemma-4-31b")
        assert not matches_model_pattern("gemma-4-[0-9][0-9]b", "gemma-4-3b")

    def test_glob_is_case_insensitive(self):
        assert matches_model_pattern("*QWEN3*", "qwen3-30b:Q4_K_M")

    def test_list_of_glob_patterns(self):
        assert matches_model_pattern(["*gemma-3*", "*qwen3*"], "qwen3-30b:Q4_K_M")
        assert not matches_model_pattern(["*gemma-3*", "*gemma-4*"], "qwen3-30b:Q4_K_M")

    def test_mixed_plain_and_glob_in_list(self):
        assert matches_model_pattern(["gemma-3", "*qwen3*-vl*"], "qwen3-30b-vl:Q4_K_M")
        assert matches_model_pattern(["qwen3", "*gemma-4*-vl*"], "qwen3-30b:Q4_K_M")
