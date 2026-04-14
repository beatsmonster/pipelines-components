"""Tests for SDG LLM pipeline."""

from kfp import compiler

from ..pipeline import sdg_llm_pipeline


class TestSdgLlmPipeline:
    """Basic tests for SDG LLM pipeline."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline function is properly defined."""
        assert callable(sdg_llm_pipeline)

    def test_pipeline_compiles(self, tmp_path):
        """Test that the pipeline compiles successfully."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=sdg_llm_pipeline,
            package_path=str(output_path),
        )
        assert output_path.exists()
