import pytest

from weave.scorers.default_models import MODEL_PATHS
from weave.scorers.fluency_scorer import WeaveFluencyScorerV1


@pytest.fixture
def weave_fluency_scorer():
    """Fixture to return a WeaveFluencyScorerV1 instance."""
    scorer = WeaveFluencyScorerV1(
        model_name_or_path=MODEL_PATHS["fluency_scorer"],
        device="cpu",
    )
    return scorer


def test_score(weave_fluency_scorer):
    """Test score with a fluent response."""
    output = "This is a fluent response."
    result = weave_fluency_scorer.score(output=output)
    # Check that the pydantic model has the expected attributes.
    # With full models, a fluent response should pass
    assert result.passed
    assert result.metadata is not None
    assert result.metadata["score"] > 0.5
