import runpy
from pathlib import Path


def test_programmatic_retromae_example_imports_without_running() -> None:
    example = Path(__file__).parents[1] / "examples" / "retromae_then_sentence_transformers.py"
    namespace = runpy.run_path(str(example), run_name="pretense_example")
    assert callable(namespace["main"])
