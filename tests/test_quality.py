from sankat_saathi_dataset.quality import unsafe_hits, validate_structured_response
from sankat_saathi_dataset.templates import make_text_examples, make_vision_examples


def test_generated_text_response_is_structured():
    example = make_text_examples(1)[0]
    assert validate_structured_response(example.assistant_response.__dict__) == []


def test_generated_vision_has_uncertainty():
    example = make_vision_examples(1)[0]
    assert example.image_path
    assert example.image_uncertainty
    assert "cannot prove safety" in example.assistant_response.uncertainty_note.lower()


def test_unsafe_phrase_detector_catches_overconfidence():
    assert unsafe_hits("This is definitely safe and guaranteed.")
