from pathlib import Path

from rag.indexing import file_digest


def test_file_digest_is_stable_and_content_sensitive(tmp_path: Path):
    manual = tmp_path / "manual.pdf"
    manual.write_bytes(b"first version")
    first_digest = file_digest(manual)
    assert first_digest == file_digest(manual)

    manual.write_bytes(b"second version")
    assert file_digest(manual) != first_digest
