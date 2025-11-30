import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.chunking import chunk_text_by_chars

def test_chunking_sanity():
    text = "A" * 3000
    
    chunk_size = 1000
    overlap = 150
    
    chunks = chunk_text_by_chars(text, chunk_size, overlap)
    
    print(f"Input text length: {len(text)}")
    print(f"Chunk size: {chunk_size}, Overlap: {overlap}")
    print(f"Number of chunks: {len(chunks)}")
    print()
    
    assert len(chunks) > 0, "Should generate chunks from non-empty text"
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: length={len(chunk)}")
        assert len(chunk) > 0, f"Chunk {i} should not be empty"
        assert len(chunk) <= chunk_size, f"Chunk {i} exceeds chunk_size"
    
    print()
    print("[OK] All chunks are non-empty")
    print("[OK] All chunks respect max size")
    
    total_covered = 0
    for i in range(len(chunks)):
        if i == 0:
            total_covered += len(chunks[i])
        else:
            total_covered += len(chunks[i]) - overlap
    
    print(f"[OK] Total text covered: {total_covered} chars (original: {len(text)})")
    assert total_covered >= len(text) - chunk_size, "Text coverage seems incorrect"
    
    print()
    print("=== Test with empty text ===")
    empty_chunks = chunk_text_by_chars("", chunk_size, overlap)
    assert len(empty_chunks) == 0, "Empty text should return empty list"
    print("[OK] Empty text returns empty list")
    
    print()
    print("=== Test with overlap >= chunk_size ===")
    bad_overlap_chunks = chunk_text_by_chars(text, chunk_size=100, overlap=150)
    assert len(bad_overlap_chunks) > 0, "Should handle overlap >= chunk_size"
    print(f"[OK] Handled overlap >= chunk_size, got {len(bad_overlap_chunks)} chunks")
    
    print()
    print("=== All sanity checks passed! ===")

if __name__ == "__main__":
    test_chunking_sanity()
