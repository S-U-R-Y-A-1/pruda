# chunking.py
import re
from typing import List, Tuple, Dict
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False

_SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')

def split_into_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sentences = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]

def tokens_length(text: str, encoder_name: str = "gpt2") -> int:
    if TIKTOKEN_AVAILABLE:
        enc = tiktoken.get_encoding(encoder_name)
        return len(enc.encode(text))
    # fallback heuristic
    words = text.split()
    return max(1, int(len(words) * 1.3))

def chunk_sentences_to_chunks(
    sentences: List[str],
    max_tokens: int = 800,
    overlap_tokens: int = 150,
    encoder_name: str = "gpt2"
) -> List[str]:
    if not sentences:
        return []
    chunks = []
    cur = []
    cur_tokens = 0

    def flush():
        nonlocal cur, cur_tokens
        if cur:
            chunks.append(" ".join(cur).strip())
        cur = []
        cur_tokens = 0

    for sent in sentences:
        sent_tokens = tokens_length(sent, encoder_name)
        if sent_tokens > max_tokens:
            # split long sentence by words into smaller pieces
            words = sent.split()
            piece = []
            piece_tokens = 0
            for w in words:
                w_tokens = tokens_length(w + " ", encoder_name)
                if piece_tokens + w_tokens > max_tokens:
                    chunks.append(" ".join(piece).strip())
                    piece = [w]
                    piece_tokens = w_tokens
                else:
                    piece.append(w)
                    piece_tokens += w_tokens
            if piece:
                if cur_tokens + piece_tokens <= max_tokens:
                    cur.append(" ".join(piece))
                    cur_tokens += piece_tokens
                else:
                    flush()
                    cur.append(" ".join(piece))
                    cur_tokens = piece_tokens
            continue

        if cur_tokens + sent_tokens <= max_tokens:
            cur.append(sent)
            cur_tokens += sent_tokens
        else:
            flush()
            cur.append(sent)
            cur_tokens = sent_tokens

    if cur:
        chunks.append(" ".join(cur).strip())

    # Create overlap (approx): prepend last N words from previous chunk to current
    if overlap_tokens > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = overlapped[-1]
            prev_words = prev.split()
            approx_words = max(1, int(overlap_tokens / 1.3))
            overlap_words = prev_words[-approx_words:] if len(prev_words) >= approx_words else prev_words
            new_chunk = " ".join(overlap_words + chunks[i].split())
            overlapped.append(new_chunk)
        chunks = overlapped

    return chunks

def chunk_page_text(page_text: str, source: str, page_no: int,
                    max_tokens: int = 800, overlap_tokens: int = 150,
                    encoder_name: str = "gpt2") -> List[Dict]:
    """
    Returns list of dicts:
    { 'source': source, 'page': page_no, 'chunk_index': i, 'text': chunk_text, 'label': label }
    """
    sentences = split_into_sentences(page_text)
    chunks = chunk_sentences_to_chunks(sentences, max_tokens, overlap_tokens, encoder_name)
    results = []
    for i, c in enumerate(chunks, start=1):
        label = f"(source:{source} page:{page_no} chunk:{i})\n{c}"
        results.append({
            "source": source,
            "page": page_no,
            "chunk_index": i,
            "text": c,
            "label": label
        })
    return results
