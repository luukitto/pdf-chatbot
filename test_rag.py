from rag import chunk_text

text = "".join(str(i % 10) for i in range(2000))
chunks = chunk_text(text, size=800, overlap=150)

assert all(len(c) <= 800 for c in chunks)
assert all(a[-150:] == b[:150] for a, b in zip(chunks, chunks[1:]))  # overlap is real
assert chunks[-1].endswith(text[-150:])                              # nothing dropped at the end
assert chunk_text("short", 800, 150) == ["short"]
assert chunk_text("", 800, 150) == [""]
print("ok")
