from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

from docx import Document
from pdfminer.high_level import extract_text as extract_pdf_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from my_react_agent.agent_memory.data_structures import Evidence
from my_react_agent.tool_management.tools.agent_tool import AgentTool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Doc:
    path: Path
    text: str


class VectorSearchTool(AgentTool):

    VECTORIZER_STOP_WORDS = "english"

    TOP_N_MATCHES = 3
    SNIPPET_MAX_CHARS = 320
    REFINER_MAX_LEN = 300

    SUPPORTED_TEXT_SUFFIXES = {".txt", ".md"}
    SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}

    CHUNK_CHARS = 1200
    CHUNK_OVERLAP = 160
    MIN_CHUNK_CHARS = 250

    def __init__(self, file_paths: Union[str, Path, List[Union[str, Path]]]):
        self._files = self._resolve_paths(file_paths)
        self._docs = self._load_documents(self._files)
        self._chunks, self._chunk_map = self._build_chunks(self._docs)

        self._vectorizer = TfidfVectorizer(stop_words=self.VECTORIZER_STOP_WORDS)
        self._chunk_matrix = self._vectorizer.fit_transform(self._chunks)

        logger.info(
            "[VectorSearchTool.__init__] loaded_files=%d chunks=%d",
            len(self._files),
            len(self._chunks),
        )

    @property
    def name(self) -> str:
        return "vector_search"

    @property
    def description(self) -> str:
        return (
            "Local semantic search over the configured document (facts_about_Austria.pdf). "
        )

    @property
    def refiner_instructions(self) -> str:
        return "Output a single concise query string. Preserve quoted phrases. No operators. No computation."

    @property
    def refiner_input_format(self) -> str:
        return "<keywords and quoted phrases>"

    @property
    def refiner_input_regex(self) -> Optional[str]:
        return rf"^[^\n\r]{{1,{self.REFINER_MAX_LEN}}}$"

    @property
    def refiner_forbidden(self) -> str:
        return "multi_line, json, commands"

    @property
    def refiner_examples(self) -> str:
        return "\n".join(
            [
                "gene therapy off-target effects",
                '"transfer learning" domain adaptation healthcare',
                "Net Zero Scope 3 emissions accounting",
            ]
        )

    def execute(self, query: str) -> Evidence:
        q = (query or "").strip()
        logger.info("[VectorSearchTool.execute] query=%r", q)

        if not q:
            return Evidence(
                tool=self.name,
                content="Vector search failed: empty query",
                url=None,
                extracted={"error": True, "message": "empty_query"},
                as_of=datetime.utcnow(),
                confidence=0.1,
            )

        try:
            q_vec = self._vectorizer.transform([q])
            scores = cosine_similarity(q_vec, self._chunk_matrix).flatten()

            top = self._topk(scores, k=self.TOP_N_MATCHES)

            if not top:
                return Evidence(
                    tool=self.name,
                    content="No relevant passages found in loaded documents.",
                    url=None,
                    extracted={
                        "query": q,
                        "matches": {},
                        "files_loaded": len(self._files),
                        "chunks_indexed": len(self._chunks),
                    },
                    as_of=datetime.utcnow(),
                    confidence=0.7,
                )

            lines: List[str] = []
            matches: Dict[str, Dict[str, object]] = {}

            for rank, (chunk_idx, score) in enumerate(top, start=1):
                m = self._chunk_map[chunk_idx]
                path = Path(m["path"])
                snippet = self._build_snippet(self._chunks[chunk_idx])

                lines.append(f"Match {rank} (score: {score:.3f}) — {path.name}\n{snippet}")

                matches[str(rank)] = {
                    "rank": rank,
                    "score": float(score),
                    "file_path": str(path),
                    "file_name": path.name,
                    "chunk_index": int(chunk_idx),
                    "chunk_start": int(m["start"]),
                    "chunk_end": int(m["end"]),
                    "snippet": snippet,
                }

            content = "\n\n".join(lines)

            return Evidence(
                tool=self.name,
                content=content,
                url=None,
                extracted={
                    "query": q,
                    "matches": matches,
                    "files_loaded": len(self._files),
                    "chunks_indexed": len(self._chunks),
                },
                as_of=datetime.utcnow(),
                confidence=0.85,
            )

        except Exception as e:
            logger.exception("[VectorSearchTool.execute] error")
            return Evidence(
                tool=self.name,
                content=f"Vector search failed: {str(e)}",
                url=None,
                extracted={"error": True, "message": str(e), "exception": repr(e), "query": q},
                as_of=datetime.utcnow(),
                confidence=0.1,
            )


    def _resolve_paths(self, file_paths: Union[str, Path, List[Union[str, Path]]]) -> List[Path]:
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        paths = [Path(p).expanduser() for p in file_paths]

        out: List[Path] = []
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"VectorSearchTool file not found: {p}")
            if p.is_dir():
                for child in sorted(p.iterdir()):
                    if child.is_file() and child.suffix.lower() in self.SUPPORTED_SUFFIXES:
                        out.append(child)
            else:
                if p.suffix.lower() not in self.SUPPORTED_SUFFIXES:
                    raise ValueError(f"Unsupported file type: {p.suffix} for {p}")
                out.append(p)
        return out

    def _load_documents(self, files: List[Path]) -> List[_Doc]:
        docs: List[_Doc] = []
        for p in files:
            txt = self._extract_text_from_file(p)
            docs.append(_Doc(path=p, text=txt or ""))
        return docs

    def _extract_text_from_file(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                return extract_pdf_text(str(file_path)) or ""
            if suffix == ".docx":
                doc = Document(str(file_path))
                return "\n".join((p.text or "") for p in doc.paragraphs).strip()
            if suffix in self.SUPPORTED_TEXT_SUFFIXES:
                return file_path.read_text(encoding="utf-8", errors="ignore")
            raise ValueError(f"Unsupported file type: {suffix}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract content from {file_path.name}: {e}") from e

    def _build_chunks(self, docs: List[_Doc]) -> Tuple[List[str], List[dict]]:
        chunks: List[str] = []
        mapping: List[dict] = []

        for di, d in enumerate(docs):
            text = self._normalise_text(d.text)
            if not text:
                continue

            if len(text) <= self.CHUNK_CHARS:
                chunks.append(text)
                mapping.append({"doc_index": di, "path": str(d.path), "start": 0, "end": len(text)})
                continue

            start = 0
            n = len(text)
            while start < n:
                end = min(n, start + self.CHUNK_CHARS)
                chunk = text[start:end].strip()

                if len(chunk) >= self.MIN_CHUNK_CHARS:
                    chunks.append(chunk)
                    mapping.append({"doc_index": di, "path": str(d.path), "start": start, "end": end})

                if end >= n:
                    break
                start = max(0, end - self.CHUNK_OVERLAP)

        return chunks, mapping

    def _normalise_text(self, s: str) -> str:
        s = (s or "").replace("\x00", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _topk(self, scores, k: int) -> List[Tuple[int, float]]:
        if scores is None:
            return []
        idxs = scores.argsort()[::-1]
        out: List[Tuple[int, float]] = []
        for i in idxs:
            sc = float(scores[i])
            if sc <= 0.0:
                break
            out.append((int(i), sc))
            if len(out) >= k:
                break
        return out

    def _build_snippet(self, chunk_text: str) -> str:
        t = (chunk_text or "").strip()
        if len(t) > self.SNIPPET_MAX_CHARS:
            return t[: self.SNIPPET_MAX_CHARS].rstrip() + "..."
        return t
