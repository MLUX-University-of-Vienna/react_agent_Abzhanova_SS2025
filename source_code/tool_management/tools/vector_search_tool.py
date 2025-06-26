import logging
from typing import List, Tuple, Union
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document

from .agent_tool import AgentTool, ToolResponse


class VectorSearchTool(AgentTool):
    """
    A semantic vector search tool that loads and indexes multiple documents (PDF, DOCX, TXT, or MD),
    vectorizes them using TF-IDF, and performs cosine similarity-based search.
    """

    def __init__(self, file_paths: Union[str, Path, List[Union[str, Path]]]):
        self._files = self._resolve_paths(file_paths)
        self._documents = [self._extract_text_from_file(p) for p in self._files]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._doc_matrix = self._vectorizer.fit_transform(self._documents)

    @property
    def name(self) -> str:
        return "vector_search"

    @property
    def description(self) -> str:
        return "Extracts and semantically searches PDF, DOCX, TXT, or MD files using TF-IDF vector similarity."

    def execute(self, input: str) -> ToolResponse:
        try:
            query_vector = self._vectorize_query(input)
            similarity_scores = self._compute_similarity_scores(query_vector)
            top_matches = self._get_top_matches(similarity_scores)

            if not top_matches:
                return ToolResponse(content="No relevant documents found", success=True)

            content = self._format_response(top_matches)
            return ToolResponse(content=content, success=True, metadata={"matches": top_matches})

        except Exception as e:
            logging.error(f"Vector search failed: {e}")
            return ToolResponse(content=f"Vector search failed: {str(e)}", success=False)

    def _resolve_paths(self, file_paths: Union[str, Path, List[Union[str, Path]]]) -> List[Path]:
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        return [Path(p) for p in file_paths]

    def _extract_text_from_file(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()

        try:
            match suffix:
                case ".pdf":
                    return extract_pdf_text(str(file_path))

                case ".docx":
                    doc = Document(str(file_path))
                    return "\n".join(p.text for p in doc.paragraphs)

                case ".txt" | ".md":
                    with open(file_path, "r", encoding="utf-8") as f:
                        return f.read()

                case _:
                    raise ValueError(f"Unsupported file type: {suffix}")

        except Exception as e:
            raise RuntimeError(f"Failed to extract content from {file_path.name}: {e}")

    def _vectorize_query(self, query: str):
        return self._vectorizer.transform([query])

    def _compute_similarity_scores(self, query_vector):
        return cosine_similarity(query_vector, self._doc_matrix).flatten()

    def _get_top_matches(self, similarity_scores, top_n: int = 3) -> List[Tuple[int, str, float]]:
        top_indices = similarity_scores.argsort()[::-1][:top_n]
        return [
            (i, self._documents[i], float(similarity_scores[i]))
            for i in top_indices if similarity_scores[i] > 0.0
        ]

    def _format_response(self, matches: List[Tuple[int, str, float]]) -> str:
        return "\n\n".join(
            f"Match {rank+1} (score: {score:.2f}):\n{doc[:300]}..."
            for rank, (i, doc, score) in enumerate(matches)
        )

    @classmethod
    def get_patterns(cls) -> List[str]:
        return [
            r"\bsearch\b",
            r"\bfind\b",
            r"\bdocument\b",
            r"\btopic\b",
            r"\bvector search\b"
        ]

    @classmethod
    def get_examples(cls) -> str:
        return (
            "User: Find documents about ReAct agents\n"
            "{\n"
            '  "thought": "Semantic search is needed",\n'
            '  "action": "vector_search",\n'
            '  "action_input": "ReAct agents and reasoning",\n'
            '  "final_answer": ""\n'
            "}"
        )
