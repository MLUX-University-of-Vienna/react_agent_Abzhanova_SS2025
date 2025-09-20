# tool_management/tools/vector_search_tool.py
import logging
from typing import List, Tuple, Union, Optional
from pathlib import Path
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document

from .agent_tool import AgentTool
from agent_core.models import ToolResponse, Observation

logger = logging.getLogger(__name__)


class VectorSearchTool(AgentTool):
    def __init__(self, file_paths: Union[str, Path, List[Union[str, Path]]]):
        self._files = self._resolve_paths(file_paths)
        self._documents = [self._extract_text_from_file(p) for p in self._files]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._doc_matrix = self._vectorizer.fit_transform(self._documents)
        logger.info(f"[VectorSearchTool.__init__] loaded_files={len(self._files)}")

    @property
    def name(self) -> str:
        return "vector_search"

    @property
    def description(self) -> str:
        return "Semantic TF-IDF search across loaded PDF, DOCX, TXT, or MD files."

    @property
    def refiner_instructions(self) -> str:
        return "Output a single concise query string. Preserve quoted phrases. Do not use operators. Do not compute."

    @property
    def refiner_input_format(self) -> str:
        return "<keywords and quoted phrases>"

    @property
    def refiner_input_regex(self) -> Optional[str]:
        return r"^[^\n\r]{1,300}$"

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

    def execute(self, query: str) -> ToolResponse:
        logger.info(f"[VectorSearchTool.execute] query={query!r}")
        try:
            query_vector = self._vectorize_query(query)
            similarity_scores = self._compute_similarity_scores(query_vector)
            top_matches = self._get_top_matches(similarity_scores)
            if not top_matches:
                logger.info("[VectorSearchTool.execute] no_matches")
                return ToolResponse(
                    success=True,
                    observations=[
                        Observation(
                            raw_text="No relevant documents found", source="vector_search", observed_at=datetime.utcnow()
                        )
                    ],
                    metadata={"tool": self.name},
                )
            observations: List[Observation] = []
            for rank, (i, doc_text, score) in enumerate(top_matches, start=1):
                path = self._files[i]
                snippet = (doc_text[:300] + "...") if len(doc_text) > 300 else doc_text
                observations.append(
                    Observation(
                        raw_text=f"Match {rank} (score: {score:.2f}):\n{snippet}",
                        title=path.name,
                        source="vector_search",
                        source_url=str(path),
                        observed_at=datetime.utcnow(),
                        meta={"score": score, "index": i},
                    )
                )
            logger.info(f"[VectorSearchTool.execute] observations_emitted={len(observations)}")
            return ToolResponse(success=True, observations=observations, metadata={"tool": self.name})
        except Exception as e:
            logger.error(f"[VectorSearchTool.execute] error={e!r}")
            return ToolResponse(
                success=False,
                observations=[
                    Observation(
                        raw_text=f"Vector search failed: {str(e)}", source="vector_search", observed_at=datetime.utcnow()
                    )
                ],
                metadata={"tool": self.name, "error": repr(e)},
            )

    def _resolve_paths(self, file_paths: Union[str, Path, List[Union[str, Path]]]) -> List[Path]:
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        return [Path(p) for p in file_paths]

    def _extract_text_from_file(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                return extract_pdf_text(str(file_path))
            if suffix == ".docx":
                doc = Document(str(file_path))
                return "\n".join(p.text for p in doc.paragraphs)
            if suffix in {".txt", ".md"}:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            raise ValueError(f"Unsupported file type: {suffix}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract content from {file_path.name}: {e}")

    def _vectorize_query(self, query: str):
        return self._vectorizer.transform([query])

    def _compute_similarity_scores(self, query_vector):
        return cosine_similarity(query_vector, self._doc_matrix).flatten()

    def _get_top_matches(self, similarity_scores, top_n: int = 3) -> List[Tuple[int, str, float]]:
        top_indices = similarity_scores.argsort()[::-1][:top_n]
        return [(i, self._documents[i], float(similarity_scores[i])) for i in top_indices if similarity_scores[i] > 0.0]
