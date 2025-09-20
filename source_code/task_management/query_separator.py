import json
import logging
import re
from typing import List, Optional

from llm_adapters.llm_base import LLMBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LLMQuerySeparator:
    action_verb_regular_expression = re.compile(
        r"\b(calculate|compute|convert|compare|list|give|define|explain|summarize|summarise|"
        r"find|get|fetch|lookup|translate|draw|plot|generate|create|show|identify|"
        r"analyze|analyse|estimate|project|rank|forecast|aggregate|determine)\b",
        re.IGNORECASE,
    )

    strong_separator_regular_expressions = [
        re.compile(r"\bthen\b", re.IGNORECASE),
        re.compile(r"\bvs\b|\bversus\b", re.IGNORECASE),
        re.compile(r"\bor\b", re.IGNORECASE),
        re.compile(r";"),
    ]

    def __init__(
        self,
        llm: LLMBase,
        *,
        max_parts: int = 8,
        max_prompt_examples: int = 6,
        fail_open: bool = True,
    ):
        """
        Parameters:
            llm: An instance of LLMBase used to generate subquery lists.
            max_parts: Maximum number of subqueries to return.
            max_prompt_examples: Number of in-prompt examples to include for the LLM.
            fail_open: If True, fall back to heuristic splitting on LLM errors; otherwise return the original query.
        """
        self.llm = llm
        self.max_parts = max(2, int(max_parts))
        self.max_prompt_examples = int(max_prompt_examples)
        self.fail_open = bool(fail_open)
        logger.info(
            f"[LLMQuerySeparator.__init__] Initialized with max_parts={self.max_parts}, "
            f"max_prompt_examples={self.max_prompt_examples}, fail_open={self.fail_open}"
        )

    # ---- Public API -------------------------------------------------------

    def separate(self, query: str) -> List[str]:
        """
        Separate a user query into atomic subqueries using an LLM-first approach
        with robust fallbacks and sanitization.
        """
        logger.info("[LLMQuerySeparator.separate] Start")
        original_query = (query or "").strip()
        if not original_query:
            logger.warning("[LLMQuerySeparator.separate] Empty query provided; returning [''].")
            return [""]

        try:
            logger.debug("[LLMQuerySeparator.separate] Requesting parts from LLM.")
            llm_parts_raw = self._ask_llm_for_parts(original_query)
            logger.debug(f"[LLMQuerySeparator.separate] LLM returned {len(llm_parts_raw)} raw parts.")
        except Exception as exception:
            logger.exception(f"[LLMQuerySeparator.separate] LLM call failed: {exception!r}")
            if self.fail_open:
                logger.info("[LLMQuerySeparator.separate] Falling back to heuristic split due to LLM error.")
                heuristic = self._heuristic_split(original_query)
                processed_heuristic = self._postprocess_parts(heuristic, original_query)
                self._log_split_decision(original_query, processed_heuristic, fallback_used=True)
                return processed_heuristic
            else:
                logger.info("[LLMQuerySeparator.separate] Returning original query due to LLM error and fail_open=False.")
                return [original_query]

        processed_parts = self._postprocess_parts(llm_parts_raw, original_query)
        logger.debug(f"[LLMQuerySeparator.separate] Post-processed parts count: {len(processed_parts)}.")

        if self._looks_effectively_single_intent(processed_parts, original_query):
            logger.info("[LLMQuerySeparator.separate] LLM suggests single intent or low-quality split; considering heuristic fallback.")
            heuristic_parts = self._heuristic_split(original_query)
            heuristic_parts = self._postprocess_parts(heuristic_parts, original_query)

            # Choose the better split: prefer LLM result if it yields multiple meaningful parts,
            # otherwise use heuristic if it results in more than one meaningful part.
            if len(processed_parts) > 1 and not self._looks_effectively_single_intent(processed_parts, original_query):
                chosen = processed_parts[: self.max_parts]
                self._log_split_decision(original_query, chosen, fallback_used=False)
                return chosen
            if len(heuristic_parts) > 1 and not self._looks_effectively_single_intent(heuristic_parts, original_query):
                chosen = heuristic_parts[: self.max_parts]
                self._log_split_decision(original_query, chosen, fallback_used=True)
                return chosen

            logger.info("[LLMQuerySeparator.separate] No convincing split; returning original query as a single part.")
            return [original_query]

        chosen = processed_parts[: self.max_parts]
        self._log_split_decision(original_query, chosen, fallback_used=False)
        return chosen

    # ---- LLM interaction --------------------------------------------------

    def _ask_llm_for_parts(self, query: str) -> List[str]:
        """
        Ask the LLM to produce a STRICT JSON array of atomic subqueries.
        Returns a list of strings; may be empty if parsing fails.
        """
        examples = [
            (
                "how many people live in Tokyo and Osaka combined",
                [
                    "latest population of Tokyo metropolitan area",
                    "latest population of Osaka metropolitan area",
                    "sum: a+b",
                ],
            ),
            (
                "is Apple revenue higher than Microsoft last year",
                [
                    "Apple total revenue for last fiscal year",
                    "Microsoft total revenue for last fiscal year",
                    "compare: a vs b",
                ],
            ),
            (
                "market cap of Tesla today converted to EUR",
                [
                    "current market capitalization of Tesla in USD",
                    "latest USD to EUR exchange rate",
                    "convert: a*b rounded to 0 decimals using calculator",
                ],
            ),
            (
                "define photosynthesis and list its primary reactants",
                [
                    "concise definition of photosynthesis",
                    "primary reactants of photosynthesis (bullet list)",
                ],
            ),
            (
                "who won the latest FIFA World Cup and how many goals they scored in the final",
                [
                    "winner of the most recent FIFA World Cup",
                    "number of goals scored by the winner in the final match",
                ],
            ),
            (
                "population of Canada and USA; compute Canada as % of USA",
                [
                    "latest population of Canada",
                    "latest total population of the United States",
                    "compute percentage: (a/b)*100 rounded to 2 decimals using calculator",
                ],
            ),
        ][: self.max_prompt_examples]

        examples_block = "\n".join(
            f'- {q} -> {json.dumps(a, ensure_ascii=False)}' for q, a in examples
        )

        prompt = (
            "Produce an ordered list of atomic subqueries that each resolve EXACTLY one fact or step.\n"
            f"Return STRICT JSON array of strings (length 1..{self.max_parts}).\n"
            "Rules:\n"
            "- Keep original order; minimize rephrasing unless needed for clarity.\n"
            "- Do NOT include placeholders like [x]/{x}/<x> or '...'. Use concrete, tool-ready phrasing.\n"
            "- Each item must be independently answerable and not require user-specific preferences.\n"
            "- If the query is single-intent, return [original query].\n\n"
            "Examples:\n"
            f"{examples_block}\n\n"
            f"User query: {query}\n"
            "JSON:"
        )

        logger.debug(
            "[LLMQuerySeparator._ask_llm_for_parts] Sending prompt to LLM "
            f"(length={len(prompt)} characters)."
        )
        raw_output = (self.llm.generate(prompt) or "").strip()
        logger.debug(
            "[LLMQuerySeparator._ask_llm_for_parts] Received raw LLM output "
            f"(length={len(raw_output)} characters)."
        )
        parsed_array = self._safe_parse_json_array(raw_output)
        if not isinstance(parsed_array, list):
            logger.warning("[LLMQuerySeparator._ask_llm_for_parts] Parsed output is not a list; returning empty list.")
            return []
        cleaned_array = [self._clean_string(s) for s in parsed_array if isinstance(s, str)]
        logger.info(
            f"[LLMQuerySeparator._ask_llm_for_parts] Parsed {len(cleaned_array)} items from LLM."
        )
        return cleaned_array

    # ---- Heuristic fallback and post-processing ---------------------------

    def _heuristic_split(self, text: str) -> List[str]:
        """
        Split using deterministic heuristics:
          - strong separators: 'then', 'vs/versus', 'or', ';'
          - cautious 'and' splitting guarded by the presence of action verbs
        """
        logger.info("[LLMQuerySeparator._heuristic_split] Running heuristic splitter.")
        text = self._clean_string(text)
        if not text:
            logger.warning("[LLMQuerySeparator._heuristic_split] Empty text after cleaning; returning [''].")
            return [""]

        segments_queue: List[str] = [text]

        # Apply strong separators first
        for separator_expression in self.strengthened_separators_for_logging():
            regular_expression = separator_expression["regex"]
            label = separator_expression["label"]
            next_segments: List[str] = []
            for segment in segments_queue:
                split_candidates = [self._clean_string(p) for p in regular_expression.split(segment)]
                split_candidates = [p for p in split_candidates if p]
                if len(split_candidates) > 1:
                    logger.debug(
                        f"[LLMQuerySeparator._heuristic_split] Split by '{label}' into {len(split_candidates)} parts."
                    )
                next_segments.extend(split_candidates or [segment])
            segments_queue = next_segments or segments_queue

        # Cautious 'and' splitting with verb guard
        output_segments: List[str] = []
        for segment in segments_queue:
            tokenized = re.split(r"\band\b", segment, flags=re.IGNORECASE)
            if len(tokenized) == 1:
                output_segments.append(segment)
                continue

            current_buffer = tokenized[0]
            for next_piece in tokenized[1:]:
                if self._string_has_action_verb(current_buffer) and self._string_has_action_verb(next_piece):
                    output_segments.append(self._clean_string(current_buffer))
                    current_buffer = next_piece
                else:
                    current_buffer = (current_buffer + " and " + next_piece).strip()
            output_segments.append(self._clean_string(current_buffer))

        output_segments = [p for p in output_segments if p]
        output_segments = self._deduplicate_preserving_order(output_segments)
        output_segments = self._merge_tiny_neighbor_segments(output_segments)
        capped = output_segments[: self.max_parts]

        logger.info(
            f"[LLMQuerySeparator._heuristic_split] Heuristic produced {len(capped)} parts (before cap: {len(output_segments)})."
        )
        return capped or [text]

    def strengthened_separators_for_logging(self) -> List[dict]:
        """
        Helper to associate human-readable labels with strong separators for better debugging.
        """
        return [
            {"label": "then", "regex": self.strong_separator_regular_expressions[0]},
            {"label": "vs/versus", "regex": self.strong_separator_regular_expressions[1]},
            {"label": "or", "regex": self.strong_separator_regular_expressions[2]},
            {"label": "semicolon", "regex": self.strong_separator_regular_expressions[3]},
        ]

    def _postprocess_parts(self, parts: List[str], fallback: str) -> List[str]:
        """
        Sanitize, drop placeholders, de-duplicate, merge tiny fragments, and cap.
        """
        if not parts:
            logger.debug("[LLMQuerySeparator._postprocess_parts] No parts to postprocess; using fallback.")
            return [fallback]

        cleaned_parts: List[str] = []
        for part in parts:
            sanitized = self._strip_placeholders(self._clean_string(part))
            if sanitized:
                cleaned_parts.append(sanitized)

        deduplicated = self._deduplicate_preserving_order(cleaned_parts)
        merged = self._merge_tiny_neighbor_segments(deduplicated)
        capped = merged[: self.max_parts]

        logger.info(
            f"[LLMQuerySeparator._postprocess_parts] Post-processing reduced parts "
            f"from {len(parts)} to {len(capped)}."
        )
        return capped or [fallback]

    # ---- Decision helpers -------------------------------------------------

    def _looks_effectively_single_intent(self, parts: List[str], original_query: str) -> bool:
        """
        Return True when the list either has one item or multiple items that are effectively the same,
        indicating single-intent or low-quality splitting.
        """
        if not parts:
            return True
        if len(parts) == 1:
            return True

        normalized_first = self._normalize_for_similarity(parts[0])
        all_same = all(self._normalize_for_similarity(p) == normalized_first for p in parts)
        if all_same:
            logger.debug("[LLMQuerySeparator._looks_effectively_single_intent] All parts normalize to the same text.")
            return True

        # If multiple parts exist but none contains an action verb while the original does, treat as low-quality split.
        original_has_verb = self._string_has_action_verb(original_query)
        parts_have_verbs = any(self._string_has_action_verb(p) for p in parts)
        if original_has_verb and not parts_have_verbs:
            logger.debug(
                "[LLMQuerySeparator._looks_effectively_single_intent] Parts lack action verbs compared to original."
            )
            return True

        return False

    def _log_split_decision(self, original_query: str, chosen_parts: List[str], fallback_used: bool) -> None:
        """
        Log a concise explanation of the split decision for observability.
        """
        decision_source = "heuristic" if fallback_used else "LLM"
        logger.info(
            f"[LLMQuerySeparator._log_split_decision] Source={decision_source}; "
            f"returned {len(chosen_parts)} part(s)."
        )
        logger.debug(
            "[LLMQuerySeparator._log_split_decision] Original query: %r; Parts: %s",
            original_query,
            json.dumps(chosen_parts, ensure_ascii=False),
        )

    # ---- Tiny utilities ---------------------------------------------------

    @staticmethod
    def _clean_string(text: Optional[str]) -> str:
        if not text:
            return ""
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text).strip(" ,;:-")
        return text.strip()

    @staticmethod
    def _strip_placeholders(text: str) -> str:
        if not text:
            return text
        text = re.sub(r"\[[^\]]+\]", " ", text)
        text = re.sub(r"\{[^}]+\}", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s*(\.\.\.|…)\s*", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _string_has_action_verb(self, text: str) -> bool:
        return bool(self.action_verb_regular_expression.search(text or ""))

    @staticmethod
    def _normalize_for_similarity(text: str) -> str:
        return re.sub(r"\W+", " ", (text or "").lower()).strip()

    @staticmethod
    def _deduplicate_preserving_order(parts: List[str]) -> List[str]:
        seen_lowercase = set()
        output_parts = []
        for part in parts:
            key = part.lower()
            if key not in seen_lowercase:
                seen_lowercase.add(key)
                output_parts.append(part)
        return output_parts

    @staticmethod
    def _merge_tiny_neighbor_segments(parts: List[str]) -> List[str]:
        """
        Merge very short neighboring segments to avoid fragmenting.
        """
        if not parts:
            return parts
        merged_parts: List[str] = []
        buffer_text = ""
        for part in parts:
            if len(part) < 4:
                buffer_text = (buffer_text + " " + part).strip()
            else:
                if buffer_text:
                    merged_parts.append((buffer_text + " " + part).strip())
                    buffer_text = ""
                else:
                    merged_parts.append(part)
        if buffer_text:
            if merged_parts:
                merged_parts[-1] = (merged_parts[-1] + " " + buffer_text).strip()
            else:
                merged_parts.append(buffer_text)
        return merged_parts

    @staticmethod
    def _safe_parse_json_array(raw_text: str) -> List:
        """
        Accept either a raw JSON array or text containing one.
        Returns [] on failure.
        """
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                return parsed
            logger.warning("[LLMQuerySeparator._safe_parse_json_array] Top-level JSON is not a list.")
            return []
        except Exception:
            # Attempt to extract the last JSON array from within text
            start_index = raw_text.rfind("[")
            end_index = raw_text.rfind("]")
            if start_index != -1 and end_index != -1 and end_index > start_index:
                fragment = raw_text[start_index : end_index + 1]
                try:
                    parsed = json.loads(fragment)
                    if isinstance(parsed, list):
                        return parsed
                    logger.warning(
                        "[LLMQuerySeparator._safe_parse_json_array] Extracted fragment is not a list."
                    )
                    return []
                except Exception:
                    logger.warning(
                        "[LLMQuerySeparator._safe_parse_json_array] Failed to parse extracted JSON fragment."
                    )
                    return []
            logger.warning("[LLMQuerySeparator._safe_parse_json_array] No JSON array found in text.")
            return []
