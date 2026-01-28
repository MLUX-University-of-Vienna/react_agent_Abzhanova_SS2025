from __future__ import annotations

from .prompts_ids import PromptId
from .prompt_template import PromptTemplate

DEFAULT_PROMPTS: dict[PromptId, PromptTemplate] = {

    PromptId.SHOULD_SEPARATE_QUESTION: PromptTemplate(
        text=(
            "You are deciding whether a user question should be decomposed into multiple sequential steps, "
            "or handled as a single step.\n\n"
            "Guidelines:\n"
            "- Answer true in 'separate' if the question clearly involves:\n"
            "  * multiple entities to compare\n"
            "  * multiple calculations or time periods\n"
            "  * dependent sub-questions (where one answer is needed for another)\n"
            "  * complex reasoning that benefits from ordered steps\n"
            "- Answer false if the question is a single direct lookup, a simple factual question, or a brief conceptual explanation.\n"
            "- Be conservative: prefer false if unsure.\n\n"
            "User question:\n{question}\n\n"
            "Return STRICT JSON with this shape:\n{schema}\n\n"
            "JSON only, no explanation:"
        ),
        required_vars={"question", "schema"},
        description="Decide if the question should be split into multiple steps",
        version="1",
    ),
    PromptId.PLAN_QUESTION_STEPS: PromptTemplate(
        text=(
            "Your task is to break this question into a small, ordered list of concrete steps that, if executed in sequence, "
            "will provide all facts and/or actions needed to answer the question.\n"
            "Think like solving a problem in the minimal number of steps.\n\n"
            "Rules:\n"
            "- Steps must be in executable order (1,2,3,...).\n"
            "- Each step must be small and focused on one micro-task.\n"
            "- Use the same words from the question (no synonyms, no re-interpretation of definitions).\n"
            "- If a step needs results of previous steps and they contain the same entity, list those indices in 'depends_on'; "
            "  otherwise use an empty list.\n"
            "- Do not add a step if it resolves the same fact as an earlier step.\n"
            "- Only describe WHAT to do, not HOW, and not WHICH tool to use.\n"
            "- Do not add steps to find information already given.\n"
            "- Use at most 6 steps unless the question clearly needs more.\n\n"
            "User question:\n{question}\n\n"
            "Available tools (for later execution):\n{tool_catalog}\n\n"
            "Return STRICT JSON with the following shape:\n{schema}\n\n"
            "JSON only, no explanation:"
        ),
        required_vars={"question", "tool_catalog", "schema"},
        description="Create StepPlan list",
        version="1",
    ),
    PromptId.DECIDE_ACTION_FOR_STEP: PromptTemplate(
        text=(
            "You are deciding how to execute the task of ONE planned step inside a larger plan.\n\n"
            "Available actions (choose exactly ONE), read precisely:\n"
            "{actions_catalog}\n\n"
            "Tools you can call (read tool descriptions carefully):\n"
            "{tool_catalog}\n\n"
            "User question:\n{question}\n\n"
            "Current step:\n"
            "- task to execute: {step_task}\n"
            "- needed_facts: {needed_facts}\n\n"
            "Previously executed steps:\n{prev_steps_block}\n\n"
            "Return STRICT JSON matching this shape:\n{schema}\n"
            "JSON:"
        ),
        required_vars={
            "question",
            "step_task",
            "needed_facts",
            "prev_steps_block",
            "tool_catalog",
            "actions_catalog",
            "schema",
        },
        description="Choose next action/tool for the current step",
        version="1",
    ),
    PromptId.LOW_CONF_TOOL_RESULT: PromptTemplate(
        text=(
            "A previously executed step produced a result that looks unreliable or not well-aligned "
            "with the current step task.\n\n"
            "You must now choose the BEST next action from this action catalogue:\n"
            "{actions_catalog}\n\n"
            "Available tools:\n"
            "{tool_catalog}\n\n"
            "General rules:\n"
            "1) Obey each tool's description and limits exactly. Do NOT imagine extra abilities. \n"
            "2) Avoid repeating the exact same (tool_name, tool_input) pair that already failed. \n"
            "Step task:\n{step_task}\n\n"
            "Previously used tool:\n"
            "- name: {prev_tool_name}\n"
            "- input: {prev_tool_input}\n\n"
            "Step result (possibly unreliable):\n"
            "{raw_text}\n\n"
            "Return STRICT JSON ONLY with this schema:\n"
            "{schema}\n"
            "JSON:"
        ),
        required_vars={
            "question",
            "step_task",
            "prev_tool_name",
            "prev_tool_input",
            "raw_text",
            "tool_catalog",
            "actions_catalog",
            "schema",
        },
        description="Pick best recovery action when a tool result is low-confidence",
        version="1",
    ),

    PromptId.SUMMARISE_OBSERVATION: PromptTemplate(
        text=(
            "You are summarising a tool result for an internal execution trace.\n"
            "Write 1-5 concise facts that will help answer the question.\n\n"
            "CRITICALLY IMPORTANT:\n"
            "- If the user question involves any quantities you MUST explicitly include the exact numeric values and units "
            "  from the tool result that help answer the question.\n"
            "- Do NOT omit such numbers when present.\n"
            "- If no relevant numbers are present, say so explicitly.\n\n"
            "Question:\n{question}\n\n"
            "Tool result:\n{tool_result}\n\n"
            "Summary:"
        ),
        required_vars={"question", "tool_result"},
        description="Summarize a tool observation into a short fact list",
        version="1",
    ),
    PromptId.UPDATE_FACT_STORE: PromptTemplate(
        text=(
            "Extract up to N compact facts from the text that help resolve the FOCUS.\n"
            "Return JSON only.\n\n"
            "N={max_items}\n"
            "FOCUS: {focus}\n"
            "Original question: {question}\n\n"
            "Text:\n{text}\n\n"
            "Schema:\n{schema}\n"
            "JSON:"
        ),
        required_vars={"max_items", "focus", "question", "text", "schema"},
        description="Extract structured facts (optionally numeric) from text into a fact store schema",
        version="1",
    ),
    PromptId.REWRITE_STEP_TASK_WITH_CONTEXT: PromptTemplate(
        text=(
            "You rewrite the CURRENT STEP TASK using only the provided memory-like inputs.\n\n"
            "Goal:\n"
            "- If the task contains pronouns or omitted entities, resolve them using:\n"
            "  1) previous_steps_extracted_facts (newer first)\n"
            "  2) recent_queries (most recent first)\n"
            "- Rewrite the task to be explicit and executable.\n"
            "- Do NOT invent entities. If you cannot resolve ambiguity, set found=false and return rewritten_task unchanged.\n"
            "- Keep the task the same except for resolving pronouns/omissions; do not add sub-tasks.\n\n"
            "CURRENT STEP TASK:\n{step_task}\n\n"
            "previous_steps_extracted_facts:\n{facts_block}\n\n"
            "recent_queries (most recent first):\n{queries_block}\n\n"
            "Return STRICT JSON with this shape:\n{schema}\n\n"
            "JSON only:"
        ),
        required_vars={"step_task", "facts_block", "queries_block", "schema"},
        description="Resolve pronouns/implicit references in a step task using prior facts and recent Q/A",
        version="1",
    ),
    PromptId.SYNTHESISE_FINAL_ANSWER: PromptTemplate(
        text=(
            "You are producing the final answer to the user's question.\n"
            "You are given a sequence of executed steps (with observations/evidence snippets) and additional context snippets.\n\n"
            "Task:\n"
            "- Combine the observations/evidence into a direct final answer.\n"
            "- Aim for 1–3 sentences unless the question clearly needs more detail.\n"
            "- You may ONLY use numbers or concrete facts that appear in the observations/evidence/context; do NOT invent numbers.\n"
            "- Do NOT change existing numbers. Do NOT round numbers.\n"
            "- If information is insufficient for a numeric part, say so explicitly instead of guessing.\n\n"
            "Question:\n{question}\n\n"
            "Executed steps, observations, and evidence:\n{executed_steps_block}\n\n"
            "Additional context snippets:\n{context_block}\n\n"
            "Final answer:"
        ),
        required_vars={"question", "executed_steps_block", "context_block"},
        description="Synthesize final answer from executed trace and context snippets",
        version="1",
    ),
    PromptId.ANSWER_BY_ITSELF: PromptTemplate(
        text=(
            "You are answering the CURRENT STEP without using any external tools.\n\n"
            "CURRENT STEP TASK:\n"
            "{step_task}\n\n"
            "DEPENDENCY ANSWERS (only if this step depends on them):\n"
            "{deps_block}\n\n"
            "Rules:\n"
            "- Use dependency answers as context if relevant.\n"
            "- If the task asks for real-time / current conditions (e.g., weather 'right now', prices today, live info), "
            " you MUST say you cannot know it without a tool and suggest what tool/data source would be needed.\n"
            "- Do NOT invent numbers, times, or current conditions.\n"
            "- Be concise: 1–3 sentences.\n\n"
            "Answer:"       
        ),
        required_vars={"step_task", "deps_block"},
        description="LLM answers the step's task according to its knowledge",
        version="1",
    ),
    PromptId.LLM_EXTRACT_NAMES: PromptTemplate(
        text=(
            "You are a deterministic entity extraction function.\n"
            "Extract the MOST IMPORTANT named entities from INPUT.\n\n"
            "RULES (CRITICAL):\n"
            "1) Return STRICT JSON only. No markdown, no commentary.\n"
            "2) Each entity 'name' MUST be an EXACT substring of INPUT.\n"
            "3) Prefer the MOST SPECIFIC mention.\n"
            "4) Only include entities that help answer the question.\n"
            "5) Return at most {max_entities} entities.\n"
            "6) Do not return generic nouns unless it is a proper name.\n\n"
            "JSON SHAPE:\n{schema}\n\n"
            "INPUT:\n{src}\n"
            "JSON:"
        ),
        required_vars={"schema", "src", "max_entities"},
        description="LLM entity extraction",
        version="1",
    ),
    PromptId.TOOL_QUERY_REFINER_MAIN: PromptTemplate(
        text=(
            "You convert CURRENT STEP TASK into the exact INPUT STRING required by one TOOL.\n"
            'Return STRICT JSON only: {{ "tool_input": "<string, <={hard_cap} chars>", "reason": "<short>" }}\n\n'
            "Global rules:\n"
            "1) Use USER QUESTION as the source of truth for facts and numbers.\n"
            "2) Use DEPENDENT STEP RESULTS as additional facts (when relevant).\n"
            "3) Do not invent facts. If something is missing, write the best tool_input you can from available context.\n"
            "4) Produce a NORMAL tool input string (not URL parameters).\n"
            "5) Obey the TOOL INSTRUCTIONS / INPUT FORMAT / REGEX.\n"
            "6) Output STRICT JSON only.\n\n"
            "TOOL CONTRACT:\n"
            "Name: {tool_name}\n"
            "Description: {tool_desc}\n"
            "Instructions: {tool_instr}\n"
            "Input format: {tool_fmt}\n"
            "Input regex (optional): {tool_regex}\n"
            "Forbidden (optional): {tool_forbid}\n"
            "Examples (optional):\n{examples}\n\n"
            "CONTEXT:\n"
            "USER QUESTION: {user_question}\n"
            "CURRENT STEP TASK: {step_task}\n\n"
            "DEPENDENT STEP RESULTS (if any):\n{dependency_block}\n\n"
            "JSON:"
        ),
        required_vars={
            "hard_cap",
            "tool_name",
            "tool_desc",
            "tool_instr",
            "tool_fmt",
            "tool_regex",
            "tool_forbid",
            "examples",
            "user_question",
            "step_task",
            "dependency_block",
        },
        description="Refine a step task into the exact tool input string according to tool contract",
        version="1",
    ),
    PromptId.TOOL_QUERY_REFINER_REPAIR: PromptTemplate(
        text=(
            "You will minimally fix a tool input string.\n"
            'Return STRICT JSON only: {{ "tool_input": "<corrected string>" }}\n\n'
            "TOOL NAME: {tool_name}\n"
            "DESCRIPTION: {tool_desc}\n"
            "INSTRUCTIONS: {tool_instr}\n"
            "INPUT FORMAT: {tool_fmt}\n"
            "INPUT REGEX (optional): {tool_regex}\n\n"
            "USER QUESTION: {user_question}\n"
            "CURRENT STEP TASK: {step_task}\n"
            "CURRENT CANDIDATE: {candidate}\n\n"
            "Constraints:\n"
            "- Do not add new facts.\n"
            "- Keep the user's intent and the step task.\n"
            "- Make the smallest edits necessary to comply with the tool contract/regex.\n"
            "- Output JSON only.\n"
            "JSON:"
        ),
        required_vars={
            "tool_name",
            "tool_desc",
            "tool_instr",
            "tool_fmt",
            "tool_regex",
            "user_question",
            "step_task",
            "candidate",
        },
        description="Repair a tool input candidate to comply with tool contract/regex with minimal edits",
        version="1",
    ),
    PromptId.CONF_ENTITY_ALIGNMENT: PromptTemplate(
        text=(
            "You are a strict evaluator for a QA system.\n\n"
            "Task: Decide whether the ANSWER is about the SAME REAL-WORLD ENTITY (or entities) as the QUESTION.\n"
            "Be careful with similarly-named words.\n"
            "If the answer is about a different entity with the same name, that is NOT the same entity.\n"
            "If the requested fact is missing, confidence should be 0.0.\n\n"
            "Output rules (CRITICAL):\n"
            "- Output MUST be a SINGLE JSON object and NOTHING else.\n"
            "- Keys MUST be exactly: same_entity, confidence, reason.\n"
            "- confidence MUST be one of: 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0\n"
            "- Do NOT include any extra keys.\n\n"
            "Schema example: {schema_example}\n\n"
            "{knowledge_cutoff_block}{result_timestamp_block}"
            "QUESTION:\n{question}\n\n"
            "ANSWER:\n{answer}\n\n"
            "{tool_result_block}\n"
            "JSON:"
        ),
        required_vars={
            "schema_example",
            "knowledge_cutoff_block",
            "result_timestamp_block",
            "question",
            "answer",
            "tool_result_block",
        },
        description="Assess whether answer addresses the same entity/entities as the question",
        version="1",
    ),
    PromptId.CONF_ANSWER_QUALITY: PromptTemplate(
        text=(
            "You are an evaluator for a QA system.\n\n"
            "Task: Score how completely the ANSWER provides the facts needed to answer the QUESTION.\n\n"
            "CRITICAL guidance:\n"
            "- The answer may be long/messy; judge presence of correct needed facts, not writing quality.\n\n"
            "Score meaning:\n"
            "- 1.0 = contains the key facts needed to answer (even if verbose/messy).\n"
            "- 0.5 = contains some relevant facts but misses key facts.\n"
            "- 0.0 = does not contain the requested fact / admits lack of info / or is outdated when real data is needed.\n\n"
            "Rules:\n"
            "- If TOOL RESULT contradicts the answer, lower the score.\n"
            "- If TOOL RESULT is irrelevant/empty, ignore it and judge using question/answer only.\n"
            "- Do NOT reward confident tone; reward correctness and presence of needed facts.\n\n"
            "Output rules (CRITICAL):\n"
            "- Output MUST be a SINGLE JSON object and NOTHING else.\n"
            "- Keys MUST be exactly: score, reason.\n"
            "- score MUST be one of: 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0\n"
            "- Do NOT include any extra keys.\n"
            "- reason MUST be a short string (<= 20 words).\n\n"
            "Schema example: {schema_example}\n\n"
            "{knowledge_cutoff_block}{result_timestamp_block}"
            "QUESTION:\n{question}\n\n"
            "ANSWER:\n{answer}"
            "{tool_result_block}\n"
            "JSON:"
        ),
        required_vars={
            "schema_example",
            "knowledge_cutoff_block",
            "result_timestamp_block",
            "question",
            "answer",
            "tool_result_block",
        },
        description="Assess completeness/quality of an answer, optionally using tool-result context",
        version="1",
    ),
    PromptId.CONF_ANSWER_REALISM: PromptTemplate(
        text=(
            "You are an evaluator for a QA system.\n\n"
            "Task: Assess REALISM of the ANSWER according to general world knowledge.\n"
            "Realism means: the answer is plausible, internally consistent, and not obviously fabricated.\n\n"
            "Important:\n"
            "- The answer may be long or messy; judge plausibility of stated facts, not writing quality.\n\n"
            "Scoring:\n"
            "- 1.0 = very plausible for the question.\n"
            "- 0.5 = somewhat plausible but has mild inconsistencies.\n"
            "- 0.0 = implausible / likely fabricated / not useful / requested fact is missing.\n\n"
            "Output rules (CRITICAL):\n"
            "- Output MUST be a SINGLE JSON object and NOTHING else.\n"
            "- Keys MUST be exactly: score, reason.\n"
            "- score MUST be one of: 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0\n"
            "- Do NOT include any extra keys.\n"
            "- reason MUST be a short string (<= 20 words).\n\n"
            "Schema example: {schema_example}\n\n"
            "{knowledge_cutoff_block}{result_timestamp_block}"
            "QUESTION:\n{question}\n\n"
            "ANSWER:\n{answer}"
            "{tool_result_block}\n"
            "JSON:"
        ),
        required_vars={
            "schema_example",
            "knowledge_cutoff_block",
            "result_timestamp_block",
            "question",
            "answer",
            "tool_result_block",
        },
        description="Assess plausibility/realism of an answer, optionally using tool-result context",
        version="1",
    ),
}