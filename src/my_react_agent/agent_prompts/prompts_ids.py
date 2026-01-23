from enum import Enum

class PromptId(str, Enum):
    # Planning / control
    SHOULD_SEPARATE_QUESTION = "should_separate_question"
    PLAN_QUESTION_STEPS = "plan_question_steps"
    DECIDE_ACTION_FOR_STEP = "decide_action_for_step"
    LOW_CONF_TOOL_RESULT = "low_conf_tool_result"

    # Execution / summarisation / memory
    SUMMARISE_OBSERVATION = "summarise_observation"
    UPDATE_FACT_STORE = "update_fact_store"
    REWRITE_STEP_TASK_WITH_CONTEXT = "rewrite_step_task_with_context"

    # Final answering
    SYNTHESISE_FINAL_ANSWER = "synthesise_final_answer"
    ANSWER_GENERATOR_MAIN = "answer_generator_main"
    ANSWER_GENERATOR_FALLBACK = "answer_generator_fallback"

    # Entity extraction
    LLM_EXTRACT_NAMES = "llm_extract_names"

    # Tool refiner
    TOOL_QUERY_REFINER_MAIN = "tool_query_refiner_main"
    TOOL_QUERY_REFINER_REPAIR = "tool_query_refiner_repair"

    # Confidence assessor
    CONF_ENTITY_ALIGNMENT = "confidence_entity_alignment"
    CONF_ANSWER_QUALITY = "confidence_answer_quality"
    CONF_ANSWER_REALISM = "confidence_answer_realism"
