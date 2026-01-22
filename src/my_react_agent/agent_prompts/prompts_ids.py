from enum import Enum

class PromptId(str, Enum):
    PLAN_QUESTION_STEPS = "plan_question_steps"
    DECIDE_ACTION_FOR_STEP = "decide_action_for_step"
    SUMMARISE_OBSERVATION = "summarise_observation"
    LOW_CONF_TOOL_RESULT = "low_conf_tool_result"
    UPDATE_FACT_STORE = "update_fact_store"
