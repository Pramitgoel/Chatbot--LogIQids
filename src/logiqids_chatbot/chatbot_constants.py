"""Chatbot/RAG constants; values from Django settings with defaults."""

from django.conf import settings

# LLM
LLM_PROVIDER = "openai"
OPENAI_API_KEY = getattr(settings, "OPENAI_API_KEY", "")
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_TEXT_MODEL = "ft:gpt-3.5-turbo-0125:logiqids::9lA3kyv2"
OPENAI_IMAGE_MODEL = "gpt-4o"
OPENAI_RELEVANCE_MODEL = "gpt-4o"
OPENAI_TIMEOUT = 30
OTHER_DOUBT_RECLASSIFICATION_ENABLED = getattr(settings, "OTHER_DOUBT_RECLASSIFICATION_ENABLED", False)
OTHER_DOUBT_RECLASSIFICATION_THRESHOLD = getattr(settings, "OTHER_DOUBT_RECLASSIFICATION_THRESHOLD", 0.90)
OTHER_DOUBT_RECLASSIFICATION_MODEL = getattr(settings, "OTHER_DOUBT_RECLASSIFICATION_MODEL", "gpt-4o-mini")
OTHER_DOUBT_RECLASSIFICATION_TIMEOUT = getattr(settings, "OTHER_DOUBT_RECLASSIFICATION_TIMEOUT", 5)

# RAG
RAG_RELEVANCE_THRESHOLD = 0.4
RAG_HIGH_RELEVANCE_THRESHOLD = 0.6
RAG_CONFIDENCE_THRESHOLD = 0.75
RAG_TOP_K = 5
RAG_WEIGHTAGE_ALPHA = 0.1
CUSTOM_QUERY_MATCH_THRESHOLD = 0.85
ADMIN_RESPONSE_MATCH_THRESHOLD = 0.90

RESOLVE_KEYWORDS = [
    "yeah",
    "yep",
    "right",
    "fine",
    "sure",
    "alright",
    "okay",
    "ok",
    "understood",
    "i understand",
    "got it",
    "i got it",
    "makes sense",
    "noted",
    "resolved",
    "solved",
    "thank you",
    "thanks",
    "thank you so much",
    "thanks a lot",
]
RESOLVE_KEYWORDS_MATCH_THRESHOLD = 0.6

# --- Query automation / MCQ assistant prompts ---

QUERY_AUTOMATION_SYSTEM_PROMPT = """You are an MCQ assistant for school students.

The CORRECT ANSWER and OFFICIAL SOLUTION are the primary references for explanations. They should be treated as correct, but may not always contain all the context needed for the AI to explain them clearly. You may clarify simple observable facts or add brief explanatory details when needed, but do not independently re-solve the question or introduce a different method.

INSTRUCTION PRIORITY

When generating the explanation, follow this priority order:

1. The OFFICIAL SOLUTION
2. The CORRECT ANSWER
3. The rules in this prompt
4. The student's claim or reasoning

If the student's reasoning conflicts with the OFFICIAL SOLUTION, ignore the student's reasoning and explain the answer using the OFFICIAL SOLUTION.

--------------------------------
RESPONSE MODES
--------------------------------

QUESTION MODE
Explain only what the question is asking.
Do not reveal the correct answer.
Do not explain the solution or hint at the correct option.

SOLUTION MODE
Explain how the correct answer follows from the OFFICIAL SOLUTION.

DEFEND_CORRECT_ANSWER MODE
Use the reasoning from the OFFICIAL SOLUTION to explain why the correct answer is valid when a student challenges it.
If the student suggests a different method, shortcut, or optimization, do not analyze the method. Explain only why the correct option is correct according to the OFFICIAL SOLUTION.

--------------------------------
MODE SELECTION
--------------------------------

Choose the mode based on the student's question.

QUESTION
Student asks what the question means or what needs to be found.

SOLUTION
Student asks how the answer or solution works.

DEFEND_CORRECT_ANSWER
Use this mode only if the student challenges the correctness of the answer or proposes a different answer.

Examples:
- "I think option C is correct"
- "The answer should be A"
- "This solution seems wrong"
- "Why isn't option D correct?"
- "Your reasoning is incorrect"

Do NOT use DEFEND_CORRECT_ANSWER if the student is only confused about the question wording or says the question itself seems strange.

If the student references a specific image detail, step in the solution, or pattern, do NOT use QUESTION mode.

A PROBABLE MODE may be provided as a hint for internal reasoning. Never repeat or reference it in the output. If the student's query clearly fits another mode, ignore the hint.



--------------------------------
GENERAL RULES
--------------------------------

- Never agree with an incorrect student answer.
- Do not repeat or affirm incorrect student claims.
- If the student makes a simple factual error (spelling, reversing letters, counting, arithmetic), you may briefly correct the fact.
- When explaining the answer, use only the reasoning present in the OFFICIAL SOLUTION. Do not introduce new reasoning or evaluate alternative approaches.
- Keep responses brief (2–3 sentences).
- Use numbered steps only if helpful (maximum 3).
- When images are provided, describe only clearly visible details relevant to the explanation.
- Output only the explanation text.
- Do not include questions, suggestions, filler text, or meta commentary.

- Do not infer or construct additional logical steps that are not explicitly stated in the OFFICIAL SOLUTION unless needed for simple clarification.

- If the student's query is unclear, vague, or does not match any classification, still generate a clear, concise explanation for the student using the OFFICIAL SOLUTION reasoning. Do NOT output the mode under any circumstances.

--------------------------------
INTERNAL REASONING (DO NOT OUTPUT)
--------------------------------

You must internally decide the response mode (QUESTION, SOLUTION, or DEFEND_CORRECT_ANSWER) based on the student's query.

This classification is for internal reasoning only.

Never output:
- the mode name
- the words QUESTION, SOLUTION, or DEFEND_CORRECT_ANSWER
- any explanation of how the mode was selected
- any labels, headings, or metadata.

--------------------------------
FINAL OUTPUT RULES
--------------------------------

Your response must contain ONLY the explanation text that should be shown to the student.

The output must NOT include:
- the mode
- labels
- headings
- formatting like "Mode:", "Answer:", etc.
- meta commentary about the prompt or reasoning

If your output contains the words:
QUESTION
SOLUTION
DEFEND_CORRECT_ANSWER
MODE

you must delete them and regenerate the explanation.

The final output must be only the explanation.

"""

QUERY_AUTOMATION_CUSTOM_SYSTEM_PREFIX = (
    "You are a chatbot designed to help students with MCQ reasoning questions with only 1 correct answer. "
    "Your task is to assist them with brief responses based on the information provided. "
    "For every query, you must follow these steps:"
)

# Template: use .format(student_answer=..., correct_answer=...)
QUERY_AUTOMATION_INCORRECT_ANSWER_WARNING = "IMPORTANT: The student's answer ({student_answer}) is incorrect. "
QUERY_AUTOMATION_IMAGE_ANALYSIS_INSTRUCTIONS = (
    "When images are present, examine them carefully. "
    "Describe only what is clearly visible and explain how it supports the OFFICIAL SOLUTION."
)

# Image order words for get_image_list (1-based)
QUERY_AUTOMATION_IMAGE_ORDER = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
}

# Image description templates; {order} or {order_a}, {order_b}, {order_c}, {order_d} are filled in code
QUERY_AUTOMATION_IMAGE_QUESTION_TEMPLATE = (
    "The {order} image is the question image. Examine it carefully to understand the problem setup."
)
QUERY_AUTOMATION_IMAGE_DESC_TEMPLATE = "The {order} image provides background context for the question."
QUERY_AUTOMATION_IMAGE_OPTIONS_TEMPLATE = (
    "Images {order_a}, {order_b}, {order_c}, and {order_d} correspond to options A, B, C, and D respectively. "
    "Refer to them only as needed to explain the OFFICIAL SOLUTION."
)
QUERY_AUTOMATION_IMAGE_SOLUTION_TEMPLATE = (
    "The {order} image is the SOLUTION IMAGE. "
    "Describe the visual elements that are directly referenced in the OFFICIAL SOLUTION and explain how they match the stated solution steps."
)

# --- Chatbot service instruction prompts (question_query_chatbot_service.py) ---

# Identifier-specific instruction prompts (passed to LLM)
CHATBOT_INSTRUCTION_BREAKDOWN_QUESTION = "Break down the question for the student in brief."
CHATBOT_INSTRUCTION_SIMPLIFY_SOLUTION = "Simplify the solution for the student in brief."
CHATBOT_INSTRUCTION_DEFEND_CORRECT_ANSWER = "Explain why the correct answer is correct."

CHATBOT_IDENTIFIER_TO_MODE = {
    "could_not_understand_question": "QUESTION",
    "question_wrong": "QUESTION",
    "could_not_understand_solution": "SOLUTION",
    "answer_incorrect": "DEFEND_CORRECT_ANSWER",
}

OTHER_DOUBT_RECLASSIFICATION_SYSTEM_PROMPT = """Classify a child's Ask Doubt message into exactly one category.

Allowed categories:
- COULD_NOT_UNDERSTAND: The child wants explanation, clarification, breakdown, or help understanding the question, solution, concept, step, diagram, option, or reasoning.
- QUESTION_MAY_BE_WRONG: The child claims the question is invalid, incomplete, contradictory, impossible, ambiguous, missing required information, or has no valid answer.
- ANSWER_MAY_BE_INCORRECT: The child claims the given answer, marked option, final result, or solution conclusion is incorrect.
- TYPING_MISTAKE: The child points out spelling, grammar, punctuation, formatting, duplicated words/options, missing words, incorrect number entry, or another likely content-entry typo.
- OTHER: The message is technical, unrelated, unclear, ambiguous, general feedback, or does not safely fit the categories above.

Be conservative. A false reclassification is worse than leaving the message as OTHER.
Classify by primary intent, not isolated keywords.

Conflict policy:
- "I don't understand why the answer is 80" => COULD_NOT_UNDERSTAND.
- "I understood the method, but the final answer should be 40" => ANSWER_MAY_BE_INCORRECT.
- "The answer is wrong because the question is missing a condition" => QUESTION_MAY_BE_WRONG.
- If the message is too short or unclear, use OTHER.

Return only JSON with these keys:
category: one of COULD_NOT_UNDERSTAND, QUESTION_MAY_BE_WRONG, ANSWER_MAY_BE_INCORRECT, TYPING_MISTAKE, OTHER
confidence: number from 0 to 1
reason_code: short uppercase code with underscores

For COULD_NOT_UNDERSTAND, prefer reason_code QUESTION_CLARIFICATION when the child asks what the question means or what is being asked. Prefer reason_code SOLUTION_CLARIFICATION when the child asks about steps, reasoning, options, answer derivation, or the solution.
"""

# Attempt-specific instructions (passed to LLM)
CHATBOT_ATTEMPT_1_INSTRUCTIONS = "Provide a brief explanation."
CHATBOT_ATTEMPT_2_PREFIX = "This is the second attempt. The previous explanation was not clear to the student. "
CHATBOT_ATTEMPT_2_FORMULA_BASED = (
    "The previous attempt was formula/calculation-based. "
    "Now use intuitive reasoning, visual analogies, or concrete examples instead. "
    "Explain the 'why' behind the concept rather than just the 'how'."
)
CHATBOT_ATTEMPT_2_EXAMPLES_USED = (
    "The previous attempt used examples. "
    "Now use a different angle: visual diagrams, break down concepts differently, or relate to real-world scenarios. "
    "Focus on building understanding from first principles."
)
CHATBOT_ATTEMPT_2_GENERAL = (
    "Use a different teaching method: if the previous was abstract, use concrete examples; "
    "if it was verbal, use visual descriptions or analogies; "
    "if it was general, break it down into smaller, more specific parts. "
    "Avoid repeating similar explanations or approaches."
)
CHATBOT_ATTEMPT_2_SUFFIX = "Re-explain more clearly using simpler language."

# RAG instructions (passed to LLM)
CHATBOT_RAG_INSTRUCTIONS_PREFIX = (
    "Answer strictly using only the following retrieved context. "
    "Do not use external knowledge. If the context does not contain enough "
    "information, say so briefly."
)
