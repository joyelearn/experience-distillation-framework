# -*- coding: utf-8 -*-
"""
组织经验淬炼师 - Knowledge Distiller Agent
Joyelearn Team | GDG Shanghai Gemma 4 Hackathon 2026
Track: A (AI Agent)

Core features:
- Multi-turn BEI interview guidance
- Gemma 4 Thinking Mode (thought filtering)
- Native Function Calling (save_skill_card, search_skills, generate_questions)
- 256K context SOP distillation
"""

import os
import json
from google import genai
from google.genai import types

# ── Model Configuration ───────────────────────────────────
MODEL_MOE   = "models/gemma-4-26b-a4b-it"   # Primary: MoE, fast + smart
MODEL_DENSE = "models/gemma-4-31b-it"        # Secondary: Dense, deep reasoning

# ── Official Sampling Parameters (Gemma 4 spec) ──────────
GENERATION_CONFIG = types.GenerateContentConfig(
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    max_output_tokens=4096,
)

# ── System Prompt (BEI Interview Guide) ──────────────────
SYSTEM_PROMPT = """You are the Chief Knowledge Distillation Guide, built by Joyelearn.
Your mission: extract tacit knowledge from domain experts through structured interviews,
then distill it into reusable organizational assets (SOPs, Skill Cards).

## Interview Framework
Use BEI (Behavioral Event Interview) method:
1. Ask for a SPECIFIC past event (not opinions or hypotheticals)
2. Follow STAR structure: Situation → Task → Action → Result
3. Probe decision points: "What made you choose that approach?"
4. Surface hidden logic: "What would a less experienced person have missed here?"

## Iron Rules
- Ask only ONE question per turn
- Never summarize for the expert — make them say it themselves
- When enough context is gathered, offer: "I can now generate a Skill Card. Shall I?"

## Output Format (when distilling)
Always output structured JSON for skill cards and SOPs.
"""

# ── In-memory Skill Library ───────────────────────────────
skill_library = {}


# ── Tool Definitions ──────────────────────────────────────
TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="save_skill_card",
            description=(
                "Call this ONLY when the interview is complete and the user confirms "
                "they want to save the distilled knowledge as a skill card. "
                "Do NOT call during active interview."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill, concise and specific"
                    },
                    "applicable_scenarios": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Business scenarios where this skill applies"
                    },
                    "core_steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step":         {"type": "integer"},
                                "action":       {"type": "string"},
                                "key_decision": {"type": "string"},
                                "pitfall":      {"type": "string"}
                            },
                            "required": ["step", "action", "key_decision"]
                        },
                        "description": "Step-by-step actions with decision points"
                    },
                    "underlying_principle": {
                        "type": "string",
                        "description": "The core mental model behind this skill"
                    },
                    "expert_name": {
                        "type": "string",
                        "description": "Name or alias of the expert interviewed"
                    }
                },
                "required": ["skill_name", "applicable_scenarios",
                             "core_steps", "underlying_principle"]
            }
        ),
        types.FunctionDeclaration(
            name="search_skills",
            description=(
                "Call this when the user mentions a domain or skill topic "
                "and wants to check if similar knowledge already exists in the library."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword: skill name, domain, or scenario"
                    }
                },
                "required": ["query"]
            }
        ),
        types.FunctionDeclaration(
            name="generate_sop",
            description=(
                "Call this when the user wants to convert the interview transcript "
                "into a formal Standard Operating Procedure document."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "The domain or process to generate SOP for"
                    },
                    "context": {
                        "type": "string",
                        "description": "Summary of key insights from the interview"
                    }
                },
                "required": ["domain", "context"]
            }
        )
    ])
]


# ── Tool Executors ────────────────────────────────────────
def execute_tool(name: str, args: dict) -> dict:
    if name == "save_skill_card":
        skill_id = f"SKILL-{len(skill_library)+1:03d}"
        skill_library[skill_id] = args
        return {
            "success": True,
            "skill_id": skill_id,
            "message": f"Skill card '{args['skill_name']}' saved successfully. ID: {skill_id}"
        }

    elif name == "search_skills":
        query = args["query"].lower()
        results = []
        for sid, skill in skill_library.items():
            if query in skill.get("skill_name", "").lower():
                results.append({"id": sid, "name": skill["skill_name"]})
        return {
            "found": len(results),
            "results": results,
            "message": f"Found {len(results)} matching skills" if results
                       else "No matching skills found. Ready to start new interview."
        }

    elif name == "generate_sop":
        return {
            "status": "ready",
            "domain": args["domain"],
            "instruction": "Generate a structured SOP based on the interview context provided."
        }

    return {"error": f"Unknown tool: {name}"}


# ── Core: Extract Answer (filter thought blocks) ──────────
def extract_answer(response) -> str:
    """
    IRON RULE: Filter out all thought blocks (part.thought == True).
    Only return the final answer parts.
    This is mandatory per Gemma 4 spec — thought content must NEVER
    be stored in conversation history.
    """
    parts = response.candidates[0].content.parts
    answer = ""
    for part in parts:
        if not part.thought and part.text:
            answer += part.text
    return answer.strip()


# ── Core: Agentic Loop ────────────────────────────────────
def agent_turn(client, history: list, user_message: str, model: str = MODEL_MOE) -> str:
    """
    Single conversation turn with automatic tool calling.
    Handles: user input → model response → tool execution → final answer
    History stores ONLY clean answers (no thought content).
    """
    # Add user message to history
    history.append({"role": "user", "parts": [{"text": user_message}]})

    config = types.GenerateContentConfig(
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        max_output_tokens=4096,
        tools=TOOLS,
        system_instruction=SYSTEM_PROMPT,
    )

    # Agentic loop: handle tool calls
    for _ in range(6):  # max 6 tool rounds per turn
        response = client.models.generate_content(
            model=model,
            contents=history,
            config=config,
        )

        parts = response.candidates[0].content.parts
        tool_calls = [p for p in parts if hasattr(p, "function_call") and p.function_call]

        if not tool_calls:
            # No tool calls — extract final answer
            answer = extract_answer(response)
            # IRON RULE: store only clean answer in history
            history.append({"role": "model", "parts": [{"text": answer}]})
            return answer

        # Execute tools and collect results
        tool_results = []
        for part in tool_calls:
            fc = part.function_call
            result = execute_tool(fc.name, dict(fc.args))
            print(f"[Tool Called] {fc.name} → {result}")
            tool_results.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result}
                    )
                )
            )

        # Append tool call + results to history
        history.append({"role": "model", "parts": parts})
        history.append({"role": "user", "parts": tool_results})

    return "[Agent] Max tool rounds reached."


# ── Main: Interactive Interview Session ───────────────────
def run_interview():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    history = []

    print("=" * 60)
    print("  Knowledge Distiller Agent | Joyelearn")
    print("  Powered by Gemma 4 (26B MoE)")
    print("  Type 'quit' to exit | 'save' to save skill card")
    print("=" * 60)
    print()

    # Opening message
    opening = agent_turn(
        client, history,
        "Hello! I'm ready to start a knowledge extraction interview. "
        "Please tell me: which expert's experience do you want to distill today, "
        "and what domain or skill are we focusing on?"
    )
    print(f"Guide: {opening}\n")

    # Interactive loop
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Session ended.")
            break

        response = agent_turn(client, history, user_input)
        print(f"\nGuide: {response}\n")

        # Show skill library status
        if skill_library:
            print(f"[Library] {len(skill_library)} skill(s) saved\n")


if __name__ == "__main__":
    run_interview()
