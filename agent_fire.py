# -*- coding: utf-8 -*-
"""
组织经验淬炼师 - Knowledge Distiller Agent (FIRE 升级版)
KnowledgeForge Team | GDG Shanghai Gemma 4 Hackathon 2026
Track A: AI Agent

核心方法论：FIRE 经验萃取分析法
F - Function  职能领域
I - Instance  事件实例
R - Reasoning 推理判断 ★ (核心创新层)
E - Essence   底层原则
"""

import os
import json
from google import genai
from google.genai import types

# ── 模型配置 ──────────────────────────────────────────────
MODEL_MOE   = "models/gemma-4-26b-a4b-it"
MODEL_DENSE = "models/gemma-4-31b-it"

GENERATION_CONFIG = types.GenerateContentConfig(
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    max_output_tokens=4096,
)

# ── FIRE System Prompt ────────────────────────────────────
# 思考模式触发标记放在开头
# FIRE 四层方法论完整注入
SYSTEM_PROMPT = """<|think|>
你是【首席经验淬炼引导师】，由 KnowledgeForge 团队构建。
你掌握 FIRE 经验萃取分析法，专门从领域专家口中提炼隐性知识，转化为组织可复用的知识资产。

## FIRE 方法论四层框架

### F 层 · Function 职能领域
目标：识别专家负责的核心职责范围，确定本次萃取的聚焦领域。
触发条件：访谈开始时，或专家描述过于宽泛时使用。
代表问题：
- 「你最核心的工作职责是什么？」
- 「哪项职责最能体现你的独特价值？」
- 「新人最容易在哪里卡住？」
- 「如果培训一个接替你的人，你最先教什么？」

### I 层 · Instance 事件实例
目标：锁定一个具体真实发生的行为事件，用 STAR 结构完整还原。
触发条件：F层确认后，或专家说"我们一般怎么做"时（需拉回到具体事件）。
代表问题：
- 「能讲一个你印象最深、难度最高的具体案例吗？」
- 「当时的背景是什么？你的目标是什么？」
- 「你是怎么开始的？第一步做了什么？为什么？」
- 「决定成败的关键动作是什么？」
- 「如果换个人来，最容易在哪里出错？」

### R 层 · Reasoning 推理判断 ★ 核心创新层
目标：挖掘专家在关键时刻的决策逻辑与判断依据——这是真正的隐性知识。
触发条件：专家描述了一个关键行动或决策转折点之后，立刻深挖。
代表问题：
- 「你当时怎么判断要这么做？」
- 「在那个关键时刻，脑子里过的第一个念头是什么？」
- 「有没有一个信号，让你决定改变策略？」
- 「为什么别人做不来这个？这个判断力是怎么培养出来的？」
- 「如果用一句话描述你的判断标准，是什么？」

### E 层 · Essence 底层原则
目标：将专家的经验提炼为可跨场景迁移的核心认知原则与心智模式。
触发条件：R层已充分挖掘后，或访谈接近尾声时。
代表问题：
- 「这个方法在什么情况下会失效？有什么边界条件？」
- 「你做决策时，最看重什么？」
- 「你最想让团队的人少犯的错误是什么？」
- 「如果重来一次，哪个决策节点最关键？」

## 引导师铁律

### 层级识别与切换规则
- 专家说「我们一般怎么做」→ 停！拉回 I 层：「能给我讲一个具体的案例吗？」
- 专家说「当时我判断…」→ 进入 R 层，立刻深挖判断依据
- 专家说「那个人…」或描述情绪 → 进入 I 层情境还原
- 专家说「其实这背后的逻辑是…」→ 进入 E 层，帮助提炼原则

### 对话行为规则
1. 每次只问一个问题，绝不轰炸式追问
2. 先共情确认，再深度追问（「这个案例很有价值，」→ 「我想再深入了解一下…」）
3. 用关键词复述法确认理解：「我听到你说的核心是『先让对方开价』，是这样吗？」
4. 善用沉默：专家停顿时，不要急着填充，等待他们回忆和思考
5. 永远不替专家「总结答案」，而是引导他们自己说出来
6. 绝不问「你觉得应该怎么做」这类抽象问题，始终锚定具体事件

### 蒸馏时机判断
当满足以下条件时，主动提出蒸馏：
- F层：已确认职能领域 ✓
- I层：已还原至少一个完整事件（STAR结构清晰）✓
- R层：已挖掘至少一个关键判断逻辑 ✓
- E层：专家已表达或暗示了底层原则 ✓

提出方式：「我现在掌握了足够的信息，可以帮你整理成一张技能卡片了。要现在生成吗？」

## 输出规范
- 访谈阶段：简洁有力的对话引导，中文回复
- 蒸馏阶段：触发 save_skill_card 工具，输出结构化 JSON
- 每轮回复控制在100-200字以内，保持访谈节奏
"""

# ── 知识库（内存） ────────────────────────────────────────
skill_library: dict = {}
_fire_stage_tracker = {"F": False, "I": False, "R": False, "E": False}


# ── Tool 定义 ─────────────────────────────────────────────
TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="save_skill_card",
            description=(
                "仅当 FIRE 四层访谈充分完成、用户确认保存时调用。"
                "访谈进行中绝对不调用。"
                "保存结构化技能卡片到知识库。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "技能名称，包含核心动词和场景，不超过20字"
                    },
                    "fire_function": {
                        "type": "string",
                        "description": "F层：该技能所属的职能领域"
                    },
                    "fire_instance": {
                        "type": "string",
                        "description": "I层：触发该技能的典型事件场景描述"
                    },
                    "fire_reasoning": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "R层：关键判断逻辑列表，每条描述一个判断依据（最重要的层）"
                    },
                    "fire_essence": {
                        "type": "string",
                        "description": "E层：可跨场景迁移的底层认知原则，一句话"
                    },
                    "core_steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step":          {"type": "integer"},
                                "action":        {"type": "string"},
                                "key_decision":  {"type": "string"},
                                "pitfall":       {"type": "string"}
                            },
                            "required": ["step", "action", "key_decision"]
                        },
                        "description": "核心操作步骤，按执行顺序"
                    },
                    "failure_conditions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "E层补充：该技能失效的边界条件"
                    },
                    "applicable_scenarios": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "适用的业务场景列表"
                    },
                    "expert_name": {
                        "type": "string",
                        "description": "被访谈专家姓名或代号（可匿名）"
                    }
                },
                "required": [
                    "skill_name", "fire_function", "fire_instance",
                    "fire_reasoning", "fire_essence",
                    "core_steps", "applicable_scenarios"
                ]
            }
        ),
        types.FunctionDeclaration(
            name="search_skills",
            description=(
                "当用户提到某个领域或技能名称，"
                "需要查询知识库中是否已有相关技能卡片时调用。"
                "开始新访谈前优先调用，避免重复萃取。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词：技能名称、职能领域或场景描述"
                    }
                },
                "required": ["query"]
            }
        ),
        types.FunctionDeclaration(
            name="generate_interview_plan",
            description=(
                "当用户指定了要萃取的专家和领域，但尚未开始访谈时调用。"
                "根据 FIRE 框架生成定制化访谈计划。"
                "访谈已经开始后不调用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expert_role": {
                        "type": "string",
                        "description": "专家的职位或角色"
                    },
                    "domain": {
                        "type": "string",
                        "description": "要萃取的技能领域"
                    },
                    "context": {
                        "type": "string",
                        "description": "业务背景或特殊情况说明"
                    }
                },
                "required": ["expert_role", "domain"]
            }
        )
    ])
]


# ── 工具执行器 ────────────────────────────────────────────
def execute_tool(name: str, args: dict) -> dict:
    if name == "save_skill_card":
        skill_id = f"SKILL-{len(skill_library)+1:03d}"
        skill_library[skill_id] = args
        return {
            "success": True,
            "skill_id": skill_id,
            "message": f"✅ 技能卡片「{args['skill_name']}」已保存，ID: {skill_id}",
            "fire_layers_captured": {
                "F": bool(args.get("fire_function")),
                "I": bool(args.get("fire_instance")),
                "R": bool(args.get("fire_reasoning")),
                "E": bool(args.get("fire_essence")),
            }
        }

    elif name == "search_skills":
        query = args["query"].lower()
        results = []
        for sid, skill in skill_library.items():
            name_match = query in skill.get("skill_name", "").lower()
            func_match = query in skill.get("fire_function", "").lower()
            scene_match = any(query in s.lower()
                              for s in skill.get("applicable_scenarios", []))
            if name_match or func_match or scene_match:
                results.append({
                    "skill_id": sid,
                    "skill_name": skill["skill_name"],
                    "function":   skill.get("fire_function", ""),
                    "essence":    skill.get("fire_essence", ""),
                })
        return {
            "found": len(results),
            "results": results,
            "message": (f"找到 {len(results)} 个相关技能卡片"
                        if results else "知识库暂无相关内容，建议开始新访谈")
        }

    elif name == "generate_interview_plan":
        return {
            "status": "ready",
            "expert_role": args["expert_role"],
            "domain": args["domain"],
            "fire_plan": {
                "F层目标": f"识别{args['expert_role']}在{args['domain']}领域的核心职责",
                "I层目标": f"还原1-2个{args['domain']}的典型高难度事件",
                "R层目标": "挖掘关键决策时刻的判断逻辑（重点）",
                "E层目标": "提炼可迁移的底层认知原则",
            },
            "instruction": "请根据此计划开始 FIRE 四层访谈"
        }

    return {"error": f"未知工具: {name}"}


# ── 思考内容过滤（铁律）────────────────────────────────────
def extract_answer(response) -> str:
    """
    铁律：过滤所有思考块（part.thought == True）
    只返回最终答案部分
    Gemma 4 官方规范：思考内容绝不进入对话历史
    """
    parts = response.candidates[0].content.parts
    return "".join(
        p.text for p in parts
        if not p.thought and hasattr(p, "text") and p.text
    ).strip()


# ── Agentic Loop ──────────────────────────────────────────
def agent_turn(client, history: list, user_message: str,
               model: str = MODEL_MOE) -> str:
    """
    单轮对话，自动处理工具调用。
    历史记录只存储清洗后的答案（无思考内容）。
    """
    history.append({"role": "user", "parts": [{"text": user_message}]})

    config = types.GenerateContentConfig(
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        max_output_tokens=4096,
        tools=TOOLS,
        system_instruction=SYSTEM_PROMPT,
    )

    for _ in range(6):
        response = client.models.generate_content(
            model=model,
            contents=history,
            config=config,
        )

        parts = response.candidates[0].content.parts
        tool_calls = [
            p for p in parts
            if hasattr(p, "function_call") and p.function_call
        ]

        if not tool_calls:
            answer = extract_answer(response)
            # 铁律：清洗后存入历史
            history.append({"role": "model", "parts": [{"text": answer}]})
            return answer

        # 执行工具
        tool_results = []
        for part in tool_calls:
            fc = part.function_call
            result = execute_tool(fc.name, dict(fc.args))
            print(f"\n[工具调用] {fc.name}")
            print(f"[工具结果] {json.dumps(result, ensure_ascii=False, indent=2)}")
            tool_results.append(
                types.Part(function_response=types.FunctionResponse(
                    name=fc.name,
                    response={"result": result}
                ))
            )

        history.append({"role": "model", "parts": parts})
        history.append({"role": "user",  "parts": tool_results})

    return "[Agent] 工具调用轮数超限"


# ── 交互式访谈主程序 ──────────────────────────────────────
def run_interview():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    history = []

    print("=" * 60)
    print("  组织经验淬炼师 | KnowledgeForge")
    print("  Powered by Gemma 4 (26B MoE)")
    print("  方法论：FIRE 经验萃取分析法")
    print("  输入 'quit' 退出 | 'library' 查看知识库")
    print("=" * 60)

    opening = agent_turn(
        client, history,
        "你好！我准备开始一次专家经验萃取访谈。"
        "请先用 FIRE 框架介绍一下今天的访谈流程，然后开始引导我。"
    )
    print(f"\n引导师：{opening}\n")

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("\n访谈结束。")
            break
        if user_input.lower() == "library":
            print(f"\n📚 知识库：{len(skill_library)} 个技能卡片")
            for sid, s in skill_library.items():
                print(f"  {sid}: {s['skill_name']}")
                print(f"       F: {s.get('fire_function','')}")
                print(f"       E: {s.get('fire_essence','')}")
            print()
            continue

        response = agent_turn(client, history, user_input)
        print(f"\n引导师：{response}\n")

        if skill_library:
            print(f"  [知识库已保存 {len(skill_library)} 个技能卡片]\n")


if __name__ == "__main__":
    run_interview()
