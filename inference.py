"""
inference.py - Baseline inference script for Customer Support Triage OpenEnv
Compliant with OpenEnv spec and competition judging criteria.
"""

import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai

# Load .env file (local dev only — judge env vars take precedence)
load_dotenv(override=False)

# ============ CONFIGURATION ============
# Standard competition vars — only API_BASE_URL and MODEL_NAME have defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default — judge injects this

# Use HF_TOKEN as the API key (judge's OpenAI-compatible endpoint authenticates via this).
# For local dev, fall back to GROQ/OPENAI keys. For open endpoints, use "EMPTY".
API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("GROQ_API_KEY")
    or HF_TOKEN
    or "EMPTY"
)

# ============ INITIALIZE CLIENT ============
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
    timeout=60.0
)

THROTTLE_SECONDS = int(os.getenv("THROTTLE_SECONDS", 1))

# ============ IMPORT ENV ============
import sys
sys.path.insert(0, ".")
from env.environment import CustomerSupportEnv
from env.models import Action, ActionType

SYSTEM_PROMPT = """You are an expert customer support triage agent.
You process tickets systematically: Classify -> Route -> Draft -> Resolve.

You MUST respond with a JSON action only:
{
  "action_type": "classify_priority",
  "ticket_id": "TKT-1000",
  "priority": "high",
  "reasoning": "Explain why"
}

Allowed actions: classify_priority, route_ticket, draft_response, resolve_ticket, escalate.
Allowed priorities: low, medium, high, critical.
Allowed departments: billing, technical, returns, general, escalation.
"""

def build_prompt(obs: dict, step: int) -> str:
    tickets = obs.get("inbox", [])
    prompt = f"Step: {step}\n"
    if obs.get("last_action_error"):
        prompt += f"ERROR FROM LAST STEP: {obs['last_action_error']}\n"
    else:
        prompt += f"LAST STEP RESULT: {obs.get('last_action_result', 'N/A')}\n"

    prompt += "\nActive Inbox:\n"
    for t in tickets:
        p = t.get('assigned_priority', 'NOT SET')
        d = t.get('assigned_department', 'NOT SET')
        r = "YES" if t.get('response_drafted') else "NO"
        prompt += f"- [{t['ticket_id']}] {t['subject']} (Priority={p}, Dept={d}, Drafted={r})\n"

    prompt += "\nNext action? (JSON only):"
    return prompt

def parse_action(response_text: str) -> dict:
    try:
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except:
        return {}

def run_task(task_id: str) -> float:
    # Required structured format: [START] task=NAME
    print(f"[START] task={task_id}", flush=True)
    env = CustomerSupportEnv()
    step_count = 0
    try:
        obs_model = env.reset(task_id=task_id)
        observation = obs_model.model_dump()
        total_reward = 0.0
        MAX_STEPS = 50

        for step in range(1, MAX_STEPS + 1):
            unresolved = [t for t in observation["inbox"] if not t.get("is_resolved")]
            if not unresolved:
                break

            time.sleep(THROTTLE_SECONDS)
            user_content = build_prompt(observation, step)

            @retry(
                stop=stop_after_attempt(5),
                wait=wait_exponential(multiplier=2, min=2, max=10),
                retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError))
            )
            def get_completion():
                return client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.0
                )

            try:
                completion = get_completion()
                response_text = completion.choices[0].message.content or ""
                action_data = parse_action(response_text)
                if not action_data:
                    # Required structured format: [STEP] step=N reward=X
                    print(f"[STEP] step={step} reward=0.001", flush=True)
                    step_count = step
                    continue

                action = Action(
                    action_type=ActionType(action_data["action_type"]),
                    ticket_id=action_data["ticket_id"],
                    priority=action_data.get("priority"),
                    department=action_data.get("department"),
                    response_text=action_data.get("response_text") or action_data.get("response"),
                    reasoning=action_data.get("reasoning", "")
                )
                result = env.step(action)
                observation = result.observation.model_dump()
                step_reward = result.reward.value
                total_reward += step_reward
                step_count = step

                # Required structured format: [STEP] step=N reward=X
                print(f"[STEP] step={step} reward={step_reward:.4f}", flush=True)

                if result.done:
                    break
            except Exception as e:
                print(f"[STEP] step={step} reward=0.001", flush=True)
                step_count = step

        raw_score = env.state().get("current_score", 0.001)
        # Judge requires score strictly between 0 and 1 (never 0.0 or 1.0 exactly)
        final_score = max(0.001, min(0.999, float(raw_score)))
        # Required structured format: [END] task=NAME score=X steps=N
        print(f"[END] task={task_id} score={final_score:.4f} steps={step_count}", flush=True)
        return final_score
    finally:
        env.close()

def main():
    tasks = ["task_easy", "task_medium", "task_hard"]
    scores = {}
    for task_id in tasks:
        scores[task_id] = run_task(task_id)
        time.sleep(2)

    avg = sum(scores.values()) / len(scores)
    print(f"\nFINAL AVERAGE SCORE: {avg:.4f}", flush=True)

    with open("baseline_scores.json", "w") as f:
        json.dump({"model": MODEL_NAME, "scores": scores, "average": avg}, f, indent=2)

if __name__ == "__main__":
    main()
