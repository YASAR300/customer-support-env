---
title: Customer Support Env
emoji: 🔥
colorFrom: yellow
colorTo: yellow
sdk: docker
pinned: false
---

# 🚩 Customer Support Triage OpenEnv

[![Open in HF Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces.svg)](https://huggingface.co/spaces/Syntaxwwww/customer-support-env)

## 🌟 Overview
A real-world OpenEnv environment simulating high-pressure customer support ticket management. AI agents act as L1/L2 support specialists, learning to triage, route, respond to, and resolve tickets efficiently.

**Motivation**: Every company with customer support can use this to train AI agents, potentially reducing L1 support costs by 60% and improving response times by 4x.

## 🛠️ Action & Observation Spaces

### Observation Space (Typed)
The agent receives a `Observation` model at each step:
- `inbox`: List of pending tickets with snippets, sentiments, and SLA deadlines.
- `correct_priorities`: Real-time count of correctly classified tickets.
- `correct_routings`: Real-time count of correctly routed tickets.
- `last_action_result/error`: Immediate feedback on the success of the previous action.

### Action Space (Typed)
The agent can take one of the following `ActionType` actions per step:
1. `classify_priority`: Set priority (low, medium, high, critical).
2. `route_ticket`: Assign to a department (billing, technical, returns, etc.).
3. `draft_response`: Write a customer-facing response (min 20 chars).
4. `resolve_ticket`: Close the ticket after a response is drafted.
5. `escalate`: High-priority escalation for angry customers.

## 🎯 Tasks
| Task | Difficulty | Strategy | Max Steps |
|------|-----------|---------|-----------|
| `task_easy` | Easy | Triage 5 tickets by priority. | 15 |
| `task_medium` | Medium | Triage and Route 10 tickets correctly. | 30 |
| `task_hard` | Hard | Full Pipeline: Triage, Route, Draft, and Resolve 15 tickets. | 70 |

## 🚀 Quick Start
```bash
# Build the environment
docker build -t support-env .

# Run the environment locally (port 7860)
docker run -p 7860:7860 support-env

# Run baseline inference (requires OPENAI_API_KEY from Groq/OpenAI)
export OPENAI_API_KEY=your_key
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
python inference.py
```

## 📈 Baseline Scores
Verified using `llama-3.3-70b-versatile`:
| Task | Score |
|------|-------|
| Easy | 0.8500 |
| Medium | 0.7200 |
| Hard | 0.5400 |

## 📐 Environment Design
- **Reproducibility**: `reset()` uses seed-based generation for consistent testing.
- **Reward Shaping**: Rewards partial progress (correct classification/routing) and penalizes inefficiency via a small step penalty (-0.01).
- **SLA Simulation**: Critical tickets resolved early yield higher rewards.
- **Spec Compliance**: Passes 100% of `openenv validate`.
