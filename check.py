# check.py - Final verification script
import subprocess
import requests
import json
import os
import sys

print("🔍 PRE-SUBMISSION CHECKLIST")
print("="*50)

# 1. Python version
py_version = sys.version_info
status = "✅" if py_version >= (3, 10) else "❌"
print(f"{status} Python version: {sys.version}")

# 2. Required packages
packages = ["fastapi", "uvicorn", "pydantic", "openai", "pytest"]
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✅ Package {pkg}: installed")
    except ImportError:
        print(f"❌ Package {pkg}: NOT INSTALLED")

# 3. Environment imports
try:
    from env.environment import CustomerSupportEnv
    from env.models import Action, ActionType, Priority
    print("✅ Environment imports: OK")
except Exception as e:
    print(f"❌ Environment imports: {e}")

# 4. All 3 tasks
try:
    env = CustomerSupportEnv()
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        obs = env.reset(task_id)
        assert obs.task_id == task_id
    print("✅ All 3 tasks: OK")
except Exception as e:
    print(f"❌ Tasks: {e}")

# 5. Graders return 0.0-1.0
try:
    from env.graders import get_grader
    env = CustomerSupportEnv()
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        env.reset(task_id)
        grader = get_grader(task_id)
        score = grader.grade(env.tickets, 5, 30)
        assert 0.0 <= score <= 1.0
    print("✅ Graders score range: OK")
except Exception as e:
    print(f"❌ Graders: {e}")

# 6. step/reset/state work
try:
    env = CustomerSupportEnv()
    obs = env.reset("task_easy")
    action = Action(
        action_type=ActionType.CLASSIFY_PRIORITY,
        ticket_id=obs.inbox[0]["ticket_id"],
        priority=Priority.HIGH
    )
    result = env.step(action)
    state = env.state()
    assert -1.0 <= result.reward.value <= 1.0
    print("✅ step/reset/state: OK")
except Exception as e:
    print(f"❌ step/reset/state: {e}")

# 7. Files exist
required_files = [
    "openenv.yaml",
    "Dockerfile", 
    "inference.py",
    "README.md",
    "server/app.py"  # updated mapping based on our fix
]
for f in required_files:
    status = "✅" if os.path.exists(f) else "❌"
    print(f"{status} File {f}: {'exists' if os.path.exists(f) else 'MISSING'}")

print("\n" + "="*50)
print("✅ = PASS | ❌ = FIX THIS")
