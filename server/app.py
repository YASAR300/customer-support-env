# server/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import sys
import os

# Add the parent directory to sys.path to allow importing from 'env'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.environment import CustomerSupportEnv
from env.models import Action, ActionType

app = FastAPI(
    title="Customer Support Triage OpenEnv",
    description="Real-world customer support ticket management environment for AI agents",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global environment instance
env = CustomerSupportEnv()


class ResetRequest(BaseModel):
    task_id: str = "task_easy"


class ActionRequest(BaseModel):
    action_type: str
    ticket_id: str
    priority: Optional[str] = None
    department: Optional[str] = None
    response_text: Optional[str] = None
    reasoning: Optional[str] = None


LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Customer Support Triage — OpenEnv</title>
<link href="https://fonts.googleapis.com/css2?family=Caveat:wght@600;700&family=Sora:wght@400;500;600&family=Fira+Code:wght@400;600&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Sora',sans-serif;background:#EBCBB0;color:#1C1107;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:48px 20px;position:relative;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background-image:radial-gradient(circle,rgba(139,90,43,0.09) 1px,transparent 1px);background-size:28px 28px;pointer-events:none;z-index:0}
*{position:relative;z-index:1}

.sketch{border:2.5px solid #7C5C3E;border-radius:255px 15px 225px 15px/15px 225px 15px 255px;box-shadow:5px 5px 0 rgba(124,92,62,0.22)}

.badge{background:#FFF3E8;border:2.5px solid #7C5C3E;padding:9px 24px;color:#5C3D1E;font-weight:600;font-size:0.82em;letter-spacing:2px;text-transform:uppercase;margin-bottom:30px;display:inline-flex;align-items:center;gap:10px}

h1{font-family:'Caveat',cursive;font-size:4.2em;font-weight:700;color:#1C1107;text-align:center;margin-bottom:14px;letter-spacing:1px;text-shadow:4px 4px 0 rgba(124,92,62,0.13)}
.subtitle{color:#5C4033;font-size:1.05em;margin-bottom:48px;text-align:center;max-width:640px;line-height:1.75}

.stats{display:flex;gap:0;margin-bottom:52px;background:#FFF3E8;overflow:hidden}
.stat{text-align:center;padding:24px 48px;position:relative}
.stat+.stat::before{content:'';position:absolute;left:0;top:18%;height:64%;width:2px;background:repeating-linear-gradient(to bottom,#7C5C3E 0,#7C5C3E 5px,transparent 5px,transparent 11px)}
.stat .num{font-family:'Caveat',cursive;font-size:3.4em;color:#5C3D1E;line-height:1;margin-bottom:5px}
.stat .label{font-size:0.7em;color:#7C5C3E;font-weight:600;text-transform:uppercase;letter-spacing:2px}

.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;max-width:980px;width:100%;margin-bottom:52px}
.card{background:#FFF3E8;padding:28px;border:2.5px solid #7C5C3E;border-radius:255px 15px 225px 15px/15px 225px 15px 255px;box-shadow:5px 5px 0 rgba(124,92,62,0.22);transition:all 0.18s}
.card:hover{transform:translate(-4px,-4px);box-shadow:9px 9px 0 rgba(124,92,62,0.32)}
.pill{display:inline-block;padding:4px 14px;font-size:0.76em;font-weight:700;border-radius:20px;margin-bottom:14px;letter-spacing:1px}
.pe{background:#D4EDDA;color:#1B5E20;border:1.5px solid #2D6A4F}
.pm{background:#FFF0CC;color:#7A4A00;border:1.5px solid #B5711A}
.ph{background:#FDDEDE;color:#7A1C1C;border:1.5px solid #A32D2D}
.card .title{font-size:1.18em;font-weight:600;margin-bottom:10px;color:#1C1107}
.card .desc{font-size:0.9em;color:#5C4033;line-height:1.65}
.meta{display:flex;gap:8px;margin-top:16px;flex-wrap:wrap}
.tag{font-family:'Fira Code',monospace;font-size:0.76em;padding:3px 10px;background:rgba(124,92,62,0.1);border:1.5px solid rgba(124,92,62,0.28);color:#5C3D1E;border-radius:4px}

.ep-section{max-width:980px;width:100%;background:#FFF3E8;padding:34px 40px;margin-bottom:44px;border:2.5px solid #7C5C3E;border-radius:255px 15px 225px 15px/15px 225px 15px 255px;box-shadow:5px 5px 0 rgba(124,92,62,0.22)}
.ep-section h2{font-family:'Caveat',cursive;font-size:2.7em;color:#1C1107;margin-bottom:26px}
.ep{display:flex;align-items:center;gap:14px;padding:15px 0;border-bottom:2px dashed rgba(124,92,62,0.32)}
.ep:last-child{border-bottom:none;padding-bottom:0}
.method{font-family:'Fira Code',monospace;padding:5px 14px;font-size:0.8em;font-weight:700;min-width:68px;text-align:center;border-radius:255px 15px 225px 15px/15px 225px 15px 255px}
.mpost{background:#D4EDDA;color:#1B5E20;border:2px solid #2D6A4F}
.mget{background:#D6EAF8;color:#154360;border:2px solid #1B4F72}
.ep-path{font-family:'Fira Code',monospace;font-size:0.98em;color:#1C1107;font-weight:600;min-width:90px}
.ep-desc{font-size:0.87em;color:#5C4033;flex:1}
.ep-arrow{margin-left:auto;color:#A0785A;font-size:1em;font-weight:600}

.btn-group{display:flex;gap:18px;flex-wrap:wrap;justify-content:center;margin-bottom:18px}
.btn{padding:14px 32px;font-family:'Sora',sans-serif;font-size:0.95em;font-weight:600;text-decoration:none;cursor:pointer;transition:all 0.18s;display:inline-flex;align-items:center;gap:10px;border-radius:255px 15px 225px 15px/15px 225px 15px 255px;border:2.5px solid}
.btn:hover{transform:translate(-4px,-4px)}
.btn-p{background:#2D6A4F;color:#FFF3E8;border-color:#1B4332;box-shadow:5px 5px 0 rgba(27,67,50,0.28)}
.btn-p:hover{box-shadow:9px 9px 0 rgba(27,67,50,0.35)}
.btn-s{background:#FFF3E8;color:#5C3D1E;border-color:#7C5C3E;box-shadow:5px 5px 0 rgba(124,92,62,0.22)}
.btn-s:hover{box-shadow:9px 9px 0 rgba(124,92,62,0.32)}

.footer{margin-top:12px;font-size:0.78em;color:#8C6B52;text-align:center;letter-spacing:0.5px}

@media(max-width:768px){
.grid{grid-template-columns:1fr}
.stats{flex-wrap:wrap}
.stat{padding:18px 28px}
.stat+.stat::before{display:none}
h1{font-size:3em}
.ep{flex-wrap:wrap}
.ep-arrow{display:none}
.ep-section{padding:24px 22px}
}
</style>
</head>
<body>

<div class="badge sketch">🏆 OpenEnv Compatible &nbsp;·&nbsp; Hackathon 2025</div>

<h1>🎫 Customer Support Triage</h1>
<p class="subtitle">
    A real-world AI environment where agents learn to triage, route,
    and resolve customer support tickets — built for the OpenEnv Hackathon.
</p>

<div class="stats sketch">
    <div class="stat">
        <div class="num">3</div>
        <div class="label">Tasks</div>
    </div>
    <div class="stat">
        <div class="num">5→15</div>
        <div class="label">Tickets</div>
    </div>
    <div class="stat">
        <div class="num">±1.0</div>
        <div class="label">Reward Range</div>
    </div>
    <div class="stat">
        <div class="num">5</div>
        <div class="label">Action Types</div>
    </div>
</div>

<div class="grid">
    <div class="card">
        <div class="pill pe">● EASY</div>
        <div class="title">Priority Classification</div>
        <div class="desc">Classify 5 tickets by urgency level — from low priority to critical incidents.</div>
        <div class="meta">
            <span class="tag">5 tickets</span>
            <span class="tag">10 steps</span>
            <span class="tag">classify_priority</span>
        </div>
    </div>
    <div class="card">
        <div class="pill pm">● MEDIUM</div>
        <div class="title">Department Routing</div>
        <div class="desc">Classify and route 10 tickets to the correct support department accurately.</div>
        <div class="meta">
            <span class="tag">10 tickets</span>
            <span class="tag">25 steps</span>
            <span class="tag">route_ticket</span>
        </div>
    </div>
    <div class="card">
        <div class="pill ph">● HARD</div>
        <div class="title">Full Resolution Pipeline</div>
        <div class="desc">Triage, route, draft responses and resolve 15 tickets within SLA windows.</div>
        <div class="meta">
            <span class="tag">15 tickets</span>
            <span class="tag">70 steps</span>
            <span class="tag">resolve_ticket</span>
        </div>
    </div>
</div>

<div class="ep-section">
    <h2>📡 API Endpoints</h2>
    <div class="ep">
        <span class="method mpost">POST</span>
        <span class="ep-path">/reset</span>
        <span class="ep-desc">Start a new episode — body: <code>{"task_id": "task_easy"}</code></span>
        <span class="ep-arrow">→</span>
    </div>
    <div class="ep">
        <span class="method mpost">POST</span>
        <span class="ep-path">/step</span>
        <span class="ep-desc">Take an action — returns observation, reward, done</span>
        <span class="ep-arrow">→</span>
    </div>
    <div class="ep">
        <span class="method mget">GET</span>
        <span class="ep-path">/state</span>
        <span class="ep-desc">Get the current full environment state</span>
        <span class="ep-arrow">→</span>
    </div>
    <div class="ep">
        <span class="method mget">GET</span>
        <span class="ep-path">/tasks</span>
        <span class="ep-desc">List all available tasks with metadata and difficulty</span>
        <span class="ep-arrow">→</span>
    </div>
    <div class="ep">
        <span class="method mget">GET</span>
        <span class="ep-path">/health</span>
        <span class="ep-desc">Health check — returns <code>{"status": "healthy"}</code></span>
        <span class="ep-arrow">→</span>
    </div>
    <div class="ep">
        <span class="method mget">GET</span>
        <span class="ep-path">/docs</span>
        <span class="ep-desc">Interactive Swagger UI — try all endpoints live</span>
        <span class="ep-arrow">→</span>
    </div>
</div>

<div class="btn-group">
    <a href="/docs" class="btn btn-p">📖 Open API Docs</a>
    <a href="/tasks" class="btn btn-s">📋 View All Tasks</a>
    <a href="/health" class="btn btn-s">❤️ Health Check</a>
</div>

<div class="footer">Customer Support Triage &nbsp;·&nbsp; OpenEnv v1.0.0</div>

</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def root():
    return LANDING_HTML


@app.get("/health")
def health():
    return {"status": "healthy"}

SWAGGER_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Caveat:wght@700&family=Sora:wght@400;600&family=Fira+Code:wght@400;600&display=swap');

body{background:#EBCBB0!important;font-family:'Sora',sans-serif!important;background-image:radial-gradient(circle,rgba(139,90,43,0.09) 1px,transparent 1px);background-size:28px 28px}
.swagger-ui{color:#1C1107!important;font-family:'Sora',sans-serif!important}
.swagger-ui .info .title{color:#1C1107!important;font-family:'Caveat',cursive!important;font-size:3.6em!important;text-shadow:3px 3px 0 rgba(124,92,62,0.15)}
.swagger-ui .info p{color:#5C4033!important;font-size:1.1em!important}
.swagger-ui .scheme-container{background:transparent!important;box-shadow:none!important;border:none!important}

.swagger-ui .opblock{
border:2.5px solid #7C5C3E!important;
border-radius:255px 15px 225px 15px/15px 225px 15px 255px!important;
box-shadow:5px 5px 0 rgba(124,92,62,0.22)!important;
background:#FFF3E8!important;margin-bottom:20px!important
}
.swagger-ui .opblock .opblock-summary{border-bottom:none!important}
.swagger-ui .opblock .opblock-summary-method{
border-radius:255px 15px 225px 15px/15px 225px 15px 255px!important;
font-family:'Fira Code',monospace!important;font-weight:700!important
}
.swagger-ui .opblock-post .opblock-summary-method{background:#D4EDDA!important;color:#1B5E20!important;border:2px solid #2D6A4F!important}
.swagger-ui .opblock-get .opblock-summary-method{background:#D6EAF8!important;color:#154360!important;border:2px solid #1B4F72!important}
.swagger-ui .opblock .opblock-summary-path{color:#1C1107!important;font-family:'Fira Code',monospace!important;font-size:1.05em!important}
.swagger-ui .opblock .opblock-summary-description{color:#5C4033!important}

.swagger-ui .btn:not(.cancel){
background:#2D6A4F!important;color:#FFF3E8!important;
border:2px solid #1B4332!important;
border-radius:255px 15px 225px 15px/15px 225px 15px 255px!important;
box-shadow:4px 4px 0 rgba(27,67,50,0.28)!important;
font-weight:600!important;font-family:'Sora',sans-serif!important;
transition:all 0.18s
}
.swagger-ui .btn:not(.cancel):hover{transform:translate(-3px,-3px)!important;box-shadow:7px 7px 0 rgba(27,67,50,0.35)!important}
.swagger-ui .btn.cancel{background:#FFF3E8!important;border-color:#7C5C3E!important;color:#5C3D1E!important}

.swagger-ui table thead tr th,.swagger-ui table thead tr td{color:#1C1107!important;font-family:'Fira Code',monospace!important;border-bottom:2px dashed rgba(124,92,62,0.35)!important}
.swagger-ui .parameters-col_name{color:#2D6A4F!important;font-weight:700!important}
.swagger-ui .parameter__type{color:#1B4F72!important}
.swagger-ui .parameter__in{color:#7C5C3E!important}
.swagger-ui input[type=text],.swagger-ui textarea{
background:#FFF3E8!important;color:#1C1107!important;
border:2px solid #7C5C3E!important;border-radius:8px!important;
font-family:'Fira Code',monospace!important
}

.swagger-ui .response-col_status{color:#2D6A4F!important;font-weight:700!important;font-size:1.1em!important}
.swagger-ui .highlight-code{border:2px solid rgba(124,92,62,0.35)!important;border-radius:10px!important;background:#FFF3E8!important}
.swagger-ui .microlight{color:#1B4F72!important;font-family:'Fira Code',monospace!important}

.swagger-ui section.models{border:2.5px solid #7C5C3E!important;border-radius:16px!important;background:#FFF3E8!important;margin-top:30px!important;padding:12px!important}
.swagger-ui section.models h4{color:#1C1107!important;font-family:'Caveat',cursive!important;font-size:2.2em!important;border-bottom:2px dashed rgba(124,92,62,0.35)!important;padding-bottom:10px!important}
.swagger-ui section.models h4 *{color:#1C1107!important}
.swagger-ui section.models .model-container{background:#EBCBB0!important;border:1.5px solid rgba(124,92,62,0.35)!important;border-radius:10px!important;margin-bottom:14px!important;padding:14px!important}
.swagger-ui .model{color:#1C1107!important;font-family:'Fira Code',monospace!important}
.swagger-ui .model-title{color:#2D6A4F!important;font-weight:700!important}
.swagger-ui .prop-name{color:#1B4F72!important;font-weight:700!important}
.swagger-ui .prop-type{color:#B5711A!important}
.swagger-ui table.model tbody tr td{border-bottom:1px dashed rgba(124,92,62,0.25)!important;padding:10px!important}
.swagger-ui .model-toggle:after{filter:none!important}

.swagger-ui .opblock .opblock-section-header{background:rgba(124,92,62,0.06)!important;border-bottom:1px dashed rgba(124,92,62,0.3)!important}
.swagger-ui .opblock .opblock-section-header h4{color:#1C1107!important;font-family:'Sora',sans-serif!important}
.swagger-ui select{background:#FFF3E8!important;color:#1C1107!important;border:1.5px solid #7C5C3E!important}
.swagger-ui .responses-inner h4,.swagger-ui .responses-inner h5{color:#1C1107!important}
.swagger-ui .markdown p{color:#5C4033!important}
.swagger-ui .topbar{display:none!important}
svg{fill:#5C4033!important}
</style>"""


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    from fastapi.openapi.docs import get_swagger_ui_html
    html = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API Docs",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )
    response_body = html.body.decode("utf-8")
    new_body = response_body.replace("</head>", SWAGGER_CSS + "</head>")
    return HTMLResponse(new_body)


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None):
    try:
        task_id = "task_easy"
        if request and request.task_id:
            task_id = request.task_id
        obs = env.reset(task_id=task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(request: ActionRequest):
    try:
        action = Action(
            action_type=ActionType(request.action_type),
            ticket_id=request.ticket_id,
            priority=request.priority,
            department=request.department,
            response_text=request.response_text,
            reasoning=request.reasoning
        )
        result = env.step(action)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    return env.state()


@app.get("/tasks")
def list_tasks():
    from env.tasks import TASKS
    return {
        task_id: {
            "name": task.name,
            "difficulty": task.difficulty,
            "description": task.description,
            "max_steps": task.max_steps,
            "ticket_count": task.ticket_count
        }
        for task_id, task in TASKS.items()
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
