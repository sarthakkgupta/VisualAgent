# VisualAgent

An AI-powered goal planning app that turns your goals into structured, time-boxed learning plans. Built with React + FastAPI, orchestrated by Azure AI agents.

## Features

- **AI Plan Generation** — Describe your goal in plain language and get a structured plan with tasks, subtasks, and timelines
- **Smart Duration Handling** — Plans are automatically sized to fit your specified timeframe
- **Progress Tracking** — Check off tasks and track completion percentage across all your plans
- **Plan Modification** — Modify existing plans with natural language requests
- **Dashboard** — Overview of all plans, progress stats, and recent activity
- **Authentication** — Secure login via Clerk

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, TypeScript, Vite, React Router, Clerk |
| Backend | FastAPI, Python 3.12 |
| AI | Azure AI Projects SDK, GPT-4 |
| Database | MongoDB Atlas |
| Auth | Clerk (frontend) |

## Project Structure

```
VisualAgent/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── models.py                # Pydantic request/response models
│   ├── requirements.txt
│   ├── api/
│   │   ├── plans.py             # POST /api/plan, POST /api/task/modify
│   │   ├── tasks.py             # Task CRUD + completion endpoints
│   │   └── health.py            # GET /api/health, GET /api/history
│   ├── db/
│   │   └── mongo_config.py      # MongoDB connection
│   └── sk_config/
│       ├── plugins.py           # Azure AI agent orchestration
│       └── plugins/             # Agent prompt configs
│           ├── goal_interpreter/
│           ├── task_breakdown/
│           ├── timeline_generator/
│           ├── plan_modifier/
│           └── scheduler/
└── frontend/
    ├── src/
    │   ├── main.tsx             # React entry point (Clerk + Router)
    │   ├── App.tsx              # Routes and navigation
    │   ├── types.ts             # TypeScript interfaces
    │   ├── pages/
    │   │   ├── HomePage.tsx
    │   │   ├── CreateGoal.tsx   # Goal input + plan generation
    │   │   ├── MyGoals.tsx      # Plan list with search/filter
    │   │   ├── GoalDetails.tsx  # Plan detail + progress tracking
    │   │   ├── DashboardPage.tsx
    │   │   ├── AboutPage.tsx
    │   │   └── PricingPage.tsx
    │   └── components/
    │       └── TypeWriter.tsx
    ├── package.json
    └── vite.config.ts           # Proxies /api → localhost:8000
```

## Prerequisites

- Python 3.12+
- Node.js 18+
- MongoDB Atlas cluster
- Azure AI Foundry project with a deployed GPT-4 model
- Clerk account

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/sarthakkgupta/VisualAgent.git
cd VisualAgent
```

### 2. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the **root** of the repo (`VisualAgent/.env`):

```env
AZURE_FOUNDRY_DEPLOYMENT=gpt-4.1
AZURE_FOUNDRY_ENDPOINT=https://<your-project>.services.ai.azure.com/api/projects/<project-name>
AZURE_OPENAI_API_KEY=<your-key>
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
```

### 3. Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env`:

```env
VITE_CLERK_PUBLISHABLE_KEY=pk_test_<your-clerk-key>
VITE_API_URL=http://127.0.0.1:8000
```

## Running Locally

Start the backend (from the `backend/` directory):

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Start the frontend (from the `frontend/` directory):

```bash
cd frontend
npm run dev
```

The app will be available at `http://localhost:5173`. The Vite dev server automatically proxies all `/api` requests to the backend on port 8000.

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/plan` | Generate a new AI plan |
| `GET` | `/api/history?user_id=` | Get all plans for a user |
| `GET` | `/api/task/{id}?user_id=` | Get a specific plan |
| `PUT` | `/api/task/{id}?user_id=` | Replace objective/tasks |
| `DELETE` | `/api/task/{id}?user_id=` | Delete a plan |
| `PATCH` | `/api/task/{id}/detail?user_id=` | Update a specific task field |
| `PATCH` | `/api/task/{id}/completion?user_id=` | Toggle task completion |
| `PATCH` | `/api/task/{id}/bulk-completion?user_id=` | Bulk toggle completions |
| `GET` | `/api/task/{id}/progress?user_id=` | Get progress stats |
| `POST` | `/api/task/modify` | Modify a plan with natural language |
| `GET` | `/api/health` | Health check |

### Example: Generate a Plan

```bash
curl -X POST http://localhost:8000/api/plan \
  -H "Content-Type: application/json" \
  -d '{"goal": "Learn Python in 2 weeks", "user_id": "user_123", "include_timeline": true}'
```

```json
{
  "objective": "Objective: Learn Python programming\nDuration: 2 weeks",
  "tasks": [
    {
      "title": "Python Basics",
      "content": "- Install Python and set up VS Code\n- Learn variables, types, and control flow\n- Complete exercises on Codecademy",
      "duration": "4 days",
      "completed": false
    }
  ]
}
```

## AI Agent Pipeline

Each plan is generated by a chain of Azure AI agents:

```
User Goal
   ↓
Goal Interpreter   →  Extracts objective + duration
   ↓
Task Breakdown     →  Creates 2–6 tasks with subtasks
   ↓
Timeline Generator →  Assigns durations that sum to total
   ↓
MongoDB            →  Stores completed plan
```

Plan modifications use a separate **Plan Modifier** agent that accepts the current plan JSON and a natural language change request.
