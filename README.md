# VisualAgent

Transform your goals into structured, AI-powered learning plans with timelines you can actually follow.

## The Problem

Most goal-setting tools are linear and rigid—they don't understand the nuance of what you're trying to learn or how much time you actually have. You describe a goal, get generic suggestions, and spend hours manually breaking it down into actionable steps. Whether you're learning a new skill, preparing for a certification, or planning a major project, you're left managing complexity without intelligent structure.

## The Solution

VisualAgent uses Azure AI agents to intelligently parse your goals and automatically generate structured, time-boxed learning plans. Describe what you want to achieve and how long you have—the system decomposes it into tasks, subtasks, and timelines that adapt to your constraints. You get real-time progress tracking, the ability to modify plans with natural language, and a clean dashboard to oversee all your goals.

## Demo

[Add screenshot/GIF of dashboard and plan generation here]

The app shows:
- **Create Goal Page**: Clean input form with goal description and desired timeline
- **Dashboard**: Overview of all your plans, progress percentages, and recent activity
- **Plan Details**: Expandable task hierarchy with completion tracking and real-time updates

## How to Use

### Prerequisites

- Python 3.12+
- Node.js 18+
- MongoDB Atlas cluster
- Azure AI Foundry project with a deployed GPT-4 model
- Clerk account

### Setup

#### 1. Clone the repo

```bash
git clone https://github.com/sarthakkgupta/VisualAgent.git
cd VisualAgent
```

#### 2. Backend Setup

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

#### 3. Frontend Setup

```bash
cd frontend
npm install
```

Create `frontend/.env`:

```env
VITE_CLERK_PUBLISHABLE_KEY=pk_test_<your-clerk-key>
VITE_API_URL=http://127.0.0.1:8000
```

### Run

Start the backend from the `backend/` directory:

```bash
cd backend
uvicorn main:app --reload --port 8000
```

In a new terminal, start the frontend from the `frontend/` directory:

```bash
cd frontend
npm run dev
```

Navigate to `http://localhost:5173` in your browser. The Vite dev server automatically proxies `/api` requests to the backend on port 8000.

### Example

**Input:**
```
Goal: Learn React functional components
Duration: 2 weeks
```

**Output:**
```
Plan: Learn React Functional Components (2 weeks)
├── Week 1: Fundamentals
│   ├── Understand React Hooks: useState, useEffect
│   ├── Build 3 practice components
│   └── Review best practices
└── Week 2: Advanced Patterns
    ├── Custom Hooks implementation
    ├── Performance optimization
    └── Final project: Todo app
```

Task completion is tracked as you check off items, and you can modify the plan at any time with natural language.

## How It Works

VisualAgent orchestrates multiple Azure AI agents to create your learning plan:

1. **Goal Interpreter** — Parses your objective and identifies key learning areas
2. **Task Breakdown** — Decomposes the goal into hierarchical tasks and subtasks
3. **Timeline Generator** — Distributes tasks across your specified timeframe
4. **Scheduler** — Creates checkpoints and milestones for accountability
5. **Plan Modifier** — Re-orchestrates the plan if you request changes

The frontend (React + TypeScript) provides a responsive UI with task completion tracking, while the FastAPI backend manages MongoDB storage and Azure AI orchestration. Clerk handles user authentication, keeping data scoped to individual users.

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

## Tradeoffs and Decisions

- **Azure AI Agents over LLM chains**: We chose multi-agent orchestration with Azure AI Projects SDK instead of simple prompt chains because it provides better composability, error handling, and the ability to have specialized agents focus on specific aspects (goal parsing, decomposition, timeline fitting). Trade-off: added complexity in orchestration logic compared to a single GPT-4 call, but gained more reliable, structured outputs.

- **Storing in MongoDB over Azure Cosmos DB**: We opted for MongoDB Atlas for faster prototyping and simpler schema flexibility during active development. We'd likely migrate to Cosmos DB with native JSON support for production scaling, as it provides better multi-region support and automatic indexing.

## What I Learned

1. **LLM output is unpredictable without structure** — Naive JSON parsing from GPT-4 broke frequently on malformed output. Using Azure AI agents with defined output schemas and the Plans API reduced hallucination and made the pipeline 10x more reliable.

2. **Timeline fitting is non-trivial** — Getting the timeline generator to respect total duration while distributing realistic task times required iterative prompt refinement and validation logic. A human still needs to sanity-check plan durations.

3. **Auth complexity compounds complexity** — Integrating Clerk for authentication seemed simple but required careful handling of user_id scoping across the backend, MongoDB, and Azure AI context. A monolithic auth solution (e.g., Azure AD) might have fewer moving parts.

## Next Steps

- [ ] **Improve timeline estimation** — Add learning-curve-based duration suggestions based on goal complexity (e.g., "beginner" vs. "advanced")
- [ ] **AI-powered progress insights** — When users mark tasks complete, analyze patterns and suggest plan adjustments
- [ ] **Export and sharing** — Let users export plans as Markdown, PDF, or share read-only links with peers
- [ ] **Mobile app** — React Native or Flutter version for on-the-go progress tracking
- [ ] **Production hardening** — Add request validation, rate limiting, caching, and observability (Application Insights)
- [ ] **Cost optimization** — Cache common goal decompositions and explore GPT-4o mini for simpler parsing tasks

## Built With

- **Frontend**: React 19, TypeScript, Vite, React Router, Clerk
- **Backend**: FastAPI, Python 3.12, Pydantic
- **AI**: Azure AI Projects SDK, GPT-4, Azure AI agents
- **Database**: MongoDB Atlas
- **Authentication**: Clerk
- **Hosting**: (Not yet deployed; ready for Azure Container Apps or App Service)

Built by [Your Name] | [Your GitHub](https://github.com/sarthakkgupta) | [Your LinkedIn](#)

## Additional Links

- [Code](https://github.com/sarthakkgupta/VisualAgent)
- [Issues](https://github.com/sarthakkgupta/VisualAgent/issues)
- [Pull Requests](https://github.com/sarthakkgupta/VisualAgent/pulls)
- [Live Demo](#) (coming soon)
