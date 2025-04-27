- Structure your project folders
- Set up a base FastAPI server
- Include placeholder Semantic Kernel agent functions
- Be extendable as you build more agents

---

### 📦 **Prompt to Scaffold Backend Project:**

```text
I want to create a backend for an AI project using Python.

Use:
- FastAPI for APIs
- Semantic Kernel for multi-agent orchestration
- Azure OpenAI or OpenAI API for LLMs
- Dotenv for environment variables
- Async support

Create a project scaffold with:
1. `main.py` to run the FastAPI app
2. `sk_config/kernel.py` to initialize Semantic Kernel
3. A `plugins/` folder inside `sk_config/` with 4 agents:
   - goal_interpreter
   - task_breakdown
   - resource_finder
   - scheduler
4. Each plugin should have a basic prompt-based semantic function
5. A sample POST `/api/plan` endpoint in `main.py` that uses all agents in sequence
6. Return a combined response like:
```json
{
  "objective": "Learn Data Structures",
  "tasks": ["Week 1: Arrays", "Week 2: Trees"],
  "schedule": {
    "Day 1": ["Read about Arrays"],
    "Day 2": ["Practice problems"]
  },
  "resources": [
    {"title": "Arrays Explained", "url": "https://youtube.com/..."}
  ]
}
```
7. Create `.env` support for Azure/OpenAI API keys

Structure folders like this:
```
backend/
├── main.py
├── .env
├── requirements.txt
└── sk_config/
    ├── kernel.py
    └── plugins/
        ├── goal_interpreter/
        ├── task_breakdown/
        ├── scheduler/
        └── resource_finder/
```

Also include a `requirements.txt` file with all the dependencies.

Finally, print a log message in `/api/plan` when the plan is generated successfully.
```

---

## ✅ Next Step After Scaffold

Once this is generated, we can:
- Add actual prompts in `plugins`
- Connect to OpenAI or Azure OpenAI
- Test it with Postman or simple React frontend

