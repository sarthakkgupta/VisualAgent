# Basic Python AI Agent Project

# VisualAgent Backend

This project is a backend for an AI project using FastAPI and Semantic Kernel. It is designed to orchestrate multiple agents and interact with OpenAI or Azure OpenAI APIs.

## Prerequisites

1. **Python**: Ensure you have Python 3.9 or later installed.
2. **Environment Variables**: Create a `.env` file in the `backend/` directory with the following keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   AZURE_API_KEY=your_azure_api_key
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd VisualAgent/backend
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate   # On Windows
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

2. The server will be available at `http://127.0.0.1:8000`.

## Testing the API

1. Use a tool like Postman or cURL to test the `/api/plan` endpoint.
2. Example request:
   ```bash
   curl -X POST "http://127.0.0.1:8000/api/plan" -H "Content-Type: application/json" -d '{"objective": "Learn Data Structures"}'
   ```
3. Example response:
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

## Project Structure

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

## Next Steps

- Add actual prompts in `plugins`.
- Connect to OpenAI or Azure OpenAI.
- Test with a frontend or Postman.
