# VisualAgent Prompt and Semantic Flow Documentation

This document provides a comprehensive overview of the VisualAgent's prompt engineering and semantic flow between different components.

## System Overview

VisualAgent is a learning plan generation system that helps users create structured learning plans for any topic or skill. The system:

1. Takes a goal input from the user (e.g., "Learn Python in 2 weeks")
2. Interprets the objective and extracts key information (including duration)
3. Breaks down the goal into high-level tasks with embedded resources
4. Generates appropriate timelines for each task
5. Returns a comprehensive plan that fits within the specified duration

## Plugin Architecture

The system uses three main semantic plugins:

1. **Goal Interpreter** - Understands the user's objective and extracts key metadata
2. **Task Breakdown** - Divides the goal into actionable tasks with integrated resources
3. **Timeline Generator** - Creates appropriate durations for tasks that collectively fit within the total timeframe

## Prompt Flow Sequence

### 1. Goal Interpreter

**Input**: Raw user goal (e.g., "Learn Python in 2 weeks")

**Prompt Design**:
```
Extract the following from this input:
1. Objective (what does the user want to do?)
2. Duration in days/weeks if mentioned

User input: {{$input}}
---
Return as: "Objective: ...\nDuration: ..."
```

**Output Example**: 
```
Objective: Learn Python
Duration: 2 weeks
```

### 2. Task Breakdown

**Input**: Interpreted objective with duration information

**Prompt Design**:
```
Take the objective and break it down into appropriate tasks that MUST collectively fit within the specified duration.

IMPORTANT DURATION INSTRUCTIONS:
- If the duration is short (e.g., 3 days or less), create just 2-3 focused tasks.
- If the duration is medium (e.g., 1-2 weeks), create 3-4 balanced tasks.
- If the duration is longer (e.g., 3+ weeks), create 4-6 comprehensive tasks.
- Ensure tasks are sized appropriately for the timeframe - shorter durations require more concise tasks.

For each task, include:
1. A clear high-level task name/description
2. 3-5 specific actionable subtasks with naturally integrated resources

Objective: {{$input}}
---
TASK 1: [High-level task name/description]
...
```

**Output Example**:
```
TASK 1: Complete an introductory Python course
- Install Python and set up a development environment by following the official Python installation guide at python.org
- Learn basic syntax and data types by completing the Python for Beginners course on Codecademy
- Practice control flow statements by working through the exercises in "Automate the Boring Stuff with Python"

TASK 2: Practice with basic Python exercises
...
```

### 3. Timeline Generator

**Input**: A specific task with context about the overall plan duration and previously allocated time

**Prompt Design**:
```
Provide a simple time duration estimate for completing this task.

IMPORTANT TIMELINE CONSTRAINTS:
- Your duration MUST respect the overall plan duration
- Express durations in days rather than weeks whenever possible
- Ensure your estimate allows ALL tasks to fit within the total timeframe
- If this task is part of a short total duration (e.g., 3 days), keep the estimate very concise (1-2 days)

Task: {{$input}}

Just return a clear and concise duration estimate like "1 day", "3 days", "5 days", etc. 
ONLY use weeks as the unit if the task genuinely requires more than 10 days.
No additional explanation needed.
```

**Output Example**:
```
5 days
```

## Duration Management Logic

The system now uses a sophisticated approach to ensure that all tasks collectively fit within the user's specified duration:

1. **Duration Categorization**: The system identifies if the duration is short (≤3 days), medium (1-2 weeks), or long (3+ weeks).

2. **Appropriate Task Quantity**: For shorter durations, fewer tasks (2-3) are created; for longer durations, more tasks (4-6) may be created.

3. **Initial Allocation Algorithm**: The system pre-allocates days to each task based on:
   - Total available days
   - Number of tasks
   - Position within the sequence (e.g., first task may get more time for setup)
   - The algorithm ensures initial allocations sum exactly to the total duration

4. **Contextual Timeline Generation**: Each task's timeline is created with full awareness of:
   - How many days have already been allocated to previous tasks
   - How many tasks remain
   - The exact constraint of the total duration

5. **Continuous Tracking**: The system maintains a running total of allocated days and adjusts remaining tasks accordingly.

6. **Final Validation**: A verification step ensures the sum of all task durations doesn't exceed the total specified duration, with proportional adjustments made if necessary.

## Backend Processing Logic

The main.py file orchestrates the flow between plugins with the following enhanced logic:

1. Parse user input and extract user_id
2. Invoke goal_interpreter to understand the objective and extract duration
3. Categorize the duration as short/medium/long and pass appropriate context to task_breakdown
4. Parse the task_breakdown response to extract individual tasks
5. Pre-allocate days across tasks to ensure they sum to exactly the total duration
6. For each task:
   - Calculate appropriate time allocation based on pre-allocated days
   - Invoke timeline_generator with detailed context about previous allocations
   - Enforce constraints to ensure tasks don't exceed available time
   - Add the duration to the task object
7. Perform a final check to ensure collective durations don't exceed the total
8. Return the complete plan to the user
9. Store the plan in MongoDB for later retrieval

## API Endpoints

- **POST /api/plan** - Generate a new learning plan
- **GET /api/history** - Retrieve a user's plan history
- **GET /api/task/{task_id}** - Retrieve a specific task's details
- **PUT /api/task/{task_id}** - Update an entire task
- **PATCH /api/task/{task_id}/detail** - Update specific details of a task
- **DELETE /api/task/{task_id}** - Delete a task

## Recent Updates

- **Improved Duration-Based Task Generation**: System now creates an appropriate number of tasks based on the available duration (fewer tasks for shorter timeframes)
- **Enhanced Timeline Constraints**: Timeline generator now respects strict duration bounds with explicit instructions about fitting within the total timeframe
- **Sophisticated Allocation Algorithm**: New algorithm pre-allocates days to tasks and ensures their sum exactly matches the total duration
- **Continuous Monitoring**: System now tracks running totals and adjusts remaining allocations to stay within bounds
- **Final Validation Check**: Added a verification step to proportionally adjust durations if they exceed the total
- **Detailed Logging**: Enhanced logging for better tracking of duration allocations and adjustments

*Last updated: April 27, 2025*