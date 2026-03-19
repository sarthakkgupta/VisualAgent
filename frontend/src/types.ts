export interface TaskDetail {
  title: string;
  content: string;
  duration: string;
}

export interface PlanResponse {
  objective: string;
  tasks: TaskDetail[];
}