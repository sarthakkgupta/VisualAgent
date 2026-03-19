import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth, useUser } from '@clerk/clerk-react';
import ReactMarkdown from 'react-markdown';
import { Link } from 'react-router-dom';
import remarkGfm from 'remark-gfm';

// Custom renderer for ReactMarkdown to make links open in a new tab
const LinkRenderer = (props: React.ComponentPropsWithRef<'a'>) => {
  return (
    <a {...props} target="_blank" rel="noopener noreferrer">
      {props.children}
    </a>
  );
};

const formatText = (text: string) => {
  // Replace \n with actual line breaks
  return text.replace(/\\n/g, '\n');
};

// Clean the content by removing unwanted markdown syntax
const cleanContent = (content: string) => {
  if (!content) return '';
  return content.replace(/\*\*/g, '');
};

// Format subtask content by removing "Subtask X" labels and making resources clickable
const formatSubtasks = (content: string) => {
  if (!content) return '';
  
  // First clean asterisks and line breaks
  let formattedContent = cleanContent(formatText(content));
  
  // Replace "Subtask X:" or "[Subtask X]" pattern with the actual content
  formattedContent = formattedContent.replace(/- (Subtask \d+:)/gi, '-');
  formattedContent = formattedContent.replace(/\[(Subtask \d+)\]/gi, '');
  
  // Convert resource links to Markdown format for proper rendering
  // This regex looks for "Resource: Name - URL" pattern and transforms it to Markdown link format
  formattedContent = formattedContent.replace(
    /Resource: (.*?) - (https?:\/\/[^\s\n]+)/gi, 
    'Resource: [$1]($2)'
  );
  
  return formattedContent;
};

interface Task {
  title: string;
  content: string;
  duration: string;
  completed?: boolean; // Add completion status
}

interface Goal {
  _id: string;
  user_id: string;
  query: string;
  result: {
    objective: string;
    tasks: Task[];
  };
  timestamp: string;
}

interface EditableFieldProps {
  value: string;
  onSave: (newValue: string) => void;
  isEditing: boolean;
  onEdit: () => void;
  onCancel: () => void;
}

const EditableField: React.FC<EditableFieldProps> = ({ value, onSave, isEditing, onEdit, onCancel }) => {
  const [draft, setDraft] = useState(value);

  useEffect(() => {
    if (isEditing) setDraft(value);
  }, [isEditing, value]);

  if (isEditing) {
    return (
      <div className="objective-edit-wrap">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          className="objective-textarea"
          autoFocus
        />
        <div className="objective-edit-actions">
          <button onClick={onCancel} className="cancel-button">Cancel</button>
          <button onClick={() => onSave(draft)} className="save-button">Save changes</button>
        </div>
      </div>
    );
  }

  return (
    <div className="objective-view-wrap">
      <div className="objective-markdown">
        <ReactMarkdown components={{ a: LinkRenderer }} remarkPlugins={[remarkGfm]}>
          {value}
        </ReactMarkdown>
      </div>
      <button className="objective-edit-btn" onClick={onEdit}>
        ✎ Edit
      </button>
    </div>
  );
};

// ProgressBar component to visualize task completion
interface ProgressBarProps {
  completed: number;
  total: number;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ completed, total }) => {
  const percentage = total > 0 ? (completed / total) * 100 : 0;

  return (
    <div className="progress-section">
      <div className="progress-info">
        <span className="progress-label-text">Progress</span>
        <span className="progress-count">{completed}/{total} tasks completed</span>
        <span className="progress-percent">{Math.round(percentage)}%</span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${percentage}%` }} />
      </div>
    </div>
  );
};

export default function GoalDetails() {
  const navigate = useNavigate();
  const { goalId } = useParams();
  useAuth();
  const { user, isLoaded } = useUser();
  const [goal, setGoal] = useState<Goal | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [editingFields, setEditingFields] = useState<{ [key: string]: boolean }>({});
  const [notification, setNotification] = useState<{message: string, type: 'success' | 'error'} | null>(null);
  const [progress, setProgress] = useState<{ completed: number; total: number }>({ completed: 0, total: 0 });
  
  // Clear notification after 5 seconds
  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => {
        setNotification(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  // Add a test function that we can access from the window object
  useEffect(() => {
    if (window) {
      // @ts-expect-error Adding a test function to window object for development testing purposes
      window.testEditApi = async (testObjective: string) => {
        if (!goal || !user) {
          console.error('Goal or user not available');
          return;
        }
        
        console.log('Testing API with:');
        console.log('- Goal ID:', goalId);
        console.log('- User ID:', user.id);
        
        try {
          const updatedGoal = { ...goal };
          updatedGoal.result.objective = testObjective || 'This is a test objective update';
          
          console.log('Sending PUT request to:', `${import.meta.env.VITE_API_URL}/api/task/${goalId}?user_id=${user.id}`);
          
          const response = await fetch(`${import.meta.env.VITE_API_URL}/api/task/${goalId}?user_id=${user.id}`, {
            method: 'PUT',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(updatedGoal)
          });
          
          const responseText = await response.text();
          console.log('Response status:', response.status);
          console.log('Response text:', responseText);
          
          if (!response.ok) {
            console.error('API test failed:', responseText);
            return false;
          }
          
          console.log('API test succeeded!');
          return true;
        } catch (err) {
          console.error('API test error:', err);
          return false;
        }
      };
    }
  }, [goal, user, goalId]);

  const handleEdit = (fieldId: string) => {
    setEditingFields(prev => ({ ...prev, [fieldId]: true }));
  };

  const handleSave = async (fieldId: string, newValue: string, taskIndex?: number) => {
    if (!goal || !user) return;
    
    try {
      const updatedGoal = { ...goal };
      console.log(`Saving ${fieldId} with value: ${newValue.substring(0, 30)}...`);
      
      if (fieldId === 'objective') {
        // Important: Update the local state first to get the correct value
        updatedGoal.result.objective = newValue;
        
        console.log('Sending updated goal to API:', JSON.stringify({
          _id: updatedGoal._id,
          user_id: updatedGoal.user_id,
          query: updatedGoal.query,
          result: updatedGoal.result
        }, null, 2));
        
        // Use PUT endpoint for updating the objective
        const response = await fetch(`${import.meta.env.VITE_API_URL}/api/task/${goalId}?user_id=${user.id}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(updatedGoal)
        });

        console.log('Response status:', response.status);
        console.log('Response headers:', Object.fromEntries([...response.headers.entries()]));
        const responseText = await response.text();
        console.log('Response text:', responseText);

        if (!response.ok) {
          console.error('Failed to update goal objective. Response:', responseText);
          setError('Failed to update goal objective: ' + response.status + ' ' + response.statusText);
          setNotification({ message: 'Failed to update goal objective', type: 'error' });
          return;
        }
        
        console.log('Goal objective updated successfully');
        setGoal(updatedGoal);
        setEditingFields(prev => ({ ...prev, [fieldId]: false }));
        setNotification({ message: 'Goal objective updated successfully', type: 'success' });
        
      } else if (taskIndex !== undefined) {
        const [field] = fieldId.split('_');
        
        // Update the local task data
        if (updatedGoal.result.tasks[taskIndex]) {
          updatedGoal.result.tasks[taskIndex] = {
            ...updatedGoal.result.tasks[taskIndex],
            [field]: newValue
          };
        }
        
        console.log(`Updating task ${field} at index ${taskIndex} with:`, newValue.substring(0, 30));
        
        // Try several variations of the request body to find what the API expects
        const requestBody = {
          task_index: taskIndex,
          taskIndex: taskIndex,
          index: taskIndex,
          field_name: field,
          fieldName: field,
          field: field,
          value: newValue,
          content: field === 'content' ? newValue : undefined,
          title: field === 'title' ? newValue : undefined,
          duration: field === 'duration' ? newValue : undefined,
        };
        
        console.log('Sending request body:', JSON.stringify(requestBody, null, 2));
        
        // Use the specialized PATCH endpoint for updating a specific subtask detail
        const response = await fetch(`${import.meta.env.VITE_API_URL}/api/task/${goalId}/detail?user_id=${user.id}`, {
          method: 'PATCH',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(requestBody)
        });

        const responseText = await response.text();
        console.log('PATCH response status:', response.status);
        console.log('PATCH response text:', responseText);

        if (!response.ok) {
          console.error(`Failed to update task ${field}. Response:`, responseText);
          setError(`Failed to update task ${field}`);
          setNotification({ message: `Failed to update task ${field}`, type: 'error' });
          return;
        }
        
        console.log(`Task ${field} updated successfully`);
        setGoal(updatedGoal);
        setEditingFields(prev => ({ ...prev, [fieldId]: false }));
        setNotification({ message: `Task ${field} updated successfully`, type: 'success' });
      }
    } catch (err) {
      console.error('Error updating goal:', err);
      setError('Failed to update goal: ' + (err instanceof Error ? err.message : String(err)));
      setNotification({ message: 'Failed to update goal', type: 'error' });
    }
  };

  const handleDelete = async () => {
    if (!window.confirm('Are you sure you want to delete this goal?')) {
      return;
    }

    try {
      // Using the new DELETE endpoint
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/task/${goalId}?user_id=${user?.id}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        navigate('/goals');
        setNotification({ message: 'Goal deleted successfully', type: 'success' });
      } else {
        const errorData = await response.text();
        console.error('Failed to delete goal:', errorData);
        throw new Error('Failed to delete goal');
      }
    } catch (err) {
      console.error('Error deleting goal:', err);
      setError('Failed to delete goal: ' + (err instanceof Error ? err.message : String(err)));
      setNotification({ message: 'Failed to delete goal', type: 'error' });
    }
  };

  // Function to handle deleting a specific task
  const handleDeleteTask = async (taskIndex: number) => {
    if (!goal || !user) return;
    
    if (!window.confirm(`Are you sure you want to delete this task?`)) {
      return;
    }
    
    try {
      // Create a copy of the goal with the task removed
      const updatedGoal = { ...goal };
      updatedGoal.result.tasks = goal.result.tasks.filter((_, index) => index !== taskIndex);
      
      // Use the PUT endpoint to update the goal with the task removed
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/task/${goalId}?user_id=${user.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(updatedGoal)
      });

      if (response.ok) {
        console.log('Task deleted successfully');
        setGoal(updatedGoal);
        setNotification({ message: 'Task deleted successfully', type: 'success' });
      } else {
        const errorData = await response.text();
        console.error('Failed to delete task:', errorData);
        setError('Failed to delete task');
        setNotification({ message: 'Failed to delete task', type: 'error' });
      }
    } catch (err) {
      console.error('Error deleting task:', err);
      setError('Failed to delete task: ' + (err instanceof Error ? err.message : String(err)));
      setNotification({ message: 'Failed to delete task', type: 'error' });
    }
  };

  // Function to toggle task completion status
  const handleTaskCompletion = async (taskIndex: number, isCompleted: boolean) => {
    if (!goal || !user || !goalId) return;
    
    try {
      // Update UI optimistically
      const updatedGoal = { ...goal };
      const updatedTasks = [...updatedGoal.result.tasks];
      updatedTasks[taskIndex] = { ...updatedTasks[taskIndex], completed: isCompleted };
      updatedGoal.result.tasks = updatedTasks;
      setGoal(updatedGoal);
      
      // Call API to update the completion status
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/task/${goalId}/completion?user_id=${user.id}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          task_index: taskIndex,
          completed: isCompleted
        })
      });

      if (!response.ok) {
        throw new Error('Failed to update task completion status');
      }
      
      // Update progress after successful API call
      fetchProgressInfo();
      
      setNotification({ 
        message: isCompleted ? 'Task marked as completed' : 'Task marked as incomplete', 
        type: 'success' 
      });
    } catch (error) {
      console.error('Error updating task completion:', error);
      // Revert UI changes if API call fails
      if (goal) {
        setGoal({ ...goal });
      }
      setNotification({ 
        message: 'Failed to update task completion status', 
        type: 'error' 
      });
    }
  };
  
  // Function to calculate local progress based on completed tasks
  const calculateLocalProgress = () => {
    if (!goal) return { completed: 0, total: 0 };
    
    const total = goal.result.tasks.length;
    const completed = goal.result.tasks.filter(task => task.completed).length;
    
    return { completed, total };
  };
  
  // Function to fetch progress information
  const fetchProgressInfo = async () => {
    if (!goal || !user || !goalId) return;
    
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/task/${goalId}/progress?user_id=${user.id}`);
      
      if (!response.ok) {
        console.error('Progress API returned error status:', response.status);
        // Fall back to local calculation if API fails
        setProgress(calculateLocalProgress());
        return;
      }
      
      const responseData = await response.json();
      console.log('Progress API response:', responseData);
      
      // Check if the response has the expected structure
      if (responseData && typeof responseData.completed === 'number' && typeof responseData.total === 'number') {
        setProgress(responseData);
      } else {
        console.warn('Progress API returned unexpected structure, using local calculation instead');
        // Use local calculation as fallback
        setProgress(calculateLocalProgress());
      }
    } catch (error) {
      console.error('Error fetching progress information:', error);
      // Use local calculation as fallback
      setProgress(calculateLocalProgress());
    }
  };

  useEffect(() => {
    const fetchGoal = async () => {
      try {
        if (!isLoaded) return;
        
        if (!user) {
          setError('Please sign in to view goal details');
          setLoading(false);
          return;
        }

        console.log('Fetching goal with ID:', goalId);
        console.log('User ID:', user.id);

        const response = await fetch(`${import.meta.env.VITE_API_URL}/api/task/${goalId}?user_id=${user.id}`, {
          headers: {
            'Content-Type': 'application/json'
          }
        });

        if (!response.ok) {
          const errorText = await response.text();
          console.error('Response not ok:', {
            status: response.status,
            statusText: response.statusText,
            body: errorText
          });
          throw new Error(`Failed to fetch goal details: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        console.log('Fetched goal details:', data);
        setGoal(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch goal details');
        setNotification({ message: 'Failed to fetch goal details', type: 'error' });
      } finally {
        setLoading(false);
      }
    };

    fetchGoal();
  }, [goalId, user, isLoaded]);

  // Add progress fetching to initial data loading
  useEffect(() => {
    if (goal && user && goalId) {
      fetchProgressInfo();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [goal, user, goalId]);

  if (!isLoaded) {
    return <div className="loading">Loading...</div>;
  }

  if (!user) {
    return <div className="error-card">Please sign in to view goal details</div>;
  }

  if (loading) {
    return <div className="loading">Loading goal details...</div>;
  }

  if (error || !goal) {
    return (
      <div className="error-container">
        <div className="error-card">{error || 'Goal not found'}</div>
        <Link to="/goals" className="button">Back to Goals</Link>
      </div>
    );
  }

  return (
    <div className="container">
      {notification && (
        <div className={`notification ${notification.type}`}>
          {notification.message}
        </div>
      )}
      <div className="goal-header">
        <div className="goal-header-left">
          <Link to="/goals" className="back-button">← Back to Goals</Link>
          <h1>{goal.query}</h1>
        </div>
        <div className="goal-actions">
          <button 
            onClick={handleDelete} 
            className="delete-goal-button"
            title="Delete this goal"
            aria-label="Delete goal"
          >
            <span className="delete-goal-icon">🗑️</span>
          </button>
        </div>
      </div>

      <div className="card objective-card">
        <h2>Objective</h2>
        <EditableField
          value={formatText(goal.result.objective)}
          onSave={(newValue) => handleSave('objective', newValue)}
          isEditing={editingFields['objective'] || false}
          onEdit={() => handleEdit('objective')}
          onCancel={() => setEditingFields(prev => ({ ...prev, objective: false }))}
        />
      </div>

      <ProgressBar completed={progress.completed} total={progress.total} />

      <div className="tasks-grid">
        {goal.result.tasks.map((task, index) => (
          <div 
            key={index} 
            className={`task-card ${task.completed ? 'completed' : ''}`}
          >
            <div className="task-header">
              <h3 className="task-title">{formatText(cleanContent(task.title))}</h3>
              <div className="task-actions">
                {!editingFields[`title_${index}`] ? (
                  <>
                    <button 
                      className="edit-task-button" 
                      onClick={(e) => {
                        e.stopPropagation();
                        handleEdit(`title_${index}`);
                      }}
                    >
                      Edit
                    </button>
                    <button 
                      className="delete-task-button" 
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteTask(index);
                      }}
                    >
                      <span>🗑️</span>
                    </button>
                  </>
                ) : null}
              </div>
            </div>
            
            {editingFields[`title_${index}`] && (
              <div className="task-edit-form">
                <input
                  value={task.title}
                  onChange={(e) => {
                    if (goal) {
                      const updatedTasks = [...goal.result.tasks];
                      updatedTasks[index] = { ...task, title: e.target.value };
                      setGoal({
                        ...goal,
                        result: { ...goal.result, tasks: updatedTasks }
                      });
                    }
                  }}
                />
                <div className="form-actions">
                  <button 
                    className="save-button" 
                    onClick={(e) => {
                      e.stopPropagation();
                      // Make sure we're using the current state value
                      if (goal) {
                        handleSave(`title_${index}`, goal.result.tasks[index].title, index);
                      }
                    }}
                  >
                    Save
                  </button>
                  <button
                    className="edit-task-button"
                    onClick={() => setEditingFields(prev => ({ ...prev, [`title_${index}`]: false }))}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
            
            {task.duration && (
              <div className="duration-badge">
                <span className="duration-icon">⏱</span>
                <span className="duration-text">{task.duration}</span>
              </div>
            )}
            
            <div className="task-details">
              <div className="task-section">
                <div className="section-header">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                    <h4>Content</h4>
                    {!editingFields[`content_${index}`] && (
                      <button 
                        className="edit-button" 
                        onClick={(e) => {
                          e.stopPropagation();
                          handleEdit(`content_${index}`);
                        }}
                      >
                        <span className="edit-icon">✎</span>
                      </button>
                    )}
                  </div>
                </div>
                
                <div className="task-content">
                  {!editingFields[`content_${index}`] ? (
                    <ReactMarkdown components={{ a: LinkRenderer }} remarkPlugins={[remarkGfm]}>
                      {formatSubtasks(task.content)}
                    </ReactMarkdown>
                  ) : (
                    <div className="editable-field">
                      <textarea
                        value={formatSubtasks(task.content)}
                        onChange={(e) => {
                          if (goal) {
                            const updatedTasks = [...goal.result.tasks];
                            updatedTasks[index] = { ...task, content: e.target.value };
                            setGoal({
                              ...goal,
                              result: { ...goal.result, tasks: updatedTasks }
                            });
                          }
                        }}
                        className="editable-textarea"
                      />
                      <div className="subtask-actions">
                        <button 
                          className="save-button" 
                          onClick={(e) => {
                            e.stopPropagation();
                            // Make sure we're using the current state value
                            if (goal) {
                              handleSave(`content_${index}`, goal.result.tasks[index].content, index);
                            }
                          }}
                        >
                          Save
                        </button>
                        <button
                          className="edit-task-button"
                          onClick={() => setEditingFields(prev => ({ ...prev, [`content_${index}`]: false }))}
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="task-completion">
              <label>
                <input
                  type="checkbox"
                  checked={task.completed || false}
                  onChange={(e) => handleTaskCompletion(index, e.target.checked)}
                />
                Mark as Completed
              </label>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}