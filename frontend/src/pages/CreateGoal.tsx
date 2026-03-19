import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useUser, useAuth } from '@clerk/clerk-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import TypeWriter from '../components/TypeWriter'
import type { PlanResponse } from '../types'

// Custom renderer for ReactMarkdown to make links open in a new tab
const LinkRenderer = ({ href, children, ...props }: React.ComponentProps<'a'>) => {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
      {children}
    </a>
  );
};

const formatText = (text: string) => {
  // Replace \n with actual line breaks
  return text.replace(/\\n/g, '\n');
}

// Clean the content by removing unwanted markdown syntax
const cleanContent = (content: string) => {
  if (!content) return '';
  return content.replace(/\*\*/g, '');
};

// Format duration to a more readable format
const formatDuration = (duration: string) => {
  if (!duration) return '';
  
  // Extract number and unit (e.g., "30 days" or "3 weeks")
  const match = duration.match(/(\d+)\s*(day|week|month|hour|minute)s?/i);
  if (!match) return duration;
  
  const [, number, unit] = match;
  const unitLower = unit.toLowerCase();
  
  // Create readable format with emoji
  const unitEmojis: { [key: string]: string } = {
    'day': '📅',
    'week': '📆',
    'month': '📊',
    'hour': '⏰',
    'minute': '⏱'
  };
  
  const emoji = unitEmojis[unitLower] || '⏱';
  return `${emoji} ${number} ${unitLower}${number !== '1' ? 's' : ''}`;
};

interface EditableFieldProps {
  value: string;
  onChange: (value: string) => void;
  isEditing: boolean;
  onEdit: () => void;
  onSave: () => void;
}

const EditableField: React.FC<EditableFieldProps> = ({ value, onChange, isEditing, onEdit, onSave }) => {
  if (isEditing) {
    return (
      <div className="editable-field">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="editable-textarea"
        />
        <button onClick={onSave} className="save-button">
          Save
        </button>
      </div>
    );
  }
  return (
    <div className="editable-field" onClick={onEdit}>
      <ReactMarkdown components={{ a: LinkRenderer }} remarkPlugins={[remarkGfm]}>
        {value}
      </ReactMarkdown>
      <button className="edit-button">Edit</button>
    </div>
  );
};

export default function CreateGoal() {
  const navigate = useNavigate()
  const { isSignedIn, user } = useUser()
  useAuth()
  const [goal, setGoal] = useState('')
  const [includeTimeline, setIncludeTimeline] = useState(false)
  const [loading, setLoading] = useState(false)
  const [plan, setPlan] = useState<PlanResponse | null>(null)
  const [error, setError] = useState('')
  const [showAuthPrompt, setShowAuthPrompt] = useState(false)
  const [editingFields, setEditingFields] = useState<{ [key: string]: boolean }>({})
  const [isInputFocused, setIsInputFocused] = useState(false)

  // Example prompts for the typewriter effect
  const examplePrompts = [
    "Learn Python in 30 days",
    "Roadmap to come up with a startup idea in space tech",
    "Train for a marathon in 3 months",
    "Launch a successful YouTube channel",
    "Master machine learning fundamentals",
    "Create a personal financial plan for retirement",
  ];

  const handleTypeWriterTextChange = () => {
    if (!isInputFocused && !goal) {
      // Only update the placeholder when input is not focused and user hasn't entered text
    }
  };

  const handleEdit = (fieldId: string) => {
    setEditingFields(prev => ({ ...prev, [fieldId]: true }));
  };

  const handleSave = async (fieldId: string, newValue: string, taskIndex?: number) => {
    if (!plan) return;
    
    try {
      const updatedPlan = { ...plan };

      if (fieldId === 'objective') {
        updatedPlan.objective = newValue;
      } else if (taskIndex !== undefined) {
        const [field] = fieldId.split('_');
        updatedPlan.tasks[taskIndex] = {
          ...updatedPlan.tasks[taskIndex],
          [field]: newValue
        };
      }

      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/update-plan`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          goal,
          plan: updatedPlan,
          user_id: user?.id
        })
      });

      if (response.ok) {
        setPlan(updatedPlan);
        setEditingFields(prev => ({ ...prev, [fieldId]: false }));
        // Navigate to the goals page after successful save
        navigate('/goals');
      } else {
        throw new Error('Failed to update plan');
      }
    } catch (err) {
      console.error('Error updating plan:', err);
      setError('Failed to update plan');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!isSignedIn) {
      setShowAuthPrompt(true)
      return
    }

    setLoading(true)
    setError('')
    console.log('Submitting request:', { goal, include_timeline: includeTimeline, user_id: user?.id })

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/plan`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          goal, 
          include_timeline: includeTimeline,
          user_id: user?.id
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to generate plan');
      }

      const data = await response.json()
      console.log('API Response:', data)
      if (data.error) {
        throw new Error(data.error);
      }
      setPlan(data)
    } catch (err) {
      console.error('Error:', err)
      setError(err instanceof Error ? err.message : 'Failed to connect to the server')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="futuristic-container">
      <div className="create-goal-header">
        <h1>Create Your <span className="highlight-text">AI-Powered</span> Plan</h1>
        <p className="subtitle">Describe your goal and our AI will generate a detailed roadmap</p>
      </div>
      
      <form onSubmit={handleSubmit} className="futuristic-card">
        <div className="futuristic-input-container">
          <div className="futuristic-input-wrapper">
            <div className="futuristic-input-icon">✦</div>
            <input
              type="text"
              className="futuristic-input"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              onFocus={() => setIsInputFocused(true)}
              onBlur={() => setIsInputFocused(false)}
              placeholder=""
              required
            />
            {!goal && !isInputFocused && (
              <div className="typewriter-placeholder">
                <TypeWriter 
                  phrases={examplePrompts} 
                  typingSpeed={70} 
                  pauseDuration={2000}
                  onTextChange={handleTypeWriterTextChange}
                />
              </div>
            )}
            <button 
              type="submit" 
              className="futuristic-button pulse"
              disabled={loading}
            >
              {loading ? (
                <div className="loading-animation">
                  <span></span><span></span><span></span>
                </div>
              ) : (
                <>Generate</>
              )}
            </button>
          </div>
          
          <div className="timeline-toggle-container">
            <label className="timeline-switch">
              <input
                type="checkbox"
                checked={includeTimeline}
                onChange={(e) => setIncludeTimeline(e.target.checked)}
              />
              <span className="timeline-slider"></span>
            </label>
            <span className="timeline-label">Include timeline</span>
          </div>
        </div>
      </form>

      {showAuthPrompt && !isSignedIn && (
        <div className="futuristic-card auth-prompt-card">
          <div className="auth-prompt-content">
            <div className="auth-icon">🔒</div>
            <p>Please sign in to generate your plan</p>
            <button className="futuristic-button auth-button">Sign In</button>
          </div>
        </div>
      )}

      {error && (
        <div className="futuristic-card error-card">
          <div className="error-icon">⚠️</div>
          <p>{error}</p>
        </div>
      )}

      {plan && (
        <div className="generated-plan">
          <div className="futuristic-card objective-card">
            <div className="card-header">
              <div className="card-header-icon">🎯</div>
              <h2>Goal</h2>
            </div>
            <EditableField
              value={formatText(plan.objective)}
              onChange={(value) => {
                const updatedPlan = { ...plan, objective: value };
                setPlan(updatedPlan);
              }}
              isEditing={editingFields['objective'] || false}
              onEdit={() => handleEdit('objective')}
              onSave={() => handleSave('objective', plan.objective)}
            />
          </div>

          <div className="tasks-grid">
            {plan.tasks.map((task, index) => (
              <div 
                key={index} 
                className="task-card"
              >
                <div className="task-number">{formatText(cleanContent(task.title))}</div>
                
                {task.duration && (
                  <div className="duration-badge">
                    <span className="duration-icon">⏳</span>
                    <span className="duration-text">{formatDuration(task.duration)}</span>
                  </div>
                )}
                
                <div className="task-section">
                  <EditableField
                    value={cleanContent(task.content)}
                    onChange={(value) => {
                      const updatedTasks = [...plan.tasks];
                      updatedTasks[index] = { ...task, content: value };
                      setPlan({ ...plan, tasks: updatedTasks });
                    }}
                    isEditing={editingFields[`content_${index}`] || false}
                    onEdit={() => handleEdit(`content_${index}`)}
                    onSave={() => handleSave(`content_${index}`, task.content, index)}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}