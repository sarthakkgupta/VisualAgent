import { useState, useEffect } from 'react';
import { useAuth, useUser } from '@clerk/clerk-react';
import { Link } from 'react-router-dom';

const cleanContent = (content: string) => {
  if (!content) return '';
  return content.replace(/\*\*/g, '').replace(/\\n/g, ' ').trim();
};

const formatDate = (timestamp: string) => {
  const date = new Date(timestamp);
  
  // Get day, month and year
  const day = date.getDate();
  const month = date.toLocaleString('default', { month: 'short' });
  const year = date.getFullYear();
  
  // Add ordinal suffix to day
  let suffix = "th";
  if (day === 1 || day === 21 || day === 31) {
    suffix = "st";
  } else if (day === 2 || day === 22) {
    suffix = "nd";
  } else if (day === 3 || day === 23) {
    suffix = "rd";
  }
  
  // Return formatted date
  return `${day}${suffix} ${month} ${year}`;
};

interface Task {
  title: string;
  content: string;
  duration: string;
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

export default function MyGoals() {
  const { getToken } = useAuth();
  const { user, isLoaded } = useUser();
  const [goals, setGoals] = useState<Goal[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortOrder, setSortOrder] = useState('newest');

  const handleDelete = async (goalId: string) => {
    if (!window.confirm('Are you sure you want to delete this plan?')) {
      return;
    }

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/task/${goalId}?user_id=${user?.id}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        setGoals(goals.filter(goal => goal._id !== goalId));
      } else {
        throw new Error('Failed to delete plan');
      }
    } catch (err) {
      console.error('Error deleting plan:', err);
      setError('Failed to delete plan');
    }
  };

  useEffect(() => {
    const fetchGoals = async () => {
      try {
        if (!isLoaded) return;

        if (!user) {
          setError('Please sign in to view your plans');
          setLoading(false);
          return;
        }

        console.log('Fetching plans for user:', user.id);

        const response = await fetch(`${import.meta.env.VITE_API_URL}/api/history?user_id=${user.id}`, {
          headers: {
            'Content-Type': 'application/json'
          }
        });

        if (!response.ok) {
          const errorData = await response.text();
          console.error('API Error:', errorData);
          throw new Error(`Failed to fetch plans: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        console.log('Received plans:', data);
        setGoals(data.history);
      } catch (err) {
        console.error('Error fetching plans:', err);
        setError('Failed to fetch plans');
      } finally {
        setLoading(false);
      }
    };

    fetchGoals();
  }, [getToken, user, isLoaded]);

  // Filter and sort goals
  const filteredGoals = goals
    .filter(goal => {
      // Apply search filter if exists
      if (searchQuery) {
        return goal.query.toLowerCase().includes(searchQuery.toLowerCase());
      }
      return true;
    })
    .sort((a, b) => {
      // Apply sorting
      if (sortOrder === 'newest') {
        return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
      } else {
        return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
      }
    });

  const getStatusColor = (goal: Goal) => {
    // This is a placeholder logic. In a real app, you might have some status tracking
    // Here we're just assigning colors based on the timestamp
    const timestamp = new Date(goal.timestamp).getTime();
    const now = new Date().getTime();
    const daysDifference = Math.floor((now - timestamp) / (1000 * 60 * 60 * 24));
    
    if (daysDifference < 3) return 'status-new';
    if (daysDifference < 7) return 'status-in-progress';
    return 'status-completed';
  };

  if (!isLoaded) {
    return (
      <div className="futuristic-loading">
        <div className="futuristic-spinner">
          <div className="spinner-ring"></div>
          <span>Loading</span>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="futuristic-container">
        <div className="futuristic-card auth-prompt-card">
          <div className="auth-prompt-content">
            <div className="auth-icon">🔒</div>
            <h2>Authentication Required</h2>
            <p>Please sign in to view your plans and progress</p>
            <Link to="/" className="futuristic-button auth-button">Sign In</Link>
          </div>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="futuristic-loading">
        <div className="futuristic-spinner">
          <div className="spinner-ring"></div>
          <span>Loading your plans</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="futuristic-container">
        <div className="futuristic-card error-card">
          <div className="error-icon">⚠️</div>
          <p>{error}</p>
          <button onClick={() => window.location.reload()} className="futuristic-button">
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="futuristic-container my-plans-page">
      <div className="plans-header">
        <div className="header-content">
          <h1>My <span className="highlight-text">Plans</span></h1>
          <p className="subtitle">Track and manage your AI-generated plans</p>
        </div>
        <Link to="/create" className="futuristic-button pulse">
          <span className="button-icon">+</span>
          Create New Plan
        </Link>
      </div>

      <div className="plans-controls">
        <div className="search-container">
          <div className="futuristic-search-wrapper">
            <span className="search-icon">🔍</span>
            <input
              type="text"
              placeholder="Search your plans..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="futuristic-search"
            />
            {searchQuery && (
              <button 
                className="clear-search"
                onClick={() => setSearchQuery('')}
              >
                ×
              </button>
            )}
          </div>
        </div>
        
        <div className="filter-options">
          <div className="filter-option">
            <label>Sort by:</label>
            <select 
              value={sortOrder}
              onChange={(e) => setSortOrder(e.target.value)}
              className="futuristic-select"
            >
              <option value="newest">Newest first</option>
              <option value="oldest">Oldest first</option>
            </select>
          </div>
        </div>
      </div>

      {filteredGoals.length === 0 && searchQuery && (
        <div className="no-results">
          <div className="no-results-icon">🔎</div>
          <h3>No matching plans found</h3>
          <p>Try adjusting your search or filters</p>
        </div>
      )}

      {filteredGoals.length === 0 && !searchQuery && (
        <div className="empty-plans">
          <div className="empty-plans-illustration">
            <div className="illustration-circle"></div>
            <div className="illustration-line"></div>
            <div className="illustration-line"></div>
            <div className="illustration-line"></div>
          </div>
          <h3>No plans yet</h3>
          <p>Create your first AI-powered plan to get started</p>
          <Link to="/create" className="futuristic-button">Create Your First Plan</Link>
        </div>
      )}

      <div className="plans-grid">
        {filteredGoals.map((goal) => (
          <div key={goal._id} className={`plan-card ${getStatusColor(goal)}`}>
            <div className="plan-card-content">
              <div className="plan-date">
                <span className="date-display">{formatDate(goal.timestamp)}</span>
              </div>
              
              <h2 className="plan-title">{goal.query}</h2>
              
              <div className="plan-stats">
                <div className="stat">
                  <span className="stat-value">{goal.result.tasks.length}</span>
                  <span className="stat-label">Tasks</span>
                </div>
                {goal.result.tasks.some(task => task.duration) && (
                  <div className="stat">
                    <span className="stat-value">
                      {goal.result.tasks.reduce((total, task) => {
                        const durationMatch = task.duration?.match(/(\d+)/);
                        return total + (durationMatch ? parseInt(durationMatch[0]) : 0);
                      }, 0)}
                    </span>
                    <span className="stat-label">Days</span>
                  </div>
                )}
              </div>
              
              <div className="plan-preview">
                <ul className="task-list">
                  {goal.result.tasks.slice(0, 4).map((task, idx) => (
                    <li key={idx} className="task-item">
                      <span className="task-checkbox" />
                      <span className="task-item-title">{cleanContent(task.title)}</span>
                    </li>
                  ))}
                  {goal.result.tasks.length > 4 && (
                    <li className="task-more">+{goal.result.tasks.length - 4} more tasks</li>
                  )}
                </ul>
              </div>
            </div>

            <div className="plan-card-actions">
              <Link to={`/goals/${goal._id}`} className="view-button">
                View Plan
              </Link>
              <button 
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  handleDelete(goal._id);
                }} 
                className="delete-button"
                aria-label="Delete plan"
              >
                <span className="delete-icon">🗑</span>
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}