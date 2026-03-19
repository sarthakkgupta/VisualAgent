import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useUser } from '@clerk/clerk-react';

interface Task {
  title: string;
  completed?: boolean;
}

interface Plan {
  _id: string;
  query: string;
  timestamp: string;
  result: {
    objective: string;
    tasks: Task[];
  };
}

const formatDate = (timestamp: string) => {
  const date = new Date(timestamp);
  return new Intl.DateTimeFormat('en-US', { 
    year: 'numeric', 
    month: 'short',
    day: 'numeric' 
  }).format(date);
};

const calculateProgress = (tasks: Task[]) => {
  if (!tasks || tasks.length === 0) return 0;
  const completedTasks = tasks.filter(task => task.completed).length;
  return Math.round((completedTasks / tasks.length) * 100);
};

export default function DashboardPage() {
  const { user, isLoaded } = useUser();
  const [recentPlans, setRecentPlans] = useState<Plan[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [stats, setStats] = useState({
    totalPlans: 0,
    completedTasks: 0,
    totalTasks: 0,
    planProgress: 0
  });

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        if (!isLoaded || !user) {
          setLoading(false);
          return;
        }

        // Fetch recent plans
        const response = await fetch(`${import.meta.env.VITE_API_URL}/api/history?user_id=${user.id}`, {
          headers: {
            'Content-Type': 'application/json'
          }
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch plans: ${response.status}`);
        }

        const data = await response.json();
        
        // Get the most recent plans (up to 5)
        const recent = data.history.slice(0, 5);
        setRecentPlans(recent);
        
        // Calculate statistics
        let totalCompleted = 0;
        let allTasks = 0;
        
        data.history.forEach((plan: Plan) => {
          const planTasks = plan.result.tasks || [];
          const completed = planTasks.filter(task => task.completed).length;
          totalCompleted += completed;
          allTasks += planTasks.length;
        });
        
        setStats({
          totalPlans: data.history.length,
          completedTasks: totalCompleted,
          totalTasks: allTasks,
          planProgress: allTasks > 0 ? Math.round((totalCompleted / allTasks) * 100) : 0
        });
        
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to load dashboard data');
      } finally {
        setLoading(false);
      }
    };

    fetchUserData();
  }, [user, isLoaded]);

  if (!isLoaded) {
    return <div className="loading">Loading...</div>;
  }

  if (!user) {
    return <div className="error-card">Please sign in to view your dashboard</div>;
  }

  if (loading) {
    return <div className="loading">Loading your dashboard...</div>;
  }

  if (error) {
    return <div className="error-card">{error}</div>;
  }

  return (
    <div className="dashboard-page">
      <div className="dashboard-header">
        <h1>Welcome to Your Dashboard, {user.firstName || user.username}</h1>
        <p className="dashboard-subtitle">Track your plans and progress at a glance</p>
      </div>

      <div className="dashboard-grid">
        <div className="dashboard-main">
          <section className="stat-cards">
            <div className="stat-card">
              <h3>Total Plans</h3>
              <div className="stat-value">{stats.totalPlans}</div>
            </div>
            <div className="stat-card">
              <h3>Tasks Completed</h3>
              <div className="stat-value">{stats.completedTasks}/{stats.totalTasks}</div>
            </div>
            <div className="stat-card">
              <h3>Overall Progress</h3>
              <div className="stat-value">{stats.planProgress}%</div>
              <div className="progress-bar-track">
                <div 
                  className="progress-bar-fill" 
                  style={{ width: `${stats.planProgress}%` }}
                ></div>
              </div>
            </div>
          </section>

          <section className="recent-plans-section">
            <div className="section-header">
              <h2>Recent Plans</h2>
              <Link to="/goals" className="see-all-link">See All</Link>
            </div>
            
            {recentPlans.length > 0 ? (
              <div className="recent-plans">
                {recentPlans.map((plan) => (
                  <div key={plan._id} className="plan-card">
                    <h3 className="plan-title">{plan.query}</h3>
                    <div className="plan-meta">
                      <div className="plan-date">{formatDate(plan.timestamp)}</div>
                      <div className="plan-progress">
                        <div className="progress-text">
                          {calculateProgress(plan.result.tasks)}% Complete
                        </div>
                        <div className="progress-bar-track">
                          <div 
                            className="progress-bar-fill" 
                            style={{ width: `${calculateProgress(plan.result.tasks)}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                    <div className="plan-actions">
                      <Link to={`/goals/${plan._id}`} className="view-details-button">
                        View Details →
                      </Link>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-plans">
                <p>You haven't created any plans yet.</p>
                <Link to="/create" className="primary-button">Create Your First Plan</Link>
              </div>
            )}
          </section>
        </div>
        
        <div className="dashboard-sidebar">
          <section className="quick-actions">
            <h2>Quick Actions</h2>
            <div className="action-buttons">
              <Link to="/create" className="action-button">
                <div className="action-icon">➕</div>
                <div className="action-text">Create New Plan</div>
              </Link>
              <Link to="/goals" className="action-button">
                <div className="action-icon">📋</div>
                <div className="action-text">View All Plans</div>
              </Link>
            </div>
          </section>

          <section className="tips-section">
            <h2>Tips & Insights</h2>
            <div className="tip-card">
              <h3>Plan Breaking Tip</h3>
              <p>Break down large goals into manageable tasks to make steady progress over time.</p>
            </div>
            <div className="tip-card">
              <h3>Timeline Magic</h3>
              <p>Enable timeline features to get a better sense of how long your goals will take to complete.</p>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}