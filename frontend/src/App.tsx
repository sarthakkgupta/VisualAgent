import { Routes, Route, Link, Navigate, useLocation } from 'react-router-dom'
import { useUser, SignInButton, UserButton } from '@clerk/clerk-react'
import MyGoals from './pages/MyGoals'
import GoalDetails from './pages/GoalDetails'
import CreateGoal from './pages/CreateGoal'
import HomePage from './pages/HomePage'
import PricingPage from './pages/PricingPage'
import AboutPage from './pages/AboutPage'
import DashboardPage from './pages/DashboardPage'
import './App.css'

export default function App() {
  const { isSignedIn, user } = useUser()
  const location = useLocation()
  
  // Helper function to determine if a path is active
  const isActive = (path: string) => {
    // Handle both exact matches and sub-paths
    if (path === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(path);
  };

  return (
    <div className="app-container">
      <header className="main-header">
        <div className="logo-container">
          <Link to="/" className="logo-link">
            <span className="logo-icon">✦</span>
            <span className="logo-text">VisualAI</span>
          </Link>
        </div>
        
        <nav className="main-nav">
          <Link to="/about" className={`nav-link ${isActive('/about') ? 'active' : ''}`}>
            About
          </Link>
          <Link to="/pricing" className={`nav-link ${isActive('/pricing') ? 'active' : ''}`}>
            Pricing
          </Link>
          {isSignedIn && (
            <>
              <Link to="/dashboard" className={`nav-link ${isActive('/dashboard') ? 'active' : ''}`}>
                Dashboard
              </Link>
              <Link to="/goals" className={`nav-link ${isActive('/goals') ? 'active' : ''}`}>
                My Plans
              </Link>
              <Link to="/create" className={`nav-link highlight ${isActive('/create') ? 'active' : ''}`}>
                Create Plan
              </Link>
            </>
          )}
        </nav>
        
        <div className="auth-section">
          {!isSignedIn ? (
            <div className="auth-buttons">
              <SignInButton mode="modal">
                <button className="login-button">Sign In</button>
              </SignInButton>
              <SignInButton mode="modal">
                <button className="signup-button">Try Free</button>
              </SignInButton>
            </div>
          ) : (
            <div className="user-section">
              <span className="welcome-text">Welcome, {user.firstName || user.username}</span>
              <UserButton />
            </div>
          )}
        </div>
      </header>

      <main className="main-content">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/pricing" element={<PricingPage />} />
          <Route path="/dashboard" element={isSignedIn ? <DashboardPage /> : <Navigate to="/" />} />
          <Route path="/goals" element={isSignedIn ? <MyGoals /> : <Navigate to="/" />} />
          <Route path="/goals/:goalId" element={isSignedIn ? <GoalDetails /> : <Navigate to="/" />} />
          <Route path="/create" element={isSignedIn ? <CreateGoal /> : <Navigate to="/" />} />
        </Routes>
      </main>
      
      <footer className="main-footer">
        <div className="footer-content">
          <div className="footer-section">
            <h4>VisualAI</h4>
            <p>Transforming ideas into actionable plans with AI</p>
          </div>
          <div className="footer-section">
            <h4>Links</h4>
            <Link to="/about">About</Link>
            <Link to="/pricing">Pricing</Link>
            <a href="#">Terms of Service</a>
            <a href="#">Privacy Policy</a>
          </div>
          <div className="footer-section">
            <h4>Contact</h4>
            <p>hello@visualai.com</p>
            <div className="social-icons">
              <a href="#" aria-label="Twitter">𝕏</a>
              <a href="#" aria-label="LinkedIn">in</a>
              <a href="#" aria-label="GitHub">⌥</a>
            </div>
          </div>
        </div>
        <div className="copyright">
          © {new Date().getFullYear()} VisualAI. All rights reserved.
        </div>
      </footer>
    </div>
  )
}
