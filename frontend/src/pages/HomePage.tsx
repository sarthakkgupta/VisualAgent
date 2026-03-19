import { Link } from 'react-router-dom';
import { useUser } from '@clerk/clerk-react';

export default function HomePage() {
  const { isSignedIn } = useUser();

  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <h1>Transform Your Ideas into Actionable Plans with AI</h1>
          <p className="hero-subtitle">
            VisualAI uses AI to create personalized, detailed plans for your goals,
            helping you achieve more with less effort.
          </p>
          <div className="hero-cta">
            {isSignedIn ? (
              <Link to="/create" className="primary-button">Create Your Plan</Link>
            ) : (
              <Link to="/create" className="primary-button">Try VisualAI Free</Link>
            )}
            <Link to="/pricing" className="secondary-button">View Pricing</Link>
          </div>
        </div>
        <div className="hero-image">
          <div className="abstract-shape"></div>
          <img src="/images/hero-visual.svg" alt="VisualAI Plan Creation" />
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <h2>How VisualAI Works</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">📝</div>
            <h3>Describe Your Goal</h3>
            <p>Simply describe what you want to achieve in plain language. Our AI understands your intent.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🧠</div>
            <h3>AI-Generated Plan</h3>
            <p>Our algorithms create a detailed, step-by-step plan customized to your specific goal.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Track Your Progress</h3>
            <p>Monitor your advancement with visual indicators as you complete tasks on your journey.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🔄</div>
            <h3>Adapt & Refine</h3>
            <p>Easily edit tasks, mark completions, and adjust your plan as you progress.</p>
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="use-cases-section">
        <h2>What Can You Plan With VisualAI?</h2>
        <div className="use-cases-grid">
          <div className="use-case">
            <h3>Learning & Education</h3>
            <p>Create structured learning paths for any subject, from programming to playing guitar.</p>
          </div>
          <div className="use-case">
            <h3>Professional Development</h3>
            <p>Map your career growth with actionable steps to acquire needed skills and certifications.</p>
          </div>
          <div className="use-case">
            <h3>Project Management</h3>
            <p>Break down complex projects into manageable tasks with clear timelines.</p>
          </div>
          <div className="use-case">
            <h3>Personal Goals</h3>
            <p>Whether fitness, finance, or personal growth - visualize your path to success.</p>
          </div>
        </div>
      </section>

      {/* Testimonial Section */}
      <section className="testimonial-section">
        <h2>What Our Users Say</h2>
        <div className="testimonials">
          <div className="testimonial-card">
            <div className="quote">"VisualAI helped me break down my career transition into manageable steps. I went from feeling overwhelmed to having a clear roadmap."</div>
            <div className="author">
              <div className="author-info">
                <div className="author-name">Alex Johnson</div>
                <div className="author-title">Marketing to UX Designer</div>
              </div>
            </div>
          </div>
          <div className="testimonial-card">
            <div className="quote">"As a manager, I use VisualAI to create onboarding plans for new team members. It's saved me hours of planning time."</div>
            <div className="author">
              <div className="author-info">
                <div className="author-name">Maya Patel</div>
                <div className="author-title">Team Lead, InnoTech</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="cta-content">
          <h2>Start Achieving Your Goals Today</h2>
          <p>Join thousands of users who have transformed their ideas into reality with VisualAI.</p>
          <div className="cta-buttons">
            {isSignedIn ? (
              <Link to="/create" className="primary-button">Create Your Plan</Link>
            ) : (
              <Link to="/create" className="primary-button">Get Started for Free</Link>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}