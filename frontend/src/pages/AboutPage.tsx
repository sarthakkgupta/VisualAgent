import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';

export default function AboutPage() {
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('animate-in');
          }
        });
      },
      { threshold: 0.1, rootMargin: '0px 0px -50px 0px' }
    );

    const elements = document.querySelectorAll('.about-animate');
    elements.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  return (
    <div className="about-page">

      {/* ── Hero ── */}
      <section className="about-hero">
        <div className="about-hero-bg" />
        <div className="about-hero-content">
          <div className="about-hero-badge">✦ Powered by AI</div>
          <h1>About <span className="about-highlight">VisualAI</span></h1>
          <p className="about-subtitle">
            Making goal achievement more accessible through advanced AI planning technologies.
          </p>

          {/* Orbit visualization */}
          <div className="about-hero-visual">
            <div className="goal-orbit">
              <div className="goal-core">🎯</div>
              <div className="orbit-ring orbit-1"><div className="orbit-dot" /></div>
              <div className="orbit-ring orbit-2"><div className="orbit-dot" /></div>
              <div className="orbit-ring orbit-3"><div className="orbit-dot" /></div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Problem Statement ── */}
      <section className="about-problem">
        <div className="about-container">
          <span className="about-section-label about-animate">01 — Problem Statement</span>
          <h2 className="about-section-h2 about-animate">Most people know <em>what</em> they want.<br />Few know <em>how</em> to get there.</h2>
          <p className="about-section-intro about-animate">
            Goal-setting tools give you a place to write things down. But between writing a goal
            and actually achieving it, there's a massive, unsupported gap.
          </p>

          <div className="pain-grid">
            {[
              { icon: '🧭', title: 'No Clear Starting Point', text: 'People know where they want to end up but have no idea what the very first step should be.' },
              { icon: '⏳', title: 'No Time to Plan', text: 'Creating a structured, realistic plan takes hours of research, thinking, and organization most people simply can\'t spare.' },
              { icon: '📚', title: 'Resource Overload', text: 'Google returns thousands of articles. YouTube has hundreds of courses. None of it is personalized to your specific situation.' },
              { icon: '📉', title: 'Unrealistic Timelines', text: 'Without domain expertise, people underestimate complexity and set timelines that doom them to failure from day one.' },
            ].map((pain, i) => (
              <div
                key={pain.title}
                className="pain-card about-animate"
                style={{ '--delay': `${i * 0.1}s` } as React.CSSProperties}
              >
                <div className="pain-icon">{pain.icon}</div>
                <h3>{pain.title}</h3>
                <p>{pain.text}</p>
              </div>
            ))}
          </div>

          <div className="problem-stats-row">
            <div className="problem-stat about-animate" style={{ '--delay': '0.1s' } as React.CSSProperties}>
              <div className="pstat-number">92%</div>
              <div className="pstat-label">of people fail to achieve their New Year's resolutions</div>
            </div>
            <div className="problem-stat about-animate" style={{ '--delay': '0.2s' } as React.CSSProperties}>
              <div className="pstat-number">4.3h</div>
              <div className="pstat-label">average time spent planning a goal before giving up</div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Customer Segments ── */}
      <section className="about-segments">
        <div className="about-container">
          <span className="about-section-label about-animate">02 — Who This Is For</span>
          <h2 className="about-section-h2 about-animate">Built for people with goals,<br />not just productivity nerds.</h2>

          <div className="segments-grid">
            {[
              { icon: '🎓', role: 'College Student', goal: 'Land a software internship', struggle: 'Doesn\'t know what to learn first — frameworks, DSA, projects?', outcome: 'Gets a week-by-week learning roadmap built around their schedule.' },
              { icon: '💼', role: 'PM / Team Lead', goal: 'Launch a product feature in Q2', struggle: 'Needs to coordinate across eng, design, and stakeholders with no clear plan.', outcome: 'Gets a cross-functional plan with milestones, dependencies, and deadlines.' },
              { icon: '🔄', role: 'Career Switcher', goal: 'Move from finance to UX design', struggle: 'Overwhelmed by conflicting advice on courses, portfolios, and timelines.', outcome: 'Gets a 90-day transition roadmap tailored to their background.' },
              { icon: '🚀', role: 'Entrepreneur', goal: 'Validate a SaaS idea in 60 days', struggle: 'Knows the end goal but not the sequential steps to get from zero to customers.', outcome: 'Gets a structured path from idea to MVP with weekly checkpoints.' },
            ].map((seg, i) => (
              <div
                key={seg.role}
                className="segment-card about-animate"
                style={{ '--delay': `${i * 0.1}s` } as React.CSSProperties}
              >
                <div className="segment-icon">{seg.icon}</div>
                <div className="segment-role">{seg.role}</div>
                <div className="segment-goal">"{seg.goal}"</div>
                <div className="segment-divider" />
                <div className="segment-row">
                  <span className="segment-tag struggle">Struggle</span>
                  <p>{seg.struggle}</p>
                </div>
                <div className="segment-row">
                  <span className="segment-tag outcome">Outcome</span>
                  <p>{seg.outcome}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Solution Bridge ── */}
      <section className="about-solution">
        <div className="about-container">
          <span className="about-section-label about-animate">03 — The Solution</span>
          <h2 className="about-section-h2 about-animate">From vague intention to<br />structured action — in seconds.</h2>
          <p className="about-section-intro about-animate">
            VisualAI takes your goal as natural language input and returns a complete,
            step-by-step plan. No templates. No friction. Just describe what you want to achieve.
          </p>

          <div className="solution-demo about-animate" style={{ '--delay': '0.1s' } as React.CSSProperties}>
            <div className="solution-prompt">
              <span className="prompt-label">You type</span>
              <div className="prompt-text">"I want to learn machine learning and get a job as an ML engineer in 6 months."</div>
            </div>
            <div className="solution-arrow">→</div>
            <div className="solution-result">
              <span className="prompt-label">VisualAI returns</span>
              <ul className="solution-list">
                <li>Week-by-week learning roadmap</li>
                <li>Project milestones to build your portfolio</li>
                <li>Resources matched to your current skill level</li>
                <li>Job application timeline with realistic targets</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* ── Values (single row) ── */}
      <section className="about-values">
        <div className="about-container">
          <span className="about-section-label about-animate">Core values</span>
          <h2 className="about-animate">What drives us</h2>
          <div className="values-row">
            {[
              { icon: '🎯', title: 'Clarity', text: 'Making the complex simple — creating clear pathways through uncertainty.' },
              { icon: '🤝', title: 'Empowerment', text: 'AI that augments human capability, never replacing human potential.' },
              { icon: '🔍', title: 'Iteration', text: 'Constantly learning and improving, both in product and as people.' },
              { icon: '🛡️', title: 'Responsibility', text: 'Ethical, private, and secure AI — always at the core of everything.' },
            ].map((v, i) => (
              <div
                key={v.title}
                className="value-card about-animate"
                style={{ '--delay': `${i * 0.1}s` } as React.CSSProperties}
              >
                <div className="value-icon">{v.icon}</div>
                <h3>{v.title}</h3>
                <p>{v.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── About Me ── */}
      <section className="team-section">
        <div className="about-container">
          <span className="about-section-label about-animate">The maker</span>
          <h2 className="about-animate">About Me</h2>
          <div className="about-me-card about-animate">
            <div className="about-me-avatar">
              <div className="avatar-placeholder">SG</div>
            </div>
            <div className="about-me-info">
              <h3>Sarthak Gupta</h3>
              <p className="member-title">AI Product Manager</p>
              <p className="about-me-bio">
                Passionate about building tools that help people achieve their goals. VisualAI
                is a personal project combining my interest in AI and productivity — designed
                to take you from a vague intention to a structured, actionable plan.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="about-cta about-animate">
        <div className="about-cta-bg" />
        <div className="about-container" style={{ position: 'relative', zIndex: 1 }}>
          <h2>Start your journey today</h2>
          <p>Experience the power of AI-driven planning to achieve your goals.</p>
          <Link to="/create" className="about-cta-btn">Try VisualAI Free →</Link>
        </div>
      </section>

    </div>
  );
}
