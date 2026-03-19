import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useUser } from '@clerk/clerk-react';

export default function PricingPage() {
  useUser();
  const [annualBilling, setAnnualBilling] = useState(true);

  const toggleBilling = () => {
    setAnnualBilling(!annualBilling);
  };

  // Calculate discounted prices for annual billing
  const getPrice = (monthlyPrice: number) => {
    if (annualBilling) {
      // 20% discount for annual billing
      return Math.round(monthlyPrice * 0.8);
    }
  };

  return (
    <div className="pricing-page">
      <section className="pricing-hero">
        <h1>Simple, Transparent Pricing</h1>
        <p className="pricing-subtitle">Choose the plan that's right for you</p>
        
        <div className="billing-toggle">
          <span className={!annualBilling ? 'active' : ''}>Monthly</span>
          <label className="switch">
            <input 
              type="checkbox" 
              checked={annualBilling} 
              onChange={toggleBilling} 
            />
            <span className="slider round"></span>
          </label>
          <span className={annualBilling ? 'active' : ''}>Annual <span className="discount-badge">Save 20%</span></span>
        </div>
      </section>

      <section className="pricing-plans">
        <div className="pricing-grid">
          <div className="pricing-card">
            <div className="plan-header">
              <h2>Free</h2>
              <div className="plan-price">
                <span className="price">$0</span>
                <span className="period">forever</span>
              </div>
              <p className="plan-description">Perfect for getting started with AI planning</p>
            </div>
            <div className="plan-features">
              <ul>
                <li>✓ Create up to 3 plans</li>
                <li>✓ Basic task tracking</li>
                <li>✓ Email support</li>
                <li>✗ Timeline feature</li>
                <li>✗ Plan exports</li>
                <li>✗ Advanced customization</li>
              </ul>
            </div>
            <div className="plan-action">
              <Link to="/create" className="secondary-button">Start Free</Link>
            </div>
          </div>

          <div className="pricing-card popular">
            <div className="popular-badge">Most Popular</div>
            <div className="plan-header">
              <h2>Pro</h2>
              <div className="plan-price">
                <span className="price">${getPrice(12)}</span>
                <span className="period">per month</span>
                {annualBilling && <div className="billed-annually">billed annually</div>}
              </div>
              <p className="plan-description">Advanced features for individual professionals</p>
            </div>
            <div className="plan-features">
              <ul>
                <li>✓ Unlimited plans</li>
                <li>✓ Advanced task tracking</li>
                <li>✓ Timeline feature</li>
                <li>✓ Priority support</li>
                <li>✓ Export as PDF/CSV</li>
                <li>✗ Team collaboration</li>
              </ul>
            </div>
            <div className="plan-action">
              <Link to="/create" className="primary-button">Choose Pro</Link>
            </div>
          </div>

          <div className="pricing-card">
            <div className="plan-header">
              <h2>Team</h2>
              <div className="plan-price">
                <span className="price">${getPrice(29)}</span>
                <span className="period">per month</span>
                {annualBilling && <div className="billed-annually">billed annually</div>}
              </div>
              <p className="plan-description">Collaboration features for teams and businesses</p>
            </div>
            <div className="plan-features">
              <ul>
                <li>✓ Everything in Pro</li>
                <li>✓ Team collaboration</li>
                <li>✓ Shared workspaces</li>
                <li>✓ Admin controls</li>
                <li>✓ API access</li>
                <li>✓ Dedicated success manager</li>
              </ul>
            </div>
            <div className="plan-action">
              <Link to="/create" className="secondary-button">Choose Team</Link>
            </div>
          </div>
        </div>
      </section>

      <section className="faq-section">
        <h2>Frequently Asked Questions</h2>
        <div className="faq-grid">
          <div className="faq-item">
            <h3>Can I change plans later?</h3>
            <p>Yes, you can upgrade, downgrade, or cancel your plan at any time. Changes will take effect at the end of your current billing period.</p>
          </div>
          <div className="faq-item">
            <h3>Is there a free trial?</h3>
            <p>Yes, all paid plans include a 14-day free trial with full access to all features. No credit card required to start your trial.</p>
          </div>
          <div className="faq-item">
            <h3>How does billing work?</h3>
            <p>We offer monthly and annual billing options. Annual plans are discounted at 20% compared to monthly plans.</p>
          </div>
          <div className="faq-item">
            <h3>What payment methods do you accept?</h3>
            <p>We accept all major credit cards, PayPal, and select local payment methods depending on your region.</p>
          </div>
          <div className="faq-item">
            <h3>How secure is my data?</h3>
            <p>VisualAI employs industry-standard security measures to protect your data. We use encryption, secure data centers, and regular security audits.</p>
          </div>
          <div className="faq-item">
            <h3>Can I get a refund?</h3>
            <p>If you're not satisfied with VisualAI within the first 30 days, contact our support team for a full refund.</p>
          </div>
        </div>
      </section>

      <section className="enterprise-section">
        <div className="enterprise-content">
          <h2>Enterprise Solutions</h2>
          <p>Need a custom solution for your large organization? Our enterprise plan includes custom integrations, dedicated support, and organization-wide deployment.</p>
          <a href="#" className="secondary-button">Contact Sales</a>
        </div>
      </section>
    </div>
  );
}