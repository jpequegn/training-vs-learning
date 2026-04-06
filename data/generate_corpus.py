"""Generate the P3 podcast corpus from episode templates.

This mirrors the podcast-quest seed-episodes.ts logic, producing a deterministic
corpus of ~210 podcast episode summaries across 20 topics and 10 podcasts.
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

SEED = 42

TOPICS = [
    "ai-agents", "llms", "world-models", "infrastructure", "fintech",
    "product-management", "growth", "leadership", "developer-tools",
    "data-engineering", "platform-engineering", "security", "open-source",
    "startups", "strategy", "engineering-culture", "mobile", "cloud-native",
    "machine-learning", "design",
]

PODCASTS = [
    "Lenny's Podcast", "Acquired", "Lex Fridman Podcast", "The a16z Podcast",
    "Latent Space", "Software Engineering Daily", "CoRecursive", "Ship It!",
    "The Changelog", "Gradient Dissent",
]

EPISODE_TEMPLATES = {
    "ai-agents": {
        "titles": [
            "Building AI Agents That Actually Work in Production",
            "The Agent Loop: Observe, Think, Act",
            "Why Most AI Agents Fail and How to Fix Them",
            "Tool Use Patterns for LLM Agents",
            "Autonomous Agents vs. Human-in-the-Loop",
            "Multi-Agent Orchestration at Scale",
            "From Chatbots to Agents: The Evolution",
            "Agent Memory: Short-term, Long-term, and Episodic",
            "Reliability Engineering for AI Agents",
            "The Agent SDK Landscape in 2026",
            "Debugging Agents: Observability and Tracing",
            "Cost Optimization for Agent Workloads",
        ],
        "summary_prefix": "This episode explores the practical challenges and solutions in building AI agents.",
        "quotes": [
            "The biggest mistake teams make is giving agents too much autonomy too early.",
            "An agent is only as good as its tools — garbage tools, garbage agent.",
            "You need to think of agents as junior engineers: capable but needing guardrails.",
        ],
        "takeaways": [
            "Start with narrow, well-defined agent tasks before expanding scope",
            "Implement circuit breakers to prevent runaway agent loops",
            "Observability is non-negotiable for production agents",
        ],
    },
    "llms": {
        "titles": [
            "How Large Language Models Actually Work",
            "Fine-tuning vs. Prompting: When to Use Each",
            "The Context Window Revolution",
            "RAG Done Right: Retrieval-Augmented Generation",
            "Evaluating LLM Output Quality at Scale",
            "LLM Inference Optimization Deep Dive",
            "Structured Output from Language Models",
            "The Economics of Running LLMs",
            "Multimodal Models: Beyond Text",
            "Open-Weight Models vs. API Providers",
            "Prompt Engineering That Actually Works",
        ],
        "summary_prefix": "A deep technical discussion on large language models and their practical applications.",
        "quotes": [
            "The model is never the bottleneck — it's always the system around it.",
            "Fine-tuning is a commitment; prompting is a conversation.",
            "If you can't evaluate it, you can't improve it.",
        ],
        "takeaways": [
            "Use structured outputs (JSON mode) for any programmatic LLM use",
            "RAG quality depends more on chunking strategy than embedding model",
            "Always benchmark cost-per-token against quality for your specific use case",
        ],
    },
    "world-models": {
        "titles": [
            "World Models: How AI Understands Reality",
            "Simulation-Based Reasoning in AI",
            "From Language Models to World Models",
            "Spatial Intelligence and AI Navigation",
            "Physics-Informed Neural Networks",
            "Predictive World Models for Robotics",
            "The Embodied AI Revolution",
            "Generative World Simulators",
            "Causal Reasoning in AI Systems",
            "Video Prediction as World Modeling",
        ],
        "summary_prefix": "Exploring how AI systems build and use internal representations of the world.",
        "quotes": [
            "A world model is what separates a reactive system from an intelligent one.",
            "You can't navigate the world if you can't predict what happens next.",
            "The gap between language understanding and world understanding is still massive.",
        ],
        "takeaways": [
            "World models enable planning by simulating future states",
            "Multimodal training is essential for grounded world understanding",
            "Current LLMs have implicit world models, but they're unreliable",
        ],
    },
    "infrastructure": {
        "titles": [
            "The Modern Infrastructure Stack Explained",
            "Infrastructure as Code: Terraform vs. Pulumi",
            "Edge Computing for Real-Time Applications",
            "Database Infrastructure at Massive Scale",
            "Serverless Is Not Dead: The Next Chapter",
            "Observability-Driven Infrastructure",
            "Multi-Cloud Strategy: Myth vs. Reality",
            "GPU Infrastructure for AI Workloads",
            "The Rise of Platform Engineering Teams",
            "Infrastructure Cost Optimization Playbook",
            "Zero-Downtime Deployments Demystified",
        ],
        "summary_prefix": "A practical discussion about modern infrastructure patterns, trade-offs, and best practices.",
        "quotes": [
            "The best infrastructure is the one your team actually understands.",
            "Multi-cloud is a strategy, not a checkbox.",
            "If you're not measuring cost per request, you're flying blind.",
        ],
        "takeaways": [
            "Start with a single cloud provider and add complexity only when needed",
            "Platform engineering teams should treat developers as customers",
            "GPU scheduling is the new container scheduling challenge",
        ],
    },
    "fintech": {
        "titles": [
            "Building Payment Systems That Never Go Down",
            "The Embedded Finance Revolution",
            "Real-Time Payments Infrastructure",
            "Fraud Detection with Machine Learning",
            "Crypto Infrastructure for Traditional Finance",
            "Open Banking APIs and the Future of Finance",
            "Compliance Engineering: RegTech Done Right",
            "Neo-Banking Architecture Patterns",
            "Insurance Tech: Underwriting with AI",
            "The Stablecoin Infrastructure Stack",
        ],
        "summary_prefix": "An inside look at the technology powering modern financial systems.",
        "quotes": [
            "In fintech, an outage isn't just inconvenient — it's someone's rent payment failing.",
            "Compliance is a feature, not a constraint.",
            "The best fintech products are invisible — users just see their money working.",
        ],
        "takeaways": [
            "Design for idempotency from day one in payment systems",
            "Regulatory compliance should be embedded in the CI/CD pipeline",
            "Event sourcing is particularly valuable for financial audit trails",
        ],
    },
    "product-management": {
        "titles": [
            "Product Sense: How the Best PMs Think",
            "Writing Product Specs That Engineers Love",
            "Measuring What Matters: Product Metrics",
            "The Art of Saying No to Features",
            "Discovery vs. Delivery: Balancing Both",
            "Platform Product Management",
            "B2B Product-Led Growth Strategies",
            "Product-Market Fit: Beyond the Buzzword",
            "Internal Tools as Products",
            "AI-Native Product Design Patterns",
            "The PM's Guide to Technical Debt",
        ],
        "summary_prefix": "Insights on building products that users love while maintaining engineering velocity.",
        "quotes": [
            "The best product spec is the one the team actually reads.",
            "Saying no is the most important product skill nobody teaches.",
            "Product-market fit isn't a moment — it's a continuous negotiation.",
        ],
        "takeaways": [
            "Write specs as living documents, not waterfall requirements",
            "Leading indicators matter more than lagging metrics for product decisions",
            "Internal tools deserve the same product rigor as customer-facing ones",
        ],
    },
    "growth": {
        "titles": [
            "Viral Loops: Engineering Network Effects",
            "Growth Engineering at Spotify",
            "Retention Is the New Acquisition",
            "Experimentation Platforms at Scale",
            "SEO Engineering for Product-Led Growth",
            "Referral Programs That Actually Work",
            "Growth Analytics: Beyond Vanity Metrics",
            "Onboarding Optimization: The First 5 Minutes",
            "International Growth: Localization Engineering",
            "Activation Rate: The Most Underrated Metric",
        ],
        "summary_prefix": "A practical guide to building growth engines and optimizing user acquisition funnels.",
        "quotes": [
            "Growth is not a hack — it's a discipline built on data and iteration.",
            "If your retention curve doesn't flatten, nothing else matters.",
            "The best referral program is a product people can't stop talking about.",
        ],
        "takeaways": [
            "Focus on activation before acquisition — fix the leaky bucket first",
            "A/B test infrastructure should be a first-class engineering concern",
            "Localization is a growth multiplier, not a translation exercise",
        ],
    },
    "leadership": {
        "titles": [
            "Engineering Management: The First 90 Days",
            "Scaling Engineering Teams from 10 to 100",
            "The Staff Engineer Path",
            "Hiring Engineers: What Actually Predicts Success",
            "Remote-First Engineering Culture",
            "Managing Technical Debt as a Leader",
            "From IC to Manager and Back Again",
            "Building High-Performing Engineering Teams",
            "The CTO's Playbook for Series A to B",
            "Mentorship Programs That Actually Work",
            "Incident Management and Blameless Culture",
        ],
        "summary_prefix": "Leadership lessons for engineering managers and technical leaders at growing companies.",
        "quotes": [
            "Your job as a manager is to make yourself unnecessary.",
            "The best engineering cultures are built on trust, not process.",
            "Hiring slowly and firing quickly sounds harsh until you've done the opposite.",
        ],
        "takeaways": [
            "New managers should focus on listening for the first 30 days",
            "Staff engineers need explicit scope and executive sponsorship",
            "Blameless postmortems only work if leadership models the behavior",
        ],
    },
    "developer-tools": {
        "titles": [
            "Building Developer Tools Developers Actually Use",
            "The IDE Revolution: AI-Powered Coding",
            "CLI Design Principles for Great DX",
            "API Design That Scales",
            "DevEx Metrics: Measuring Developer Productivity",
            "The Testing Tool Renaissance",
            "Monorepo Tooling at Scale",
            "Code Review Automation and AI",
            "Documentation as a Developer Tool",
            "Build Systems Compared: Turborepo vs. Nx vs. Bazel",
        ],
        "summary_prefix": "How to build, evaluate, and adopt developer tools that genuinely improve productivity.",
        "quotes": [
            "The best developer tool is the one that gets out of your way.",
            "If your CLI requires a manual, you've already lost.",
            "Developer experience is user experience for engineers.",
        ],
        "takeaways": [
            "Measure time-to-first-value when evaluating developer tools",
            "Progressive disclosure applies to CLIs just as much as UIs",
            "API versioning strategy should be decided before v1 ships",
        ],
    },
    "data-engineering": {
        "titles": [
            "The Modern Data Stack in 2026",
            "Streaming vs. Batch: When to Use Each",
            "Data Quality Engineering at Scale",
            "Building Data Pipelines with dbt",
            "Real-Time Analytics Architecture",
            "Data Mesh: Lessons from the Trenches",
            "The Lakehouse Architecture Explained",
            "Feature Stores for ML Engineering",
            "Data Contracts: Schema Management Done Right",
            "Event-Driven Data Architecture",
        ],
        "summary_prefix": "Practical patterns for building reliable, scalable data infrastructure.",
        "quotes": [
            "Bad data is worse than no data — at least with no data you know you're guessing.",
            "The modern data stack is less about tools and more about contracts.",
            "If your pipeline fails silently, it's not a pipeline — it's a prayer.",
        ],
        "takeaways": [
            "Data contracts between producers and consumers prevent most pipeline failures",
            "Start with batch processing and add streaming only where latency matters",
            "Data quality checks should run in the pipeline, not after the fact",
        ],
    },
    "platform-engineering": {
        "titles": [
            "Platform Engineering: Building the Golden Path",
            "Internal Developer Platforms That Scale",
            "Platform as a Product Mindset",
            "Self-Service Infrastructure for Developers",
            "Backstage and the Developer Portal Ecosystem",
            "Platform Team Topology and Org Design",
            "Standardization vs. Flexibility in Platforms",
            "Measuring Platform Engineering Success",
            "Migration Strategies for Platform Teams",
            "The Platform Engineering Maturity Model",
        ],
        "summary_prefix": "How platform engineering teams build internal developer platforms that accelerate delivery.",
        "quotes": [
            "A platform that nobody uses is just infrastructure with a marketing problem.",
            "The golden path should be paved, not enforced.",
            "Platform teams fail when they forget they're building for developers, not for themselves.",
        ],
        "takeaways": [
            "Treat your internal platform as a product with real users and feedback loops",
            "Start with the most painful developer workflow and automate that first",
            "Platform adoption should be measured, not mandated",
        ],
    },
    "security": {
        "titles": [
            "Zero Trust Architecture in Practice",
            "Supply Chain Security for Software",
            "Secrets Management at Scale",
            "Security Engineering for AI Systems",
            "Penetration Testing Automation",
            "Identity and Access Management Modernization",
            "Securing the Software Development Lifecycle",
            "Threat Modeling for Engineering Teams",
            "Runtime Security and eBPF",
            "The CISO's Engineering Perspective",
        ],
        "summary_prefix": "Security engineering practices that scale with your organization without slowing development.",
        "quotes": [
            "Security is everyone's job, but it's no one's job if there's no team owning it.",
            "The fastest way to fail a security audit is to treat security as an afterthought.",
            "Zero trust isn't a product — it's a principle applied consistently.",
        ],
        "takeaways": [
            "Shift security left by integrating scanning into CI/CD pipelines",
            "Secrets rotation should be automated, not a quarterly manual task",
            "Threat modeling at design time is 100x cheaper than fixing in production",
        ],
    },
    "open-source": {
        "titles": [
            "Sustaining Open Source: Business Models That Work",
            "Open Source Community Building from Scratch",
            "The Open Source Licensing Landscape in 2026",
            "Corporate Open Source Strategy",
            "From Side Project to OSS Ecosystem",
            "Open Source Security and Supply Chain",
            "Maintainer Burnout: Causes and Solutions",
            "Building a Business Around Open Source",
            "Open Source Governance Models",
            "Contributing to Open Source as a Company",
        ],
        "summary_prefix": "The business, community, and technical aspects of building and maintaining open source software.",
        "quotes": [
            "Open source isn't free — someone is always paying, usually the maintainer.",
            "The best open source projects have opinions, not just options.",
            "Community is the moat that no proprietary competitor can replicate.",
        ],
        "takeaways": [
            "Choose a license that matches your business model from day one",
            "Invest in contributor documentation as much as user documentation",
            "Corporate OSS programs need executive sponsorship to survive reorgs",
        ],
    },
    "startups": {
        "titles": [
            "Technical Co-Founder Matching: What Actually Works",
            "Building the First Version: Speed vs. Quality",
            "Startup Architecture: Start Simple, Scale Later",
            "Fundraising for Technical Founders",
            "When to Hire Your First Engineer",
            "Pivoting Without Losing Momentum",
            "Startup Metrics That Investors Actually Care About",
            "Building in Public: Lessons Learned",
            "Solo Founder Engineering Productivity",
            "The First 100 Customers: Technical Founders' Guide",
        ],
        "summary_prefix": "Hard-won lessons from technical founders on building companies from zero to one.",
        "quotes": [
            "Your first version should embarrass you — if it doesn't, you shipped too late.",
            "Technical debt in a startup is just speed you haven't paid for yet.",
            "The best startup architecture is one that lets you iterate in hours, not weeks.",
        ],
        "takeaways": [
            "Ship the simplest thing that could work, then iterate based on user feedback",
            "Choose boring technology for your startup's foundation",
            "Revenue is the best validation — not users, not signups, revenue",
        ],
    },
    "strategy": {
        "titles": [
            "Platform Strategy: Building Moats with APIs",
            "Competitive Analysis for Technical Products",
            "Build vs. Buy: A Framework for Decisions",
            "The Marketplace Flywheel Effect",
            "Pricing Strategy for Developer Products",
            "Network Effects in B2B Software",
            "Technical Due Diligence for Acquisitions",
            "Vertical SaaS Strategy and Execution",
            "The Bundling and Unbundling Cycle",
            "Ecosystem Strategy: Partners as Growth Engines",
        ],
        "summary_prefix": "Strategic thinking for technical leaders navigating competitive markets and business decisions.",
        "quotes": [
            "The best strategy is the one your team can actually execute.",
            "Build vs. buy isn't a technical decision — it's a time-to-market decision.",
            "Network effects are earned through value, not virality.",
        ],
        "takeaways": [
            "Document your build-vs-buy decisions with a decision log",
            "Pricing should be revisited quarterly, not annually",
            "API-first companies create natural platform lock-in through integration depth",
        ],
    },
    "engineering-culture": {
        "titles": [
            "Code Review Culture That Doesn't Slow You Down",
            "On-Call Engineering: Making It Sustainable",
            "Technical RFC Processes That Work",
            "Engineering Guilds and Communities of Practice",
            "Continuous Deployment Culture",
            "Knowledge Sharing in Distributed Teams",
            "Psychological Safety in Engineering Teams",
            "Tech Debt Fridays and Other Rituals",
            "Engineering Career Ladders Done Right",
            "Post-Incident Reviews That Drive Change",
        ],
        "summary_prefix": "Building engineering cultures that attract top talent and ship great software consistently.",
        "quotes": [
            "Culture is what happens when management isn't looking.",
            "The fastest way to kill innovation is to punish failure.",
            "Good engineering culture scales; good engineers without culture don't.",
        ],
        "takeaways": [
            "Code review should be a learning opportunity, not a gatekeeping exercise",
            "On-call rotations need adequate compensation and recovery time",
            "RFCs should have clear decision deadlines to prevent analysis paralysis",
        ],
    },
    "mobile": {
        "titles": [
            "React Native vs. Flutter in 2026",
            "Mobile Performance Optimization Deep Dive",
            "App Store Optimization Engineering",
            "Offline-First Mobile Architecture",
            "Mobile CI/CD Pipeline Design",
            "Cross-Platform UI Components",
            "Mobile Security Best Practices",
            "Push Notification Infrastructure",
            "Mobile Analytics and Crash Reporting",
            "The Future of Mobile Development",
        ],
        "summary_prefix": "Technical deep dives into mobile engineering challenges and cross-platform development.",
        "quotes": [
            "Mobile performance isn't optional — it's the difference between installed and deleted.",
            "Cross-platform doesn't mean write once; it means maintain once.",
            "Offline-first isn't just for airplanes — it's for elevators, subways, and bad WiFi.",
        ],
        "takeaways": [
            "Measure app startup time as a key performance indicator",
            "Offline-first architecture prevents an entire class of UX failures",
            "Mobile CI/CD should include real-device testing, not just emulators",
        ],
    },
    "cloud-native": {
        "titles": [
            "Kubernetes Operators: Building Custom Controllers",
            "Service Mesh Patterns: Istio vs. Linkerd",
            "Cloud-Native Databases and NewSQL",
            "Container Runtime Deep Dive",
            "GitOps for Cloud-Native Deployments",
            "Cloud-Native Observability Stack",
            "Scaling Microservices Without the Pain",
            "FinOps: Cloud Cost Engineering",
            "Cloud-Native Security Posture Management",
            "Serverless Containers and the Future of Compute",
        ],
        "summary_prefix": "Best practices for building and operating cloud-native applications at scale.",
        "quotes": [
            "Kubernetes is the Linux of the cloud — powerful, complex, and everywhere.",
            "Microservices are a deployment strategy, not an architecture.",
            "FinOps is what happens when engineering meets the credit card bill.",
        ],
        "takeaways": [
            "Start with a service mesh only when you have more than 10 services",
            "GitOps eliminates configuration drift by making git the source of truth",
            "Tag every cloud resource from day one to enable cost attribution",
        ],
    },
    "machine-learning": {
        "titles": [
            "MLOps: Getting Models to Production",
            "Feature Engineering That Scales",
            "ML Model Monitoring and Drift Detection",
            "Training Infrastructure Optimization",
            "Experiment Tracking Best Practices",
            "ML System Design Patterns",
            "Responsible AI and Fairness Engineering",
            "Edge ML: Running Models on Devices",
            "AutoML and Neural Architecture Search",
            "The ML Platform Engineering Playbook",
        ],
        "summary_prefix": "Practical machine learning engineering: from model development to production deployment.",
        "quotes": [
            "The model is 5% of the work; the system around it is the other 95%.",
            "Data quality problems masquerade as model quality problems.",
            "If you can't reproduce a training run, you can't debug it.",
        ],
        "takeaways": [
            "Version your training data with the same rigor as your code",
            "Model monitoring should detect data drift before performance degrades",
            "Start with simple models and established baselines before going deep",
        ],
    },
    "design": {
        "titles": [
            "Design Systems: Building and Maintaining at Scale",
            "Accessibility Engineering: Beyond Compliance",
            "Design Tokens: Bridging Design and Code",
            "Component Library Architecture",
            "Design-Engineering Collaboration Models",
            "Performance-Conscious UI Design",
            "Motion Design in Product Interfaces",
            "Design Review Processes That Work",
            "Responsive Design Engineering Patterns",
            "The Design System Team: Roles and Workflows",
        ],
        "summary_prefix": "Where design meets engineering: building scalable, accessible, and beautiful product interfaces.",
        "quotes": [
            "A design system is a product, not a project — it needs continuous investment.",
            "Accessibility isn't a feature; it's a quality of good engineering.",
            "The best design tokens are the ones designers and engineers both understand.",
        ],
        "takeaways": [
            "Design systems need adoption metrics, not just component counts",
            "Accessibility testing should be automated in CI, not manual and quarterly",
            "Design tokens create a shared language between design tools and code",
        ],
    },
}


def generate_episodes(seed: int = SEED) -> list[dict]:
    """Generate the full episode corpus deterministically."""
    rng = random.Random(seed)
    episodes = []

    for topic in TOPICS:
        template = EPISODE_TEMPLATES[topic]
        secondary_topics = [t for t in TOPICS if t != topic]

        for title in template["titles"]:
            podcast = rng.choice(PODCASTS)
            cross_topics = rng.sample(secondary_topics, rng.randint(1, 2))

            # Deterministic date: 2023-01-01 to 2026-06-30
            ts = rng.uniform(
                datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp(),
                datetime(2026, 6, 30, tzinfo=timezone.utc).timestamp(),
            )
            published_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

            quotes = rng.sample(template["quotes"], 2)
            takeaways = rng.sample(template["takeaways"], 2)

            full_summary = (
                f"{template['summary_prefix']} The guest shares insights on "
                f'"{title.lower()}", covering practical strategies, common pitfalls, '
                f"and lessons learned from real-world implementations. Key themes include "
                f"the importance of iterative approaches, measuring outcomes over outputs, "
                f"and building systems that are maintainable long-term."
            )

            episodes.append({
                "title": title,
                "podcast": podcast,
                "published_at": published_at,
                "key_topics": [topic] + cross_topics,
                "quotes": quotes,
                "key_takeaways": takeaways,
                "full_summary": full_summary,
            })

    return episodes


if __name__ == "__main__":
    episodes = generate_episodes()
    out = Path(__file__).parent / "episodes_raw.json"
    with open(out, "w") as f:
        json.dump(episodes, f, indent=2)
    print(f"Generated {len(episodes)} episodes -> {out}")
