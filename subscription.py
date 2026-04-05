import streamlit as st
import pandas as pd

PLANS = {
    "free": {
        "name": "Free",
        "price": "0",
        "color": "#6b7280",
        "servers": 1,
        "features": [
            ("Live CPU Monitor", True),
            ("Anomaly Detection", True),
            ("Basic Prediction", True),
            ("Data & Training", True),
            ("SHAP Explainability", False),
            ("Multi-Step Forecast", False),
            ("MLflow Tracking", False),
            ("Slack Alerts", False),
            ("Multi-Server Monitor", False),
            ("Ablation Study", False),
            ("Carbon-Aware Scheduling", False),
            ("Federated Learning", False),
            ("PDF Reports", False),
        ]
    },
    "pro": {
        "name": "Pro",
        "price": "5,000",
        "color": "#6366f1",
        "servers": 5,
        "features": [
            ("Live CPU Monitor", True),
            ("Anomaly Detection", True),
            ("Advanced Prediction", True),
            ("Data & Training", True),
            ("SHAP Explainability", True),
            ("Multi-Step Forecast", True),
            ("MLflow Tracking", True),
            ("Slack + Email Alerts", True),
            ("Multi-Server Monitor", True),
            ("Ablation Study", False),
            ("Carbon-Aware Scheduling", False),
            ("Federated Learning", False),
            ("PDF Reports", False),
        ]
    },
    "enterprise": {
        "name": "Enterprise",
        "price": "1,00,000",
        "color": "#f59e0b",
        "servers": 999,
        "features": [
            ("Live CPU Monitor", True),
            ("Anomaly Detection", True),
            ("Research-Grade Prediction", True),
            ("Data & Training", True),
            ("SHAP Explainability", True),
            ("Multi-Step Forecast", True),
            ("MLflow Tracking", True),
            ("Slack + Email + SMS Alerts", True),
            ("Multi-Server Monitor", True),
            ("Ablation Study", True),
            ("Carbon-Aware Scheduling", True),
            ("Federated Learning", True),
            ("PDF Reports + Dedicated Support", True),
        ]
    }
}

PLAN_LEVELS = {"free": 0, "pro": 1, "enterprise": 2}

FEATURE_REQUIREMENTS = {
    "shap":           "pro",
    "multistep":      "pro",
    "mlflow":         "pro",
    "alerts_slack":   "pro",
    "multiserver":    "pro",
    "confidence":     "pro",
    "online":         "pro",
    "expl_alerts":    "pro",
    "error_analysis": "pro",
    "ablation":       "enterprise",
    "carbon":         "enterprise",
    "federated":      "enterprise",
    "pdf":            "enterprise",
    "auto_retrain":   "enterprise",
}


def check_feature_access(feature_key):
    current = st.session_state.get("current_plan", "free")
    required = FEATURE_REQUIREMENTS.get(feature_key, "free")
    return PLAN_LEVELS.get(current, 0) >= PLAN_LEVELS.get(required, 0)


def show_locked_message(feature_name, feature_key, instance_id=""):
    """
    instance_id — pass a unique string when the same feature_key
    appears in more than one tab (e.g. "tab16", "tab17").
    This guarantees the Streamlit button key is always unique.
    """
    required_plan = FEATURE_REQUIREMENTS.get(feature_key, "pro")
    plan_prices   = {"pro": "₹5,000/mo", "enterprise": "₹1,00,000/mo"}
    current       = st.session_state.get("current_plan", "free")
    plan_color    = "#6366f1" if required_plan == "pro" else "#f59e0b"
    plan_label    = "PRO"     if required_plan == "pro" else "ENTERPRISE"

    # Unique key = feature_key + optional instance_id
    btn_key = f"upgrade_{feature_key}"
    if instance_id:
        btn_key += f"_{instance_id}"

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
        border: 1px solid {plan_color}40;
        border-radius: 16px;
        padding: 60px 40px;
        text-align: center;
        margin: 20px 0;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute; top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, transparent, {plan_color}, transparent);
        "></div>
        <div style="
            display: inline-flex; align-items: center; justify-content: center;
            width: 64px; height: 64px; border-radius: 16px;
            background: {plan_color}20; border: 1px solid {plan_color}40;
            font-size: 28px; margin-bottom: 20px;
        ">🔒</div>
        <h2 style="color:#fff; font-size:22px; font-weight:600; margin:0 0 8px;">
            {feature_name}
        </h2>
        <p style="color:#888; font-size:15px; margin:0 0 24px;">
            This feature requires the
            <span style="color:{plan_color}; font-weight:600;">{plan_label}</span>
            plan
        </p>
        <div style="
            display:inline-block;
            background:{plan_color}15;
            border:1px solid {plan_color}40;
            border-radius:8px;
            padding:12px 24px;
            margin-bottom:8px;
        ">
            <span style="color:{plan_color}; font-size:13px; font-weight:500;">
                Starting at {plan_prices.get(required_plan, "")} per organization
            </span>
        </div>
        <p style="color:#555; font-size:13px; margin:16px 0 0;">
            Currently on:
            <span style="color:#888; font-weight:500;">{current.upper()}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            f"Upgrade to {plan_label} →",
            key=btn_key,
            use_container_width=True
        ):
            st.session_state["logged_in"]              = False
            st.session_state["show_pro_form"]          = (required_plan == "pro")
            st.session_state["show_enterprise_form"]   = (required_plan == "enterprise")
            st.rerun()


def show_pricing_page():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
    .stApp { background: #080810 !important; }
    .block-container { padding-top: 0 !important; max-width: 100% !important; }
    .cg-hero {
        background:#080810; padding:80px 60px 60px;
        text-align:center; position:relative; overflow:hidden;
    }
    .cg-hero::before {
        content:''; position:absolute; top:-200px; left:50%;
        transform:translateX(-50%); width:800px; height:600px;
        background:radial-gradient(ellipse,#6366f130 0%,transparent 70%);
        pointer-events:none;
    }
    .cg-badge {
        display:inline-flex; align-items:center; gap:8px;
        background:#6366f115; border:1px solid #6366f130;
        border-radius:100px; padding:6px 16px;
        font-family:'DM Sans',sans-serif;
        font-size:13px; color:#818cf8; font-weight:500; margin-bottom:32px;
    }
    .cg-title {
        font-family:'Syne',sans-serif; font-size:clamp(42px,6vw,72px);
        font-weight:800; color:#fff; line-height:1.05;
        letter-spacing:-2px; margin:0 0 20px;
    }
    .cg-title span {
        background:linear-gradient(135deg,#6366f1,#a78bfa,#818cf8);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        background-clip:text;
    }
    .cg-subtitle {
        font-family:'DM Sans',sans-serif; font-size:18px; color:#666;
        line-height:1.6; max-width:560px; margin:0 auto 48px;
    }
    .cg-stats {
        display:flex; justify-content:center; gap:0;
        padding:32px 0; border-top:1px solid #ffffff08;
        border-bottom:1px solid #ffffff08; margin:0 0 60px;
    }
    .cg-stat { flex:1; max-width:200px; text-align:center; padding:0 32px; border-right:1px solid #ffffff08; }
    .cg-stat:last-child { border-right:none; }
    .cg-stat-value { font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#fff; }
    .cg-stat-label { font-family:'DM Sans',sans-serif; font-size:13px; color:#555; margin-top:4px; }
    .cg-stat-delta { font-family:'DM Sans',sans-serif; font-size:12px; color:#4ade80; margin-top:2px; font-weight:500; }
    div[data-testid="stMetric"] {
        background:#0d0d1a !important; border:1px solid #1a1a2e !important;
        border-radius:12px !important; padding:20px !important;
    }
    div[data-testid="stMetricValue"] { font-family:'Syne',sans-serif !important; font-size:28px !important; color:#fff !important; }
    div[data-testid="stMetricLabel"] { font-family:'DM Sans',sans-serif !important; color:#555 !important; font-size:13px !important; }
    .stTextInput input { background:#080810 !important; border:1px solid #1a1a2e !important; border-radius:10px !important; color:#fff !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── HERO ──
    st.markdown("""
    <div class="cg-hero">
        <div class="cg-badge">⚡ Now in Beta — Join 200+ engineering teams</div>
        <h1 class="cg-title">
            Stop watching crashes happen.<br>
            <span>Predict them 5 minutes early.</span>
        </h1>
        <p class="cg-subtitle">
            CrashGuard AI uses deep learning to forecast CPU spikes before they crash your servers —
            with explainability features that even Google Vertex AI doesn't offer.
        </p>
        <div class="cg-stats">
            <div class="cg-stat">
                <div class="cg-stat-value">94.7%</div>
                <div class="cg-stat-label">Prediction accuracy</div>
                <div class="cg-stat-delta">↑ TCN model</div>
            </div>
            <div class="cg-stat">
                <div class="cg-stat-value">5 min</div>
                <div class="cg-stat-label">Early warning time</div>
                <div class="cg-stat-delta">↑ before spike</div>
            </div>
            <div class="cg-stat">
                <div class="cg-stat-value">₹1L</div>
                <div class="cg-stat-label">Avg crash cost/min</div>
                <div class="cg-stat-delta">↑ saved per event</div>
            </div>
            <div class="cg-stat">
                <div class="cg-stat-value">3</div>
                <div class="cg-stat-label">Unique features</div>
                <div class="cg-stat-delta">↑ not in Vertex AI</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── PLANS HEADER ──
    st.markdown("""
    <div style="text-align:center; padding:20px 0 40px;">
        <p style="font-family:Syne,sans-serif; font-size:32px; font-weight:700; color:#fff; margin:0 0 8px;">
            Simple, transparent pricing
        </p>
        <p style="font-family:DM Sans,sans-serif; font-size:15px; color:#555; margin:0;">
            Start free. Scale as you grow. No hidden fees.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── PLAN CARDS ──
    col_free, col_pro, col_ent = st.columns(3)

    with col_free:
        st.markdown("""
        <div style="background:#0d0d1a; border:1px solid #1a1a2e; border-radius:16px; padding:28px;">
            <p style="font-family:DM Sans,sans-serif; font-size:11px; font-weight:600;
                      letter-spacing:2px; text-transform:uppercase; color:#6b7280; margin:0 0 12px;">FREE</p>
            <p style="font-family:Syne,sans-serif; font-size:36px; font-weight:800; color:#fff; margin:0 0 2px;">₹0</p>
            <p style="font-family:DM Sans,sans-serif; font-size:13px; color:#444; margin:0 0 16px;">forever · 1 server</p>
            <div style="height:1px; background:#ffffff08; margin:16px 0;"></div>
        </div>
        """, unsafe_allow_html=True)
        for feat, active in PLANS["free"]["features"]:
            color = "#ccc" if active else "#333"
            icon_color = "#4ade80" if active else "#333"
            icon = "✓" if active else "–"
            st.markdown(f'<p style="font-family:DM Sans,sans-serif; font-size:13px; color:{color}; margin:5px 0;"><span style="color:{icon_color}; margin-right:8px;">{icon}</span>{feat}</p>', unsafe_allow_html=True)
        st.markdown("")
        if st.button("Start for free →", key="plan_btn_free", use_container_width=True):
            st.session_state.update({"current_plan": "free", "logged_in": True, "user_name": "Free User", "user_company": ""})
            st.rerun()

    with col_pro:
        st.markdown("""
        <div style="background:linear-gradient(180deg,#0d0d2a 0%,#0d0d1a 100%);
                    border:2px solid #6366f140; border-radius:16px; padding:28px; position:relative;">
            <div style="position:absolute; top:-12px; left:50%; transform:translateX(-50%);
                        background:#6366f1; color:#fff; font-family:DM Sans,sans-serif;
                        font-size:11px; font-weight:700; letter-spacing:1.5px;
                        padding:4px 14px; border-radius:100px; white-space:nowrap;">
                MOST POPULAR
            </div>
            <p style="font-family:DM Sans,sans-serif; font-size:11px; font-weight:600;
                      letter-spacing:2px; text-transform:uppercase; color:#818cf8; margin:12px 0 12px;">PRO</p>
            <p style="font-family:Syne,sans-serif; font-size:36px; font-weight:800; color:#fff; margin:0 0 2px;">₹5,000</p>
            <p style="font-family:DM Sans,sans-serif; font-size:13px; color:#444; margin:0 0 16px;">per month · 5 servers</p>
            <div style="height:1px; background:#ffffff08; margin:16px 0;"></div>
        </div>
        """, unsafe_allow_html=True)
        for feat, active in PLANS["pro"]["features"]:
            color = "#ccc" if active else "#333"
            icon_color = "#4ade80" if active else "#333"
            icon = "✓" if active else "–"
            st.markdown(f'<p style="font-family:DM Sans,sans-serif; font-size:13px; color:{color}; margin:5px 0;"><span style="color:{icon_color}; margin-right:8px;">{icon}</span>{feat}</p>', unsafe_allow_html=True)
        st.markdown("")
        if st.button("Get Pro →", key="plan_btn_pro", use_container_width=True, type="primary"):
            st.session_state["show_pro_form"] = True
            st.session_state["show_enterprise_form"] = False

    with col_ent:
        st.markdown("""
        <div style="background:#0d0d1a; border:1px solid #f59e0b30; border-radius:16px; padding:28px;">
            <p style="font-family:DM Sans,sans-serif; font-size:11px; font-weight:600;
                      letter-spacing:2px; text-transform:uppercase; color:#f59e0b; margin:0 0 12px;">ENTERPRISE</p>
            <p style="font-family:Syne,sans-serif; font-size:36px; font-weight:800; color:#fff; margin:0 0 2px;">₹1,00,000</p>
            <p style="font-family:DM Sans,sans-serif; font-size:13px; color:#444; margin:0 0 16px;">per month · unlimited</p>
            <div style="height:1px; background:#ffffff08; margin:16px 0;"></div>
        </div>
        """, unsafe_allow_html=True)
        for feat, active in PLANS["enterprise"]["features"]:
            color = "#ccc" if active else "#333"
            icon_color = "#4ade80" if active else "#333"
            icon = "✓" if active else "–"
            st.markdown(f'<p style="font-family:DM Sans,sans-serif; font-size:13px; color:{color}; margin:5px 0;"><span style="color:{icon_color}; margin-right:8px;">{icon}</span>{feat}</p>', unsafe_allow_html=True)
        st.markdown("")
        if st.button("Contact Sales →", key="plan_btn_ent", use_container_width=True):
            st.session_state["show_enterprise_form"] = True
            st.session_state["show_pro_form"] = False

    # ── PRO FORM ──
    if st.session_state.get("show_pro_form"):
        st.divider()
        st.markdown('<p style="font-family:Syne,sans-serif; font-size:22px; font-weight:700; color:#fff; text-align:center;">Get started with Pro</p>', unsafe_allow_html=True)
        with st.form("pro_signup_form"):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Full name")
            with c2:
                company = st.text_input("Company")
            email = st.text_input("Work email")
            if st.form_submit_button("Activate Pro →", use_container_width=True):
                if name and email and company:
                    st.session_state.update({
                        "current_plan": "pro", "logged_in": True,
                        "user_name": name, "user_company": company,
                        "show_pro_form": False
                    })
                    st.success(f"Welcome {name}! Pro plan activated for {company}!")
                    st.info("In production: Razorpay processes ₹5,000/month automatically.")
                    st.balloons()
                    st.rerun()

    # ── ENTERPRISE FORM ──
    if st.session_state.get("show_enterprise_form"):
        st.divider()
        st.markdown('<p style="font-family:Syne,sans-serif; font-size:22px; font-weight:700; color:#fff; text-align:center;">Talk to our team</p>', unsafe_allow_html=True)
        with st.form("enterprise_signup_form"):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Full name")
            with c2:
                company = st.text_input("Company")
            c3, c4 = st.columns(2)
            with c3:
                email = st.text_input("Work email")
            with c4:
                servers = st.number_input("Number of servers", 1, 100000, 100)
            if st.form_submit_button("Request demo →", use_container_width=True):
                if name and email and company:
                    st.session_state.update({
                        "current_plan": "enterprise", "logged_in": True,
                        "user_name": name, "user_company": company,
                        "show_enterprise_form": False
                    })
                    st.success(f"Welcome {name}! Enterprise plan activated!")
                    st.info("In production: Sales team contacts within 24 hours.")
                    st.balloons()
                    st.rerun()

    # ── COMPARISON TABLE ──
    st.divider()
    st.markdown('<p style="font-family:Syne,sans-serif; font-size:26px; font-weight:700; color:#fff; text-align:center; margin-bottom:8px;">How we compare</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:DM Sans,sans-serif; font-size:14px; color:#555; text-align:center; margin-bottom:24px;">CrashGuard AI has 3 features no enterprise tool offers.</p>', unsafe_allow_html=True)

    comparison_data = {
        "Feature": [
            "CPU Spike Prediction",
            "SHAP Explainable Alerts",
            "Natural Language Insights",
            "Carbon-Aware Scheduling",
            "Federated Learning",
            "Multi-Server Monitoring",
            "Price/month"
        ],
        "Datadog": ["❌", "❌", "❌", "❌", "❌", "✅", "$15/server"],
        "New Relic": ["❌", "❌", "❌", "❌", "❌", "✅", "$25/server"],
        "Google Vertex": ["✅", "⚠️ Basic", "❌", "❌", "✅ Enterprise", "✅", "$$$$"],
        "CrashGuard AI ✦": ["✅", "✅ Full + Unique", "✅ Unique", "✅ Unique", "✅", "✅", "₹5K–1L/org"]
    }
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    # ── ROI CALCULATOR ──
    st.divider()
    st.markdown('<p style="font-family:Syne,sans-serif; font-size:26px; font-weight:700; color:#fff; text-align:center; margin-bottom:8px;">ROI Calculator</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:DM Sans,sans-serif; font-size:14px; color:#555; text-align:center; margin-bottom:24px;">See exactly how much CrashGuard saves your company every month.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        crashes = st.slider("Crashes per month without prediction", 1, 30, 5)
        duration = st.slider("Avg crash duration (minutes)", 5, 120, 30)
        revenue = st.number_input("Revenue lost per minute (₹)", min_value=1000, max_value=5000000, value=50000, step=5000)
    with col2:
        total_loss = crashes * duration * revenue
        saved      = total_loss * 0.90
        net        = saved - 5000
        roi        = int(saved / 5000)
        st.markdown(f"""
        <div style="background:#0d0d1a; border:1px solid #1a1a2e; border-radius:16px; padding:32px;">
            <p style="font-family:DM Sans,sans-serif; font-size:13px; color:#555; margin:0 0 4px;">Monthly loss WITHOUT CrashGuard</p>
            <p style="font-family:Syne,sans-serif; font-size:34px; font-weight:800; color:#f87171; margin:0 0 20px;">₹{total_loss:,.0f}</p>
            <p style="font-family:DM Sans,sans-serif; font-size:13px; color:#555; margin:0 0 4px;">Monthly savings WITH CrashGuard Pro</p>
            <p style="font-family:Syne,sans-serif; font-size:34px; font-weight:800; color:#4ade80; margin:0 0 20px;">₹{saved:,.0f}</p>
            <div style="background:#4ade8010; border:1px solid #4ade8025; border-radius:10px; padding:16px;">
                <p style="font-family:DM Sans,sans-serif; font-size:14px; color:#4ade80; font-weight:600; margin:0 0 4px;">
                    {roi}x ROI — Net savings: ₹{net:,.0f}/month
                </p>
                <p style="font-family:DM Sans,sans-serif; font-size:12px; color:#555; margin:0;">
                    CrashGuard Pro costs only ₹5,000/month
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)


def show_current_plan():
    plan_key = st.session_state.get("current_plan", "free")
    plan     = PLANS[plan_key]
    user     = st.session_state.get("user_name", "User")
    color    = plan["color"]

    st.sidebar.markdown(f"""
    <div style="
        background:{color}10; border:1px solid {color}30;
        border-radius:10px; padding:12px 14px; margin-top:8px;
    ">
        <div style="font-family:DM Sans,sans-serif; font-size:11px; font-weight:600;
                    letter-spacing:1px; text-transform:uppercase; color:{color}; margin-bottom:4px;">
            {plan["name"]} Plan
        </div>
        <div style="font-family:DM Sans,sans-serif; font-size:13px; color:#ccc;">
            {user}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if plan_key == "free":
        st.sidebar.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        if st.sidebar.button("⬆ Upgrade to Pro", use_container_width=True, key="sidebar_upgrade_btn"):
            st.session_state["logged_in"]     = False
            st.session_state["show_pro_form"] = True
            st.rerun()