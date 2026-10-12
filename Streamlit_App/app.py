# ============================================================
# AI-Based Student Academic Performance Prediction System
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import joblib

from chatbot import student_chat

APP_DIR = Path(__file__).parent

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Load CSS
# -----------------------------

def load_css():
    css_path = APP_DIR / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("style.css not found — running with default Streamlit styling.")

load_css()

# -----------------------------
# Load Machine Learning Model (cached so it isn't reloaded on every rerun)
# -----------------------------

@st.cache_resource
def load_model():
    model_path = APP_DIR / ".." / "Model" / "student_model.pkl"
    return joblib.load(model_path)

try:
    model = load_model()
    model_error = None
except Exception as e:
    model = None
    model_error = str(e)

# -----------------------------
# Chart theme (dark background)
# -----------------------------

CHART_FONT = {"family": "Inter", "color": "#93A8C7"}
ACCENT = "#3B82F6"
MAROON = "#1E3A8A"
SUCCESS = "#3DDC97"
AMBER = "#F5A524"
RED = "#F5646B"
GRID = "rgba(255,255,255,0.08)"
INK = "#EAF2FF"

# -----------------------------
# Helpers
# -----------------------------

def grade_for(score):
    if score >= 90:
        return "A+", "Outstanding", "good"
    elif score >= 80:
        return "A", "Excellent", "good"
    elif score >= 70:
        return "B", "Very Good", "good"
    elif score >= 60:
        return "C", "Good", "mid"
    elif score >= 50:
        return "D", "Average", "mid"
    else:
        return "F", "Needs Improvement", "low"

def tag(label, kind):
    return f'<span class="tag {kind}">{label}</span>'

def data_row(label, value):
    return f'<div class="data-row"><span class="d-label">{label}</span><span class="d-value">{value}</span></div>'

def score_color(score):
    if score >= 70:
        return SUCCESS
    elif score >= 50:
        return AMBER
    else:
        return RED

def section_label(icon, title, sub=None):
    sub_html = f'<div class="label-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="section-label">
        <div class="icon-badge">{icon}</div>
        <div><div class="label-text">{title}</div>{sub_html}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HERO
# ============================================================

today = datetime.now()

st.markdown(f"""
<div class="hero">
    <div class="main-title">Predict student outcomes<br><span class="accent">before results day</span></div>
    <div class="sub-title">Enter a student's profile and get an instant, data-driven forecast of their final academic score — powered by a trained machine learning model.</div>
</div>
""", unsafe_allow_html=True)

if model_error:
    st.error(
        "The prediction model could not be loaded, so predictions are disabled. "
        f"Expected it at `Model/student_model.pkl` relative to the app's parent folder. "
        f"Details: {model_error}"
    )

st.write("")

# ============================================================
# INPUT SECTION
# ============================================================

col1, col2, col3 = st.columns([1, 1.25, 1], gap="medium")

with col1:
    with st.container(border=True):
        section_label("👤", "Student Profile", "Who we're predicting for")

        student_name = st.text_input("Student Name", placeholder="e.g. Aditi Sharma")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 15, 25, 18)

with col2:
    with st.container(border=True):
        section_label("📚", "Academic Details", "Effort & performance signals")

        assignments = st.slider("Assignments Submitted (out of 10)", 0, 10, 5)
        internal_marks = st.slider("Internal Marks (out of 10)", 0, 10, 5)
        study_hours = st.slider("Self Study Hours (per day)", 0, 10, 3)
        failures = st.number_input("Previous Failures", min_value=0, max_value=10, value=0)
        extra_classes = st.selectbox("Extra Classes Attended", ["Yes", "No"])

with col3:
    with st.container(border=True):
        section_label("🗓️", "Attendance", "Classroom presence")

        total_classes = st.number_input("Total Classes Conducted", min_value=1, value=100)
        attended_classes = st.number_input(
            "Classes Attended", min_value=0, max_value=int(total_classes), value=90
        )
        attendance = (attended_classes / total_classes) * 100

        st.metric("Attendance Percentage", f"{attendance:.1f}%")
        st.progress(attendance / 100)

gender_value = 1 if gender == "Male" else 0
extra_classes_value = 1 if extra_classes == "Yes" else 0

# ============================================================
# LIVE SNAPSHOT — inline preview strip (replaces the old sidebar)
# ============================================================

snapshot_tip = ""
if attendance < 75:
    snapshot_tip = '<div class="preview-tip warn">⚠️ Attendance is below 75% — this will weigh on the forecast.</div>'
elif study_hours < 3:
    snapshot_tip = '<div class="preview-tip warn">📖 Low study hours logged — consider raising this input.</div>'
elif failures > 0:
    snapshot_tip = '<div class="preview-tip warn">❌ Previous failures on record — this lowers the projected score.</div>'
else:
    snapshot_tip = '<div class="preview-tip good">✅ Profile looks strong across the board.</div>'

st.markdown(f"""
<div class="snapshot-bar">
    <div class="preview-header"><span class="preview-pulse"></span>Live Snapshot</div>
    <div class="snapshot-grid">
        <div class="snapshot-item">
            <div class="snap-label">Student</div>
            <div class="snap-value">{student_name.strip() if student_name.strip() else "Unnamed"} · {gender}, {age}</div>
        </div>
        <div class="snapshot-item">
            <div class="snap-label">Attendance</div>
            <div class="snap-value">{attendance:.0f}%</div>
        </div>
        <div class="snapshot-item">
            <div class="snap-label">Assignments</div>
            <div class="snap-value">{assignments}/10</div>
        </div>
        <div class="snapshot-item">
            <div class="snap-label">Study Hours</div>
            <div class="snap-value">{study_hours} hrs/day</div>
        </div>
    </div>
    {snapshot_tip}
</div>
""", unsafe_allow_html=True)

# ============================================================
# PREDICT BUTTON
# ============================================================

st.write("")
predict = st.button("Predict Final Score →", use_container_width=True, disabled=model is None)

if predict and not student_name.strip():
    st.warning("Add the student's name above so the report card can be personalised.")

# ============================================================
# PREDICTION SECTION
# ============================================================

if predict and model is not None:

    input_data = pd.DataFrame({
        "Gender": [gender_value],
        "Age": [age],
        "Assignments_Submitted": [assignments],
        "Internal_Marks": [internal_marks],
        "Attendance_Percentage": [attendance],
        "Self_Study_Hours": [study_hours],
        "Previous_Failures": [failures],
        "Extra_Classes": [extra_classes_value]
    })

    try:
        raw_prediction = model.predict(input_data)[0]
    except Exception as e:
        st.error(f"The model could not generate a prediction: {e}")
        st.stop()

    prediction = max(0.0, min(100.0, float(raw_prediction)))
    grade, performance, tone = grade_for(prediction)
    ring_color = score_color(prediction)

    # Persist this student's profile + prediction in session_state so the
    # floating AI Assistant widget can still access it on the next rerun —
    # Streamlit reruns the whole script on every interaction, so a plain
    # local variable here would be lost the moment the person opens the
    # chat or sends a message. Keys match exactly what chatbot.py expects.
    st.session_state["student_data"] = {
        "name": student_name if student_name.strip() else "the student",
        "gender": gender,
        "age": age,
        "assignments": assignments,
        "internal_marks": internal_marks,
        "attendance": attendance,
        "study_hours": study_hours,
        "failures": failures,
        "extra_classes": extra_classes,
        "prediction": prediction,
    }

    st.markdown('<div class="reveal">', unsafe_allow_html=True)
    st.divider()

    # ---- Score ring + performance breakdown, side by side ----

    result_col, breakdown_col = st.columns([1, 1.3], gap="large")

    with result_col:
        with st.container(border=True):
            section_label("📊", "Prediction Result")

            donut = go.Figure(go.Pie(
                values=[prediction, 100 - prediction],
                hole=0.78,
                marker=dict(colors=[ring_color, "rgba(255,255,255,0.08)"], line=dict(color="#050B14", width=2)),
                textinfo="none",
                sort=False,
                direction="clockwise"
            ))
            donut.update_layout(
                showlegend=False,
                height=190,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                annotations=[dict(
                    text=f"<b>{prediction:.0f}</b><span style='font-size:13px;color:{CHART_FONT['color']}'>/100</span>",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(family="Sora", size=30, color=INK)
                )]
            )
            st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})

            st.markdown(f"""
            <div style="text-align:center;">
                <span class="score-badge {tone}">Grade {grade} · {performance}</span>
                <p style="margin-top:14px;color:{CHART_FONT['color']};font-size:13.5px;line-height:1.6;">
                    Projected final score: <b style="color:{INK};">{prediction:.1f} / 100</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

    with breakdown_col:
        with st.container(border=True):
            section_label("📈", "Performance Breakdown", "Each factor, scored out of 100")

            categories = ["Assignments", "Internal Marks", "Attendance", "Study Hours", "Discipline"]
            values = [
                assignments * 10,
                internal_marks * 10,
                round(attendance, 1),
                min(study_hours, 10) * 10,
                max(0, 10 - failures) * 10
            ]
            bar_colors = [score_color(v) for v in values]

            bar = go.Figure(go.Bar(
                x=values,
                y=categories,
                orientation="h",
                marker=dict(color=bar_colors, cornerradius=6),
                text=[f"{v:.0f}" for v in values],
                textposition="outside",
                textfont=dict(color=INK, family="Inter", size=13)
            ))
            bar.update_layout(
                height=280,
                margin=dict(l=10, r=30, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[0, 112], showgrid=True, gridcolor=GRID, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, autorange="reversed", color=CHART_FONT["color"]),
                font=CHART_FONT,
                bargap=0.35
            )
            st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

    if prediction >= 85:
        st.balloons()

    st.divider()

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    section_label("💡", "Recommendations", "What would move the needle most")

    recommendations = []
    if attendance < 75:
        recommendations.append("📅 Improve attendance above 75%.")
    if assignments < 7:
        recommendations.append("📚 Submit all assignments on time.")
    if study_hours < 3:
        recommendations.append("📖 Increase self-study to at least 3 hours daily.")
    if internal_marks < 6:
        recommendations.append("📝 Focus more on internal assessments.")
    if failures > 0:
        recommendations.append("❌ Revise previous subjects to avoid backlogs.")
    if extra_classes == "No":
        recommendations.append("🎯 Attend extra classes for better understanding.")

    if not recommendations:
        st.markdown(
            '<div class="rec-item ok">🎉 Excellent! This academic profile is strong — keep maintaining this consistency.</div>',
            unsafe_allow_html=True
        )
    else:
        for item in recommendations:
            st.markdown(f'<div class="rec-item">{item}</div>', unsafe_allow_html=True)

    st.divider()

    # --------------------------------------------------------
    # Summary + Insights
    # --------------------------------------------------------

    summary_col, insight_col = st.columns(2, gap="large")

    with summary_col:
        with st.container(border=True):
            section_label("🧾", "Student Summary")
            st.markdown(
                data_row("Student Name", student_name if student_name else "—") +
                data_row("Gender", gender) +
                data_row("Age", age) +
                data_row("Attendance", f"{attendance:.1f}%") +
                data_row("Assignments", f"{assignments}/10") +
                data_row("Internal Marks", f"{internal_marks}/10") +
                data_row("Study Hours", f"{study_hours} hrs/day") +
                data_row("Previous Failures", failures) +
                data_row("Extra Classes", extra_classes) +
                data_row("Predicted Score", f"{prediction:.1f}/100") +
                data_row("Grade", grade),
                unsafe_allow_html=True
            )

    with insight_col:
        with st.container(border=True):
            section_label("🏷️", "Quick Insights")

            tags_html = ""
            if attendance >= 85:
                tags_html += tag("✓ Excellent Attendance", "good")
            elif attendance >= 75:
                tags_html += tag("Good Attendance", "warn")
            else:
                tags_html += tag("⚠ Improve Attendance", "bad")

            if assignments >= 8:
                tags_html += tag("✓ Strong Assignment Record", "good")
            else:
                tags_html += tag("Submit More Assignments", "warn")

            if study_hours >= 4:
                tags_html += tag("✓ Good Study Habits", "good")
            else:
                tags_html += tag("Increase Self Study", "warn")

            if failures == 0:
                tags_html += tag("✓ No Previous Failures", "good")
            else:
                tags_html += tag("Clear Previous Backlogs", "bad")

            st.markdown(tags_html, unsafe_allow_html=True)

    st.divider()

    # --------------------------------------------------------
    # Report download
    # --------------------------------------------------------

    section_label("📥", "Downloadable Report")

    report = f"""========================================================
        STUDENT PERFORMANCE REPORT
========================================================
Student Name : {student_name if student_name else '-'}
Gender       : {gender}
Age          : {age}
Attendance   : {attendance:.1f} %
Assignments  : {assignments}/10
Internal     : {internal_marks}/10
Study Hours  : {study_hours} hrs/day
Failures     : {failures}
Extra Classes: {extra_classes}
--------------------------------------------------------
Predicted Score : {prediction:.1f}/100
Grade            : {grade}
Performance      : {performance}
========================================================
Generated on {today.strftime('%d %B %Y')}
"""

    st.download_button(
        label="Download Report",
        data=report,
        file_name=f"{(student_name or 'student').strip().replace(' ', '_')}_Report.txt",
        mime="text/plain"
    )

    st.success("Prediction completed successfully.")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FLOATING AI ACADEMIC ASSISTANT — Gemini-powered chat widget
# ============================================================
#
# A circular toggle button fixed to the bottom-right corner opens a
# slide-in glass panel. Both are built with st.container(key=...), which
# stamps a stable CSS class (st-key-<key>) onto the wrapping div — that's
# what style.css hooks into to pin them to the viewport. Requires
# Streamlit >= 1.32 (for container `key` and `height`).
#
# Chat history + open/closed state persist in session_state, and
# student_data (captured above when a prediction was run) is what gets
# sent to Gemini as context for every answer.

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

has_profile = "student_data" in st.session_state

# ---- Floating toggle button ----

with st.container(key="fab_container"):
    fab_icon = "✕" if st.session_state.chat_open else "💬"
    if st.button(fab_icon, key="fab_button", help="Ask the AI Academic Assistant"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()

    if not st.session_state.chat_open and has_profile and not st.session_state.chat_history:
        st.markdown('<div class="fab-ping"></div>', unsafe_allow_html=True)

# ---- Slide-in chat panel ----

if st.session_state.chat_open:
    with st.container(key="chat_panel"):

        st.markdown("""
        <div class="chat-panel-header">
            <div class="chat-panel-avatar">🤖</div>
            <div class="chat-panel-heading">
                <div class="chat-panel-name">AI Academic Assistant</div>
                <div class="chat-panel-status"><span class="chat-dot"></span>Powered by Gemini</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not has_profile:
            st.markdown(
                '<div class="chat-empty">Run a prediction above first — '
                'I use that student\'s profile to answer your questions. 📊</div>',
                unsafe_allow_html=True
            )
        else:
            with st.container(key="chat_messages", height=300):
                if not st.session_state.chat_history:
                    sd = st.session_state["student_data"]
                    st.markdown(
                        f'<div class="chat-empty">Hi! I can see {sd["name"]}\'s predicted score '
                        f'is <b>{sd["prediction"]:.1f}/100</b>. Ask me anything about it — '
                        'like how to improve, or what\'s holding the score back. 👋</div>',
                        unsafe_allow_html=True
                    )
                for msg in st.session_state.chat_history:
                    role_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
                    st.markdown(f'<div class="chat-bubble {role_class}">{msg["text"]}</div>', unsafe_allow_html=True)

            with st.form(key="chat_form", clear_on_submit=True):
                form_cols = st.columns([5, 1])
                with form_cols[0]:
                    user_question = st.text_input(
                        "Ask a question",
                        placeholder="How can I improve my score?",
                        label_visibility="collapsed",
                    )
                with form_cols[1]:
                    submitted = st.form_submit_button("➤", use_container_width=True)

            if submitted and user_question.strip():
                question = user_question.strip()
                st.session_state.chat_history.append({"role": "user", "text": question})
                try:
                    with st.spinner("Thinking..."):
                        answer = student_chat(question, st.session_state["student_data"])
                    st.session_state.chat_history.append({"role": "assistant", "text": answer})
                except Exception as e:
                    st.session_state.chat_history.append(
                        {"role": "assistant", "text": f"⚠️ I couldn't respond: {e}"}
                    )
                st.rerun()

            if st.session_state.chat_history:
                with st.container(key="chat_clear_container"):
                    if st.button("Clear conversation", key="chat_clear"):
                        st.session_state.chat_history = []
                        st.rerun()

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown('<div class="footer-note">PREDICTIVE ACADEMIC INTELLIGENCE · POWERED BY MACHINE LEARNING</div>', unsafe_allow_html=True)
