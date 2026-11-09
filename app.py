import os
import hashlib

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Budget vs. Actuals AI",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# 2. CONFIGURATION
# =========================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

try:
    if "OPENROUTER_API_KEY" in st.secrets:
        OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    # st.secrets may not exist when running locally.
    pass

MODEL_NAME = "deepseek/deepseek-v4.1-flash"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """
You are an expert Senior Financial Analyst, Management Accountant,
and Financial Controller.

Your task is to perform financial analysis, not merely summarize data.

ANALYTICAL FRAMEWORK

1. DATA UNDERSTANDING
   - Identify expense, revenue, profit, and subtotal accounts.
   - Identify unusual, negative, missing, or inconsistent values.
   - Do not assume every row is an independent transaction.

2. VARIANCE ANALYSIS
   - Analyze Budget versus Actual.
   - Identify material favorable and unfavorable variances.
   - Support findings with the relevant figures and percentages.
   - Explain the financial significance of each finding.

3. FINANCIAL PERFORMANCE
   - Assess budget execution.
   - Analyze revenue, expenses, and profitability separately.
   - Explain relationships between accounts only when supported
     by the supplied figures.

4. RISK AND ANOMALY IDENTIFICATION
   - Identify accounts requiring further investigation.
   - Distinguish unusual values from confirmed errors.
   - Do not label a value as fraudulent or erroneous without evidence.

5. ROOT-CAUSE ANALYSIS
   - Separate facts observed in the dataset from possible explanations.
   - Do not invent causes that cannot be established from the data.
   - State what additional information is needed to verify a hypothesis.

6. RECOMMENDATIONS
   - Give practical, prioritized recommendations.
   - Link each recommendation to an observed finding.
   - Explain what additional evidence should be reviewed.

IMPORTANT ACCOUNTING RULES

- For expense accounts, Actual greater than Budget generally indicates
  an unfavorable overspend, subject to the account's nature.
- For revenue accounts, Actual below Budget generally indicates
  an unfavorable shortfall.
- For profit accounts, interpret the movement according to the
  relevant performance measure.
- A positive or negative variance does not have the same meaning
  for every account type.
- Do not add detail accounts and their subtotals together.
- If account classifications are not provided, state the assumptions.
- Do not treat a raw sum of mixed detail and subtotal accounts as a
  reliable grand total without checking the account hierarchy.
- Identify missing currency, period, and accounting context.

RESPONSE STYLE

- Respond in the same language as the user's question.
- Answer the specific question first.
- Use professional accounting terminology.
- Use numerical evidence and tables where useful.
- Provide interpretation and implications, not just descriptions.
- Be precise, objective, and evidence-based.
"""


# =========================================================
# 3. OPENROUTER / DEEPSEEK FUNCTION
# =========================================================

def call_deepseek(messages, enable_reasoning=True):
    """Send chat messages to DeepSeek through OpenRouter."""

    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY tidak ditemukan. "
            "Tambahkan API Key pada Streamlit Secrets atau file .env."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Budget vs. Actuals AI",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "reasoning": {"enabled": enable_reasoning},
        "max_tokens": 4000,
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=180,
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "Request ke OpenRouter melewati batas waktu. "
            "Silakan coba kembali."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Tidak dapat terhubung ke OpenRouter. "
            "Periksa koneksi internet atau status layanan."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Kesalahan request OpenRouter: {exc}") from exc

    if response.status_code != 200:
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get(
                "message", response.text
            )
        except (ValueError, AttributeError):
            error_message = response.text

        raise RuntimeError(
            f"OpenRouter API Error ({response.status_code}): "
            f"{error_message}"
        )

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError("OpenRouter mengembalikan respons JSON tidak valid.") from exc

    choices = result.get("choices", [])
    if not choices:
        raise RuntimeError("API tidak mengembalikan pilihan jawaban.")

    assistant_message = choices[0].get("message", {})
    answer = assistant_message.get("content")

    if isinstance(answer, list):
        # Handle providers that return content as a list of text blocks.
        answer = "\n".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict)
        )

    if not answer or not str(answer).strip():
        raise RuntimeError(
            "Model tidak menghasilkan teks jawaban. "
            "Periksa respons model atau pengaturan token."
        )

    return str(answer), assistant_message


# =========================================================
# 4. SESSION STATE
# =========================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "dataset_signature" not in st.session_state:
    st.session_state.dataset_signature = None


# =========================================================
# 5. APPLICATION HEADER
# =========================================================

st.title("📊 Budget vs. Actuals AI")
st.subheader("Variance Analysis & Financial Commentary")

st.write(
    "Upload your Budget vs. Actuals dataset and generate "
    "financial insights using DeepSeek V4.1 Flash."
)

st.caption(
    "AI Engine: DeepSeek V4.1 Flash | Provider: OpenRouter"
)

if not OPENROUTER_API_KEY:
    st.error(
        "OPENROUTER_API_KEY belum dikonfigurasi. "
        "Tambahkan API Key pada Streamlit Secrets."
    )
    st.stop()


# =========================================================
# 6. FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload your financial dataset (Excel)",
    type=["xlsx"],
)


if uploaded_file is not None:
    try:
        # -------------------------------------------------
        # 7. READ AND VALIDATE DATA
        # -------------------------------------------------

        file_bytes = uploaded_file.getvalue()
        file_signature = hashlib.md5(file_bytes).hexdigest()

        df = pd.read_excel(uploaded_file)

        if df.empty:
            st.error("Dataset tidak memiliki data.")
            st.stop()

        required_columns = ["Category", "Budget", "Actual"]
        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            st.error(
                "Kolom berikut tidak ditemukan: "
                + ", ".join(missing_columns)
            )
            st.stop()

        df = df.copy()
        df["Category"] = (
            df["Category"].fillna("Uncategorized").astype(str)
        )
        df["Budget"] = pd.to_numeric(df["Budget"], errors="coerce")
        df["Actual"] = pd.to_numeric(df["Actual"], errors="coerce")

        if df[["Budget", "Actual"]].isna().any().any():
            st.error(
                "Kolom Budget atau Actual mengandung nilai kosong "
                "atau bukan angka. Periksa dataset."
            )
            st.stop()

        # Reset AI results when a different file is uploaded.
        if st.session_state.dataset_signature != file_signature:
            st.session_state.dataset_signature = file_signature
            st.session_state.chat_history = []
            st.session_state.analysis_result = None

        # -------------------------------------------------
        # 8. VARIANCE CALCULATION
        # -------------------------------------------------

        df["Variance"] = df["Actual"] - df["Budget"]

        df["Variance %"] = (
            df["Variance"]
            / df["Budget"].replace(0, float("nan"))
        ) * 100

        df["Variance Status"] = df["Variance"].apply(
            lambda value: (
                "Overspending" if value > 0
                else "Underspending" if value < 0
                else "On Budget"
            )
        )

        # -------------------------------------------------
        # 9. AGGREGATION AND KPI
        # -------------------------------------------------

        chart_df = (
            df.groupby("Category", as_index=False)[
                ["Budget", "Actual", "Variance"]
            ].sum()
        )

        chart_df["Variance %"] = (
            chart_df["Variance"]
            / chart_df["Budget"].replace(0, float("nan"))
        ) * 100

        total_budget = df["Budget"].sum()
        total_actual = df["Actual"].sum()
        total_variance = total_actual - total_budget

        total_variance_pct = (
            (total_variance / total_budget) * 100
            if total_budget != 0
            else None
        )

        execution_rate = (
            (total_actual / total_budget) * 100
            if total_budget != 0
            else None
        )

        # -------------------------------------------------
        # 10. DATA PREVIEW
        # -------------------------------------------------

        st.divider()
        st.subheader("📋 Financial Data Preview")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        # -------------------------------------------------
        # 11. KPI DASHBOARD
        # -------------------------------------------------

        st.subheader("📌 Financial Performance Indicators")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Budget", f"{total_budget:,.2f}")
        col2.metric("Total Actual", f"{total_actual:,.2f}")

        col3.metric(
            "Total Variance",
            f"{total_variance:,.2f}",
            delta=(
                f"{total_variance_pct:.2f}%"
                if total_variance_pct is not None
                else None
            ),
            delta_color="inverse",
        )

        col4.metric(
            "Execution Rate",
            f"{execution_rate:.2f}%"
            if execution_rate is not None
            else "N/A",
        )

        # -------------------------------------------------
        # 12. CHARTS
        # -------------------------------------------------

        st.divider()
        st.subheader("📈 Budget vs. Actual Analysis")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_bar = px.bar(
                chart_df,
                x="Category",
                y="Variance",
                color="Variance",
                title="Budget Variance by Category",
                text_auto=".2s",
                color_continuous_scale=["red", "yellow", "green"],
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            fig_line = px.line(
                chart_df,
                x="Category",
                y=["Budget", "Actual"],
                markers=True,
                title="Budget vs. Actual Performance",
            )
            st.plotly_chart(fig_line, use_container_width=True)

        # -------------------------------------------------
        # 13. FINANCIAL DATA CONTEXT
        # -------------------------------------------------

        financial_summary = chart_df.to_string(index=False)

        data_context = f"""
FINANCIAL DATA BY CATEGORY

{financial_summary}

TOTAL BUDGET (raw sum of uploaded rows): {total_budget:,.2f}
TOTAL ACTUAL (raw sum of uploaded rows): {total_actual:,.2f}
TOTAL VARIANCE: {total_variance:,.2f}
TOTAL VARIANCE PERCENTAGE:
{total_variance_pct if total_variance_pct is not None else "N/A"}

EXECUTION RATE:
{execution_rate if execution_rate is not None else "N/A"}%

IMPORTANT DATA LIMITATION:
The uploaded data may contain detail accounts, revenue accounts,
profit accounts, and subtotal accounts. The raw totals above are
simple sums of uploaded rows and may double-count values if detail
accounts and subtotals are both included. Verify the account hierarchy
before treating these totals as consolidated financial totals.
The currency and reporting period were not supplied unless stated
in the uploaded dataset.
"""

        # -------------------------------------------------
        # 14. AI-POWERED FINANCIAL ANALYSIS
        # -------------------------------------------------

        st.divider()
        st.subheader("🤖 AI-Powered Financial Analysis")

        if st.button(
            "Generate AI Financial Analysis",
            type="primary",
            use_container_width=True,
        ):
            analysis_prompt = f"""
Perform a comprehensive financial variance analysis using
the dataset below.

{data_context}

Present the analysis in this structure:

1. Executive Summary
2. Significant Budget Variances
3. Categories with Unfavorable Variances
4. Categories with Favorable Variances
5. Relationships between Revenue, Expenses, and Profit
6. Financial Risks and Issues Requiring Investigation
7. Recommendations and Follow-up Procedures

Do not merely repeat the data. Explain the financial meaning
of the most material findings. Distinguish verified facts from
possible explanations. Do not assume that all categories are
expense accounts.
"""

            with st.spinner(
                "DeepSeek V4.1 Flash is analyzing financial data..."
            ):
                try:
                    analysis_messages = [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": analysis_prompt,
                        },
                    ]

                    answer, _ = call_deepseek(analysis_messages)
                    st.session_state.analysis_result = answer

                except Exception as error:
                    st.error(f"AI Analysis Error: {error}")

        if st.session_state.analysis_result:
            st.markdown(st.session_state.analysis_result)

        # -------------------------------------------------
        # 15. AI CHAT
        # -------------------------------------------------

        st.divider()
        st.subheader("🗣️ Chat with AI About Financial Performance")

        st.caption(
            "Ask questions about budget execution, variance, "
            "financial risks, and recommendations."
        )

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_query = st.chat_input(
            "Ask DeepSeek about your financial data..."
        )

        if user_query:
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("DeepSeek is preparing your answer..."):
                    try:
                        history = st.session_state.chat_history[-12:]

                        chat_messages = [
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT,
                            },
                            {
                                "role": "system",
                                "content": data_context,
                            },
                        ] + history + [
                            {
                                "role": "user",
                                "content": user_query,
                            }
                        ]

                        answer, assistant_message = call_deepseek(
                            chat_messages
                        )

                        st.markdown(answer)

                        # Preserve provider reasoning metadata if supplied.
                        stored_assistant_message = {
                            "role": "assistant",
                            "content": answer,
                        }

                        reasoning_details = assistant_message.get(
                            "reasoning_details"
                        )
                        if reasoning_details is not None:
                            stored_assistant_message[
                                "reasoning_details"
                            ] = reasoning_details

                        st.session_state.chat_history.extend(
                            [
                                {
                                    "role": "user",
                                    "content": user_query,
                                },
                                stored_assistant_message,
                            ]
                        )

                    except Exception as error:
                        st.error(f"AI Chat Error: {error}")

    except Exception as error:
        st.error(f"Error processing uploaded file: {error}")


# =========================================================
# 16. FOOTER
# =========================================================

st.divider()
st.caption(
    "Budget vs. Actuals AI | Powered by DeepSeek V4.1 Flash "
    "via OpenRouter"
)
