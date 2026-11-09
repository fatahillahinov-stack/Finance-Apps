import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os

from dotenv import load_dotenv

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Budget vs. Actuals AI",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# 2. CONFIGURATION AND API KEY
# =========================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

try:
    if "OPENROUTER_API_KEY" in st.secrets:
        OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    pass

MODEL_NAME = "deepseek/deepseek-v4.1-flash"

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# =========================================================
# 3. OPENROUTER API FUNCTION
# =========================================================

def call_deepseek(messages, enable_reasoning=True):

    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY tidak ditemukan. "
            "Silakan konfigurasi Streamlit Secrets."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Budget vs. Actuals AI"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "reasoning": {
            "enabled": enable_reasoning
        },
        "max_tokens": 4000
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=180
        )

        if response.status_code != 200:
            try:
                error_data = response.json()
                error_message = error_data.get(
                    "error", {}
                ).get("message", response.text)
            except Exception:
                error_message = response.text

            raise RuntimeError(
                f"OpenRouter API Error "
                f"({response.status_code}): {error_message}"
            )

        result = response.json()

        if not result.get("choices"):
            raise RuntimeError(
                "API tidak mengembalikan pilihan jawaban."
            )

        assistant_message = result["choices"][0]["message"]

        answer = assistant_message.get("content")

        if not answer:
            raise RuntimeError(
                "Model tidak menghasilkan teks jawaban."
            )

        return answer, assistant_message

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Request timeout. Silakan coba kembali."
        )

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Tidak dapat terhubung ke OpenRouter."
        )


# =========================================================
# 4. INITIALIZE SESSION STATE
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

st.subheader(
    "Variance Analysis & Financial Commentary"
)

st.markdown(
    """
    Upload your Budget vs. Actuals dataset and generate
    financial insights using DeepSeek V4.1 Flash.
    """
)

st.caption(
    "AI Engine: DeepSeek V4.1 Flash | Provider: OpenRouter"
)

if not OPENROUTER_API_KEY:
    st.error(
        "OPENROUTER_API_KEY belum dikonfigurasi. "
        "Silakan tambahkan API Key pada Streamlit Secrets."
    )
    st.stop()


# =========================================================
# 6. FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload your financial dataset (Excel)",
    type=["xlsx"]
)


if uploaded_file:

    try:

        # =================================================
        # 7. READ AND VALIDATE DATA
        # =================================================

        df = pd.read_excel(uploaded_file)

        if df.empty:
            st.error("Dataset tidak memiliki data.")
            st.stop()

        required_columns = [
            "Category",
            "Budget",
            "Actual"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            st.error(
                f"Kolom berikut tidak ditemukan: "
                f"{', '.join(missing_columns)}"
            )
            st.stop()

        df = df.copy()

        df["Category"] = (
            df["Category"].fillna("Uncategorized").astype(str)
        )

        df["Budget"] = pd.to_numeric(
            df["Budget"], errors="coerce"
        )

        df["Actual"] = pd.to_numeric(
            df["Actual"], errors="coerce"
        )

        if df[["Budget", "Actual"]].isna().any().any():
            st.error(
                "Budget atau Actual mengandung nilai "
                "kosong atau bukan angka. Periksa dataset."
            )
            st.stop()

        # =================================================
        # 8. DATASET SIGNATURE AND CHAT RESET
        # =================================================

        signature = (
            uploaded_file.name,
            uploaded_file.size
        )

        if st.session_state.dataset_signature != signature:

            st.session_state.dataset_signature = signature
            st.session_state.chat_history = []
            st.session_state.analysis_result = None

        # =================================================
        # 9. VARIANCE CALCULATION
        # =================================================

        df["Variance"] = (
            df["Actual"] - df["Budget"]
        )

        df["Variance %"] = (
            df["Variance"]
            / df["Budget"].replace(0, float("nan"))
        ) * 100

        df["Variance Status"] = df["Variance"].apply(
            lambda x:
                "Overspending" if x > 0
                else "Underspending" if x < 0
                else "On Budget"
        )

        # =================================================
        # 10. AGGREGATED DATA
        # =================================================

        chart_df = (
            df.groupby("Category", as_index=False)
            [["Budget", "Actual", "Variance"]]
            .sum()
        )

        chart_df["Variance %"] = (
            chart_df["Variance"]
            / chart_df["Budget"].replace(0, float("nan"))
        ) * 100

        total_budget = df["Budget"].sum()
        total_actual = df["Actual"].sum()

        total_variance = (
            total_actual - total_budget
        )

        total_variance_pct = (
            (total_variance / total_budget) * 100
            if total_budget != 0 else None
        )

        # =================================================
        # 11. DATA PREVIEW
        # =================================================

        st.divider()

        st.subheader("📋 Financial Data Preview")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # 12. KPI DASHBOARD
        # =================================================

        st.subheader("📌 Financial Performance Indicators")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Budget",
            f"{total_budget:,.2f}"
        )

        col2.metric(
            "Total Actual",
            f"{total_actual:,.2f}"
        )

        col3.metric(
            "Total Variance",
            f"{total_variance:,.2f}",
            delta=(
                f"{total_variance_pct:.2f}%"
                if total_variance_pct is not None
                else None
            ),
            delta_color="inverse"
        )

        col4.metric(
            "Execution Rate",
            (
                f"{(total_actual / total_budget) * 100:.2f}%"
                if total_budget != 0 else "N/A"
            )
        )

        # =================================================
        # 13. VARIANCE ANALYSIS
        # =================================================

        st.divider()

        st.subheader("📈 Budget vs. Actual Analysis")

        col1, col2 = st.columns(2)

        with col1:

            fig_bar = px.bar(
                chart_df,
                x="Category",
                y="Variance",
                color="Variance",
                title="Budget Variance by Category",
                text_auto=".2s",
                color_continuous_scale=[
                    "red", "yellow", "green"
                ]
            )

            st.plotly_chart(
                fig_bar,
                use_container_width=True
            )

        with col2:

            fig_line = px.line(
                chart_df,
                x="Category",
                y=["Budget", "Actual"],
                markers=True,
                title="Budget vs. Actual Performance"
            )

            st.plotly_chart(
                fig_line,
                use_container_width=True
            )

        # =================================================
        # 14. FINANCIAL ANALYSIS CONTEXT
        # =================================================

        financial_summary = chart_df.to_string(
            index=False
        )

        system_prompt = """
        You are a senior financial analyst specializing
        in budgeting, financial management, and variance analysis.

        Analyze financial information accurately and objectively.

        Important rules:

        1. Base your analysis on the supplied dataset.
        2. Positive expenditure variance (Actual > Budget)
           indicates overspending.
        3. Negative expenditure variance (Actual < Budget)
           indicates underspending.
        4. Distinguish observed facts from possible explanations.
        5. Never invent financial figures or causes.
        6. Explain calculations when relevant.
        7. Provide actionable recommendations.
        8. Use professional financial and accounting terminology.
        9. Respond in the same language as the user's question.
        """

        data_context = f"""
        FINANCIAL DATA:

        {financial_summary}

        TOTAL BUDGET: {total_budget:,.2f}
        TOTAL ACTUAL: {total_actual:,.2f}
        TOTAL VARIANCE: {total_variance:,.2f}

        TOTAL VARIANCE PERCENTAGE:
        {total_variance_pct if total_variance_pct is not None else "N/A"}

        EXECUTION RATE:
        {
            (total_actual / total_budget) * 100
            if total_budget != 0 else "N/A"
        }
        """

        # =================================================
        # 15. AI-POWERED FINANCIAL ANALYSIS
        # =================================================

        st.divider()

        st.subheader("🤖 AI-Powered Financial Analysis")

        if st.button(
            "Generate AI Financial Analysis",
            type="primary",
            use_container_width=True
        ):

            analysis_prompt = f"""
            Perform a comprehensive financial variance analysis.

            {data_context}

            Structure your analysis as follows:

            1. Executive Summary
            2. Significant Budget Variances
            3. Categories with Overspending
            4. Categories with Underspending
            5. Possible Causes of Variances
            6. Financial Risks and Implications
            7. Recommendations

            Use tables where appropriate.

            Distinguish between facts established by the
            dataset and possible causes requiring further
            investigation.
            """

            with st.spinner(
                "DeepSeek V4.1 Flash is analyzing financial data..."
            ):

                try:

                    messages = [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": (
                                data_context + "\n\n" +
                                analysis_prompt
                            )
                        }
                    ]

                    answer, _ = call_deepseek(messages)

                    st.session_state.analysis_result = answer

                except Exception as e:

                    st.error(str(e))

        if st.session_state.analysis_result:

            st.markdown(
                st.session_state.analysis_result
            )

        # =================================================
        # 16. AI CHAT INTERFACE
        # =================================================

        st.divider()

        st.subheader(
            "🗣️ Chat with AI About Financial Performance"
        )

        st.caption(
            "Ask questions about budget execution, "
            "variance, financial risks, and recommendations."
        )

        # Display conversation history

        for message in st.session_state.chat_history:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

        # Chat input

        user_query = st.chat_input(
            "Ask DeepSeek about your financial data..."
        )

        if user_query:

            with st.chat_message("user"):

                st.markdown(user_query)

            with st.chat_message("assistant"):

                with st.spinner(
                    "DeepSeek is preparing your answer..."
                ):

                    try:

                        history = (
                            st.session_state.chat_history[-12:]
                        )

                        messages = [
                            {
                                "role": "system",
                                "content": (
                                    system_prompt +
                                    "\n\n" +
                                    data_context
                                )
                            }
                        ] + history + [
                            {
                                "role": "user",
                                "content": user_query
                            }
                        ]

                        answer, assistant_message = call_deepseek(
                            messages
                        )

                        st.markdown(answer)

                        # Preserve reasoning_details if returned.
                        stored_message = {
                            "role": "assistant",
                            "content": answer
                        }

                        if assistant_message.get(
                            "reasoning_details"
                        ):
                            stored_message[
                                "reasoning_details"
                            ] = assistant_message[
                                "reasoning_details"
                            ]

                        st.session_state.chat_history.extend([
                            {
                                "role": "user",
                                "content": user_query
                            },
                            stored_message
                        ])

                    except Exception as e:

                        st.error(str(e))

    except Exception as e:

        st.error(
            f"Error processing uploaded file: {str(e)}"
        )


# =========================================================
# 17. FOOTER
# =========================================================

st.divider()

st.caption(
    "Budget vs. Actuals AI | Powered by DeepSeek V4.1 Flash "
    "via OpenRouter"
)
