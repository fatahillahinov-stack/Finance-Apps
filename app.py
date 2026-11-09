
import streamlit as st
import pandas as pd
import plotly.express as px
import os

from google import genai
from dotenv import load_dotenv

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Budget vs. Actuals AI",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# 2. LOAD GEMINI API KEY
# =====================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Prioritaskan Streamlit Secrets jika tersedia
try:
    if "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not GEMINI_API_KEY:
    st.error(
        "🚨 Gemini API Key is missing! "
        "Set GEMINI_API_KEY in Streamlit Secrets or .env file."
    )
    st.stop()

# =====================================================
# 3. INITIALIZE GEMINI CLIENT
# =====================================================

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash"

# =====================================================
# 4. STREAMLIT APP UI
# =====================================================

st.title("📊 Budget vs. Actuals AI")
st.subheader("Variance Analysis & Financial Commentary")

st.write(
    "Upload your Budget vs. Actuals file "
    "and get AI-driven financial insights powered by Google Gemini."
)

# =====================================================
# 5. FILE UPLOADER
# =====================================================

uploaded_file = st.file_uploader(
    "📂 Upload your dataset (Excel format)",
    type=["xlsx"]
)

if uploaded_file:

    try:
        df = pd.read_excel(uploaded_file)

        required_columns = ["Category", "Budget", "Actual"]

        if not all(col in df.columns for col in required_columns):
            st.error(
                "The uploaded file must contain "
                "'Category', 'Budget', and 'Actual' columns!"
            )
            st.stop()

        # Convert numeric columns
        df["Budget"] = pd.to_numeric(df["Budget"], errors="coerce")
        df["Actual"] = pd.to_numeric(df["Actual"], errors="coerce")

        if df[["Budget", "Actual"]].isna().any().any():
            st.error("Budget and Actual columns contain invalid numeric values.")
            st.stop()

        # =================================================
        # 6. VARIANCE CALCULATION
        # =================================================

        df["Variance"] = df["Actual"] - df["Budget"]

        df["Variance %"] = (
            df["Variance"] / df["Budget"].replace(0, float("nan"))
        ) * 100

        # =================================================
        # 7. DATA PREVIEW
        # =================================================

        st.subheader("📊 Data Preview with Variance Calculation")

        st.dataframe(df, use_container_width=True)

        # =================================================
        # 8. KEY PERFORMANCE INDICATORS
        # =================================================

        total_budget = df["Budget"].sum()
        total_actual = df["Actual"].sum()
        total_variance = total_actual - total_budget

        col1, col2, col3 = st.columns(3)

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
            f"{total_variance:,.2f}"
        )

        # =================================================
        # 9. VISUALIZATION
        # =================================================

        st.subheader("📈 Budget vs. Actual Variance Analysis")

        fig_bar = px.bar(
            df,
            x="Category",
            y="Variance",
            color="Variance",
            title="Variance by Category",
            text_auto=".2s",
            color_continuous_scale=["red", "yellow", "green"]
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        fig_line = px.line(
            df,
            x="Category",
            y=["Budget", "Actual"],
            markers=True,
            title="Budget vs. Actual Performance"
        )

        st.plotly_chart(fig_line, use_container_width=True)

        # =================================================
        # 10. AI-POWERED FINANCIAL ANALYSIS
        # =================================================

        st.subheader("🤖 AI-Powered Variance Analysis")

        if st.button("Generate AI Financial Analysis"):

            with st.spinner("Gemini is analyzing your financial data..."):

                financial_summary = df.to_string(index=False)

                prompt = f"""
                You are an expert financial analyst specializing
                in budget and actual expenditure analysis.

                Analyze the following financial data:

                {financial_summary}

                Total Budget: {total_budget:,.2f}
                Total Actual: {total_actual:,.2f}
                Total Variance: {total_variance:,.2f}

                Provide your analysis in the following structure:

                1. Executive Summary
                2. Significant Budget Variances
                3. Categories with Overspending
                4. Categories with Underspending
                5. Possible Causes of Variances
                6. Financial Risks
                7. Recommendations

                Distinguish between observed data and possible causes.
                Do not invent explanations that cannot be established
                from the supplied data.
                """

                try:
response = client.interactions.create(
    model="gemini-3.8-flash",
    input=prompt
)

st.markdown(response.output_text)

                except Exception as e:
                    st.error(f"Gemini API Error: {str(e)}")

        # =================================================
        # 11. AI CHAT
        # =================================================

        st.subheader("🗣️ Chat with AI About Variance Analysis")

        user_query = st.text_input(
            "🔍 Ask the AI about your variance data:"
        )

        if st.button("Ask Gemini") and user_query:

            with st.spinner("Gemini is preparing your answer..."):

                prompt = f"""
                You are an AI financial analyst.

                Analyze the following Budget vs Actual data:

                {df.to_string(index=False)}

                User question:
                {user_query}

                Answer based on the provided financial data.
                Explain calculations where relevant.
                Do not invent missing information.
                """

                try:
                    chat_response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=prompt
                    )

                    st.markdown(chat_response.text)

                except Exception as e:
                    st.error(f"Gemini API Error: {str(e)}")

    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
