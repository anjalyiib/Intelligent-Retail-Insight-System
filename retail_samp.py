import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from mlxtend.frequent_patterns import fpgrowth, association_rules
from huggingface_hub import InferenceClient


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Intelligent Retail Insight System",
    layout="wide"
)

st.title("🛒 Intelligent Retail Insight System")


# --------------------------------------------------
# HUGGING FACE CLIENT
# --------------------------------------------------

HF_TOKEN = st.secrets["HF_TOKEN"]

@st.cache_resource
def load_client():
    return InferenceClient(token=HF_TOKEN)

client = load_client()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "chat" not in st.session_state:
    st.session_state.chat = []


# --------------------------------------------------
# SIDEBAR DATA UPLOAD
# --------------------------------------------------

st.sidebar.title("📂 Upload Retail Dataset")

file = st.sidebar.file_uploader(
    "Upload CSV file",
    type="csv"
)

if file is None:
    st.info("⬅ Upload a retail dataset to begin")
    st.stop()


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data(file):

    df = pd.read_csv(file)

    required_cols = [
        "order_id",
        "product_name",
        "user_id",
        "Quantity",
        "UnitPrice",
        "order_date"
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["user_id"] = df["user_id"].astype(int)
    df["Quantity"] = df["Quantity"].astype(int)
    df["UnitPrice"] = df["UnitPrice"].astype(float)

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    df["TotalSales"] = df["Quantity"] * df["UnitPrice"]

    return df


df = load_data(file)


# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("💰 Total Sales", f"{df['TotalSales'].sum():,.2f}")
c2.metric("👥 Customers", df["user_id"].nunique())
c3.metric("🧾 Orders", df["order_id"].nunique())
c4.metric("🏆 Top Product", df["product_name"].value_counts().idxmax())

st.divider()


# --------------------------------------------------
# HELPER FUNCTION
# --------------------------------------------------

@st.cache_data
def create_basket(data):

    basket = (
        data.groupby(["order_id", "product_name"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )

    return basket > 0

# --------------------------------------------------
# CHATBOT
# --------------------------------------------------

st.sidebar.title("🤖 Retail AI Assistant")

query = st.sidebar.text_area("Ask retail questions")

ask = st.sidebar.button("Analyze")

clear = st.sidebar.button("Clear History")

if clear:
    st.session_state.chat = []

if ask and query:

    # Dataset insights
    total_sales = df["TotalSales"].sum()
    total_customers = df["user_id"].nunique()
    total_orders = df["order_id"].nunique()

    top_products = df["product_name"].value_counts().head(5)

    # small dataset preview
    sample_data = df.head(5).to_string()

    # context sent to AI
    dataset_context = f"""
Retail Dataset Summary

Total Sales: {total_sales}
Total Customers: {total_customers}
Total Orders: {total_orders}

Top Selling Products:
{top_products.to_string()}

Dataset Sample:
{sample_data}
"""

    with st.spinner("Analyzing dataset..."):

        completion = client.chat.completions.create(

            model="meta-llama/Meta-Llama-3-8B-Instruct",

            messages=[

                {
                    "role": "system",
                    "content":
                    """
You are a retail business analyst.

Answer questions using the dataset insights provided.
Explain results clearly in simple human language.

Rules:
- Do NOT write Python code or SQL queries.
- Use numbers from the dataset summary if available.
- Provide business insights like a retail consultant.
- Keep answers short and clear.
"""
                },

                {
                    "role": "user",
                    "content":
                    dataset_context + "\n\nUser Question:\n" + query
                }

            ],

            temperature=0.2,
            max_tokens=200
        )

        response = completion.choices[0].message["content"]

        st.sidebar.success(response)

        st.session_state.chat.append((query, response))


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📊 Market Basket",
    "👥 Customer Segmentation",
    "📈 Demand Forecast"
])


# --------------------------------------------------
# MARKET BASKET
# --------------------------------------------------

with tab1:

    st.subheader("📊 Market Basket Analysis")

    filtered_df = df[
        df["product_name"].map(df["product_name"].value_counts()) > 50
    ]

    basket = create_basket(filtered_df)

    min_support = max(0.002, 3 / basket.shape[0])

    item_support = basket.mean()

    basket = basket.loc[:, item_support >= min_support]

    freq = fpgrowth(
        basket,
        min_support=min_support,
        use_colnames=True,
        max_len=2
    )

    rules = association_rules(
        freq,
        metric="confidence",
        min_threshold=0.2
    )

    rules = rules[rules["lift"] > 1.2]

    if rules.empty:

        st.warning("No strong associations found.")

    else:

        rules["Product"] = rules["antecedents"].apply(
            lambda x: ", ".join(x)
        )

        rules["Recommended"] = rules["consequents"].apply(
            lambda x: ", ".join(x)
        )

        rules["Probability (%)"] = (rules["confidence"] * 100).round(2)

        final_rules = rules[
            ["Product", "Recommended", "Probability (%)"]
        ].sort_values(
            "Probability (%)",
            ascending=False
        ).head(15)

        st.dataframe(final_rules, use_container_width=True)


# --------------------------------------------------
# CUSTOMER SEGMENTATION
# --------------------------------------------------

with tab2:

    st.subheader("👥 Customer Segmentation")

    cust_df = df.groupby("user_id").agg({

        "Quantity": "sum",
        "TotalSales": "sum"

    })

    scaler = StandardScaler()

    scaled = scaler.fit_transform(cust_df)

    k = st.slider("Number of Segments", 2, 5, 3)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(scaled)

    cust_df["Cluster"] = clusters
    
    cluster_summary = cust_df.groupby("Cluster").mean().sort_values("TotalSales")

    labels = [
        "Low Value Customers",
        "Regular Customers",
        "High Value Customers",
        "Premium Customers",
        "Elite Customers"
    ]

    cust_df["Segment"] = cust_df["Cluster"].map(
        {c: labels[i] for i, c in enumerate(cluster_summary.index)}
    )

    # Remove cluster column
    cust_df = cust_df.drop(columns=["Cluster"])

    result = cust_df.groupby("Segment").mean().round(2)

    st.dataframe(result, use_container_width=True)
# --------------------------------------------------
# DEMAND FORECAST
# --------------------------------------------------

with tab3:

    st.subheader("📈 Product Demand Forecast")

    demand_df = (
        df.groupby(["product_name", "order_date"])["Quantity"]
        .sum()
        .reset_index()
        .sort_values("order_date")
    )

    top_products = (
        demand_df.groupby("product_name")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(30)
        .index
    )

    product = st.selectbox("Select Product", top_products)

    product_demand = demand_df[
        demand_df["product_name"] == product
    ].copy()

    product_demand["date_ordinal"] = product_demand["order_date"].map(
        pd.Timestamp.toordinal
    )

    model = LinearRegression()

    model.fit(
        product_demand[["date_ordinal"]],
        product_demand["Quantity"]
    )

    future_days = 30

    last_date = product_demand["order_date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=future_days
    )

    future_ord = future_dates.map(pd.Timestamp.toordinal)

    forecast = model.predict(
        future_ord.values.reshape(-1, 1)
    )

    forecast_df = pd.DataFrame({

        "order_date": future_dates,
        "Forecast Demand": forecast

    })

    # Limit history for better chart balance
    recent_history = product_demand.tail(60)

    actual_df = recent_history[[
        "order_date",
        "Quantity"
    ]].rename(columns={
        "Quantity": "Actual Demand"
    })

    chart_df = pd.merge(
        actual_df,
        forecast_df,
        on="order_date",
        how="outer"
    ).set_index("order_date")

    st.line_chart(chart_df)

    st.caption(
        "Last 60 days demand with 30-day forecast"
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.caption(
    "AI-powered Retail Analytics Dashboard for business insights."
)