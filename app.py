import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Cafe Orders – Apriori Analysis",
    page_icon="☕",
    layout="wide"
)

st.title("☕ Frequent Itemset Mining in Cafe Orders")
st.write("Apriori Algorithm for discovering frequently ordered coffee combinations")

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("cafe_order_dataset.csv")
    df["coffee_name"] = df["coffee_name"].str.lower()
    return df

df = load_data()

# ---------------- Sidebar ----------------
st.sidebar.header("Apriori Parameters")

min_support = st.sidebar.slider(
    "Minimum Support",
    0.01, 0.1, 0.02, 0.01
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    0.1, 1.0, 0.4, 0.05
)

# ---------------- Transactions ----------------
transactions = (
    df.groupby(["date", "cash_type"])["coffee_name"]
    .apply(list)
    .tolist()
)

# ---------------- Encoding ----------------
te = TransactionEncoder()
encoded_data = te.fit(transactions).transform(transactions)
encoded_df = pd.DataFrame(encoded_data, columns=te.columns_)

# ---------------- Apriori ----------------
frequent_items = apriori(
    encoded_df,
    min_support=min_support,
    use_colnames=True
)

rules = association_rules(
    frequent_items,
    metric="confidence",
    min_threshold=min_confidence
)

# ---------------- Dataset Preview ----------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

# ---------------- Frequent Items ----------------
st.subheader("🔥 Frequent Itemsets")
st.dataframe(frequent_items.sort_values("support", ascending=False))

# ---------------- Association Rules ----------------
st.subheader("🔗 Association Rules")
rules_display = rules[["antecedents", "consequents", "support", "confidence", "lift"]]
st.dataframe(rules_display.sort_values("lift", ascending=False))

# ---------------- Bar Chart ----------------
st.subheader("📈 Top 10 Most Ordered Coffee Items")

top_items = df["coffee_name"].value_counts().head(10)

fig1, ax1 = plt.subplots()
top_items.plot(kind="barh", ax=ax1)
ax1.set_xlabel("Order Count")
ax1.set_ylabel("Coffee Type")
st.pyplot(fig1)

# ---------------- Network Graph ----------------
st.subheader("🕸️ Association Rules Network")

G = nx.DiGraph()

for _, row in rules.iterrows():
    for ant in row["antecedents"]:
        for con in row["consequents"]:
            G.add_edge(ant, con, weight=row["lift"])

fig2, ax2 = plt.subplots(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", ax=ax2)
st.pyplot(fig2)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Apriori Algorithm | Cafe Order Dataset | Streamlit ML App")
