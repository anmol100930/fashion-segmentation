# 👗 Fashion E-Commerce Customer Segmentation
### RFM Analysis + K-Means Clustering | Interactive Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-pink?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-hotpink?style=for-the-badge&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Dashboard-deeppink?style=for-the-badge&logo=plotly)
![Dash](https://img.shields.io/badge/Dash-Interactive-ff69b4?style=for-the-badge)

---

## 📌 Project Overview

This project applies **RFM Analysis** (Recency, Frequency, Monetary) combined with **K-Means Clustering** to segment customers of a fashion e-commerce store into meaningful groups. The goal is to help businesses like Myntra and Flipkart identify which customers to target, retain, or win back — and how.

An interactive **Plotly Dash dashboard** is included for real-time exploration of the segments with filters, charts, and a 3D cluster view.

---

## 🎯 Business Problem

> *"Not all customers are equal. How do we identify our most valuable customers, those at risk of leaving, and those we can grow?"*

By segmenting customers using RFM, businesses can:
- Focus marketing budgets on high-value customers
- Launch win-back campaigns for lost customers
- Onboard new customers with the right offers
- Maximize revenue per customer

---

## 🧠 What is RFM?

| Metric | Description |
|--------|-------------|
| **Recency** | How recently did the customer make a purchase? |
| **Frequency** | How many times have they purchased? |
| **Monetary** | How much total money have they spent? |

---

## 🔬 Methodology

```
Raw Data → Cleaning → RFM Feature Engineering → Log Normalization
→ StandardScaler → Elbow Method + Silhouette Score → K-Means (K=4)
→ Segment Labeling → Visualization → Interactive Dashboard
```

### Customer Segments Identified

| Segment | Description | Strategy |
|---------|-------------|----------|
| 💎 Loyal Champions | High spend, frequent, recent buyers | VIP program, early access |
| 🆕 New / Promising | Recent buyers, low frequency | Welcome discounts, onboarding |
| 🔄 Occasional Buyers | Moderate activity | Seasonal campaigns, bundles |
| 😴 Lost / At-Risk | Haven't bought in a long time | Win-back campaigns, strong offers |

---

## 📊 Dashboard Features

The interactive dashboard (`dashboard.py`) includes:

- **6 KPI Cards** — Total customers, revenue, avg order value, recency, loyal count, at-risk count
- **3 Filters** — Segment, Category, Country dropdowns
- **Monetary Range Slider** — Filter customers by spend
- **8 Interactive Charts:**
  - Donut chart — segment distribution
  - Scatter plot — frequency vs monetary
  - Bar chart — avg revenue per segment
  - Monthly revenue trend by segment
  - Revenue by product category
  - Box plot — recency distribution
  - **3D Cluster View** — rotate and explore clusters in 3D
  - RFM Heatmap — normalized scores per segment
- **Customer Table** — color-coded, paginated customer list

---

## 🗂️ Project Structure

```
fashion-segmentation/
│
├── customer_segmentation_rfm.py   # Main RFM + K-Means analysis script
├── dashboard.py                   # Interactive Plotly Dash dashboard
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/anmol100930/fashion-segmentation.git
cd fashion-segmentation
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the analysis
```bash
python customer_segmentation_rfm.py
```

### 5. Launch the dashboard
```bash
python dashboard.py
```
Then open **http://127.0.0.1:8050** in your browser.

---

## 📦 Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scikit-learn` | K-Means clustering, StandardScaler |
| `matplotlib` | Static charts |
| `seaborn` | Heatmap visualization |
| `plotly` | Interactive charts |
| `dash` | Web dashboard framework |
| `openpyxl` | Excel export |

---

## 📈 Output Files

After running `customer_segmentation_rfm.py`:

| File | Description |
|------|-------------|
| `rfm_distributions.png` | Recency, Frequency, Monetary histograms |
| `optimal_k.png` | Elbow method + Silhouette score chart |
| `segment_analysis.png` | Pie chart + bar chart of segments |
| `rfm_heatmap.png` | Normalized RFM heatmap |
| `customer_segments.xlsx` | Full customer list with segments |
| `segment_summary.xlsx` | Cluster profile summary |

---

## 💡 Real-World Applications

This type of RFM segmentation is used by:
- **Myntra** — personalized push notifications
- **Flipkart** — targeted ad spend optimization
- **Amazon** — product recommendation engine
- **Nykaa** — loyalty program design

---

## 👩‍💻 Author

**Anmol** — Data Science Enthusiast  
📧 anmol100930@gmail.com  
🔗 [github.com/anmol100930](https://github.com/anmol100930)

---

⭐ *If you found this project helpful, give it a star!*
