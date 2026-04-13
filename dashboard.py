import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
n = 5000
categories = ['Dresses', 'Tops', 'Footwear', 'Accessories', 'Jeans', 'Outerwear', 'Lingerie', 'Sportswear']
df = pd.DataFrame({
    'CustomerID': np.random.randint(1000, 6000, n),
    'InvoiceNo': [f'INV{i:05d}' for i in range(n)],
    'InvoiceDate': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, n), unit='D'),
    'Quantity': np.random.randint(1, 10, n),
    'UnitPrice': np.round(np.random.uniform(5, 200, n), 2),
    'Category': np.random.choice(categories, n),
    'Country': np.random.choice(['UK', 'France', 'Germany', 'India', 'USA', 'Australia'], n, p=[0.4,0.15,0.15,0.1,0.1,0.1])
})
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['Month'] = df['InvoiceDate'].dt.to_period('M').astype(str)

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalPrice', 'sum')
).reset_index()

rfm_scaled = StandardScaler().fit_transform(np.log1p(rfm[['Recency','Frequency','Monetary']]))
rfm['Cluster'] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(rfm_scaled)

cs = rfm.groupby('Cluster').agg(
    Avg_Recency=('Recency','mean'),
    Avg_Frequency=('Frequency','mean'),
    Avg_Monetary=('Monetary','mean')
).round(1)

segment_map = {
    cs['Avg_Monetary'].idxmax(): 'Loyal Champions',
    cs['Avg_Recency'].idxmin(): 'New / Promising',
    cs['Avg_Recency'].idxmax(): 'Lost / At-Risk',
}
for c in range(4):
    if c not in segment_map:
        segment_map[c] = 'Occasional Buyers'

rfm['Segment'] = rfm['Cluster'].map(segment_map)
df = df.merge(rfm[['CustomerID','Segment','Recency','Frequency','Monetary']], on='CustomerID', how='left')

COLORS = {
    'Loyal Champions':   '#E91E8C',
    'New / Promising':   '#FF6EC7',
    'Lost / At-Risk':    '#C2185B',
    'Occasional Buyers': '#F48FB1',
}
BG      = '#0d0d1a'
CARD_BG = '#1a1a2e'
BORDER  = '#3d1a4a'
PINK    = '#E91E8C'
LIGHT   = '#FF6EC7'
TEXT    = '#f8e8f5'
MUTED   = '#c084a8'

def card(children, style={}):
    base = {
        'backgroundColor': CARD_BG,
        'borderRadius': '16px',
        'padding': '20px',
        'border': f'1px solid {BORDER}',
    }
    base.update(style)
    return html.Div(children, style=base)

def kpi(label, value, color=PINK):
    return card([
        html.P(label, style={'color': MUTED, 'fontSize': '12px', 'margin': '0 0 6px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
        html.H2(value, style={'color': color, 'margin': '0', 'fontSize': '26px', 'fontWeight': '700'}),
    ], {'textAlign': 'center', 'flex': '1'})

chart_layout = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color=TEXT,
    font_family='Arial',
    margin=dict(t=40, b=20, l=20, r=20),
    legend=dict(bgcolor='rgba(0,0,0,0)', font_color=TEXT),
    colorway=[PINK, LIGHT, '#C2185B', '#F48FB1'],
)

app = Dash(__name__)
app.title = "Fashion Customer Segmentation"

app.layout = html.Div(style={'backgroundColor': BG, 'minHeight': '100vh', 'fontFamily': 'Arial', 'color': TEXT, 'padding': '0 0 40px'}, children=[

    # Header
    html.Div(style={
        'background': f'linear-gradient(135deg, #1a0a2e 0%, #2d0a3e 50%, #1a0a2e 100%)',
        'borderBottom': f'1px solid {BORDER}',
        'padding': '30px 40px',
        'marginBottom': '30px',
        'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'
    }, children=[
        html.Div([
            html.H1('Fashion E-Commerce', style={'color': PINK, 'margin': '0', 'fontSize': '26px', 'fontWeight': '700'}),
            html.P('Customer Segmentation Dashboard  •  RFM + K-Means', style={'color': MUTED, 'margin': '4px 0 0', 'fontSize': '13px'}),
        ]),
        html.Div([
            html.Span('LIVE', style={
                'backgroundColor': '#2d0a1e', 'color': PINK, 'fontSize': '11px',
                'padding': '4px 10px', 'borderRadius': '20px', 'border': f'1px solid {PINK}',
                'fontWeight': '700', 'letterSpacing': '2px'
            })
        ])
    ]),

    html.Div(style={'padding': '0 40px'}, children=[

        # KPI Row
        html.Div(style={'display': 'flex', 'gap': '16px', 'marginBottom': '24px'}, children=[
            kpi('Total Customers', f"{rfm.shape[0]:,}"),
            kpi('Total Revenue', f"£{rfm['Monetary'].sum():,.0f}", LIGHT),
            kpi('Avg Order Value', f"£{rfm['Monetary'].mean():,.0f}", '#FF6EC7'),
            kpi('Avg Recency', f"{rfm['Recency'].mean():.0f}d", '#F48FB1'),
            kpi('Loyal Champions', f"{(rfm['Segment']=='Loyal Champions').sum():,}", PINK),
            kpi('At-Risk Customers', f"{(rfm['Segment']=='Lost / At-Risk').sum():,}", '#C2185B'),
        ]),

        # Filters row
        card([
            html.Div(style={'display': 'flex', 'gap': '24px', 'alignItems': 'center', 'flexWrap': 'wrap'}, children=[
                html.Div([
                    html.Label('Segment', style={'color': MUTED, 'fontSize': '12px', 'display': 'block', 'marginBottom': '6px'}),
                    dcc.Dropdown(
                        id='seg-filter',
                        options=[{'label': 'All Segments', 'value': 'All'}] + [{'label': s, 'value': s} for s in rfm['Segment'].unique()],
                        value='All', clearable=False,
                        style={'width': '200px', 'backgroundColor': CARD_BG, 'color': '#000', 'border': f'1px solid {BORDER}'}
                    )
                ]),
                html.Div([
                    html.Label('Category', style={'color': MUTED, 'fontSize': '12px', 'display': 'block', 'marginBottom': '6px'}),
                    dcc.Dropdown(
                        id='cat-filter',
                        options=[{'label': 'All Categories', 'value': 'All'}] + [{'label': c, 'value': c} for c in categories],
                        value='All', clearable=False,
                        style={'width': '200px', 'backgroundColor': CARD_BG, 'color': '#000'}
                    )
                ]),
                html.Div([
                    html.Label('Country', style={'color': MUTED, 'fontSize': '12px', 'display': 'block', 'marginBottom': '6px'}),
                    dcc.Dropdown(
                        id='country-filter',
                        options=[{'label': 'All Countries', 'value': 'All'}] + [{'label': c, 'value': c} for c in df['Country'].unique()],
                        value='All', clearable=False,
                        style={'width': '180px', 'backgroundColor': CARD_BG, 'color': '#000'}
                    )
                ]),
                html.Div([
                    html.Label('Monetary Range (£)', style={'color': MUTED, 'fontSize': '12px', 'display': 'block', 'marginBottom': '6px'}),
                    dcc.RangeSlider(
                        id='monetary-slider',
                        min=0, max=int(rfm['Monetary'].max()), step=100,
                        value=[0, int(rfm['Monetary'].max())],
                        marks={0: {'label': '£0', 'style': {'color': MUTED}},
                               int(rfm['Monetary'].max()): {'label': f"£{int(rfm['Monetary'].max()):,}", 'style': {'color': MUTED}}},
                        tooltip={'placement': 'bottom', 'always_visible': False},
                    )
                ], style={'flex': '1', 'minWidth': '250px'}),
            ])
        ], {'marginBottom': '24px'}),

        # Row 1: Pie + Scatter + Bar
        html.Div(style={'display': 'flex', 'gap': '16px', 'marginBottom': '16px'}, children=[
            card([dcc.Graph(id='pie-chart', style={'height': '320px'})], {'flex': '1'}),
            card([dcc.Graph(id='scatter-chart', style={'height': '320px'})], {'flex': '2'}),
            card([dcc.Graph(id='bar-chart', style={'height': '320px'})], {'flex': '1'}),
        ]),

        # Row 2: Monthly trend + Category breakdown + Box plot
        html.Div(style={'display': 'flex', 'gap': '16px', 'marginBottom': '16px'}, children=[
            card([dcc.Graph(id='trend-chart', style={'height': '300px'})], {'flex': '2'}),
            card([dcc.Graph(id='category-chart', style={'height': '300px'})], {'flex': '1'}),
            card([dcc.Graph(id='box-chart', style={'height': '300px'})], {'flex': '1'}),
        ]),

        # Row 3: RFM 3D scatter + Heatmap
        html.Div(style={'display': 'flex', 'gap': '16px', 'marginBottom': '16px'}, children=[
            card([dcc.Graph(id='scatter3d', style={'height': '420px'})], {'flex': '1'}),
            card([dcc.Graph(id='heatmap', style={'height': '420px'})], {'flex': '1'}),
        ]),

        # Customer Table
        card([
            html.H3('Customer Details', style={'color': PINK, 'margin': '0 0 16px', 'fontSize': '16px'}),
            html.Div(id='customer-table')
        ]),

    ])
])


@app.callback(
    [Output('pie-chart','figure'), Output('scatter-chart','figure'),
     Output('bar-chart','figure'), Output('trend-chart','figure'),
     Output('category-chart','figure'), Output('box-chart','figure'),
     Output('scatter3d','figure'), Output('heatmap','figure'),
     Output('customer-table','children')],
    [Input('seg-filter','value'), Input('cat-filter','value'),
     Input('country-filter','value'), Input('monetary-slider','value')]
)
def update(seg, cat, country, money_range):
    fr = rfm.copy()
    fd = df.copy()

    if seg != 'All':
        fr = fr[fr['Segment'] == seg]
        fd = fd[fd['Segment'] == seg]
    if cat != 'All':
        fd = fd[fd['Category'] == cat]
        cids = fd['CustomerID'].unique()
        fr = fr[fr['CustomerID'].isin(cids)]
    if country != 'All':
        fd = fd[fd['Country'] == country]
        cids = fd['CustomerID'].unique()
        fr = fr[fr['CustomerID'].isin(cids)]

    fr = fr[(fr['Monetary'] >= money_range[0]) & (fr['Monetary'] <= money_range[1])]
    fd = fd[fd['CustomerID'].isin(fr['CustomerID'])]

    cl = dict(**chart_layout)

    # Pie
    pie = px.pie(fr, names='Segment', color='Segment', color_discrete_map=COLORS, hole=0.5,
                 title='Segment Distribution')
    pie.update_traces(textfont_color=TEXT)
    pie.update_layout(**cl, title_font_color=PINK)

    # Scatter
    scatter = px.scatter(fr, x='Frequency', y='Monetary', color='Segment',
                         color_discrete_map=COLORS, hover_data=['CustomerID','Recency'],
                         title='Frequency vs Monetary', opacity=0.65, size_max=10)
    scatter.update_layout(**cl, title_font_color=PINK)

    # Bar avg monetary
    bar_d = fr.groupby('Segment')['Monetary'].mean().reset_index()
    bar = px.bar(bar_d, x='Segment', y='Monetary', color='Segment',
                 color_discrete_map=COLORS, title='Avg Revenue / Segment')
    bar.update_layout(**cl, title_font_color=PINK, showlegend=False,
                      xaxis=dict(tickfont_color=MUTED), yaxis=dict(gridcolor='#2a0a3a'))

    # Monthly trend
    monthly = fd.groupby(['Month','Segment'])['TotalPrice'].sum().reset_index()
    trend = px.line(monthly, x='Month', y='TotalPrice', color='Segment',
                    color_discrete_map=COLORS, title='Monthly Revenue by Segment', markers=True)
    trend.update_layout(**cl, title_font_color=PINK,
                         xaxis=dict(tickangle=45, tickfont_color=MUTED),
                         yaxis=dict(gridcolor='#2a0a3a'))

    # Category bar
    cat_d = fd.groupby('Category')['TotalPrice'].sum().reset_index().sort_values('TotalPrice', ascending=True)
    cat_chart = px.bar(cat_d, x='TotalPrice', y='Category', orientation='h',
                       title='Revenue by Category', color='TotalPrice',
                       color_continuous_scale=[[0, '#2d0a3e'], [0.5, '#C2185B'], [1, PINK]])
    cat_chart.update_layout(**cl, title_font_color=PINK, coloraxis_showscale=False,
                             yaxis=dict(tickfont_color=MUTED), xaxis=dict(gridcolor='#2a0a3a'))

    # Box plot
    box = px.box(fr, x='Segment', y='Recency', color='Segment',
                 color_discrete_map=COLORS, title='Recency by Segment')
    box.update_layout(**cl, title_font_color=PINK, showlegend=False,
                      xaxis=dict(tickfont_color=MUTED), yaxis=dict(gridcolor='#2a0a3a'))

    # 3D Scatter
    s3d = px.scatter_3d(fr.sample(min(800, len(fr))), x='Recency', y='Frequency', z='Monetary',
                        color='Segment', color_discrete_map=COLORS, opacity=0.75,
                        title='3D RFM Cluster View')
    s3d.update_layout(**cl, title_font_color=PINK,
                      scene=dict(
                          bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='#3d1a4a', color=MUTED),
                          yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='#3d1a4a', color=MUTED),
                          zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='#3d1a4a', color=MUTED),
                      ))

    # Heatmap
    hm_data = fr.groupby('Segment')[['Recency','Frequency','Monetary']].mean().round(1)
    hm_norm = (hm_data - hm_data.min()) / (hm_data.max() - hm_data.min())
    heatmap = go.Figure(go.Heatmap(
        z=hm_norm.values, x=hm_norm.columns.tolist(), y=hm_norm.index.tolist(),
        colorscale=[[0,'#0d0d1a'],[0.5,'#C2185B'],[1,PINK]],
        text=hm_data.values, texttemplate='%{text}', showscale=False,
    ))
    heatmap.update_layout(**cl, title='RFM Heatmap by Segment', title_font_color=PINK,
                          xaxis=dict(tickfont_color=MUTED), yaxis=dict(tickfont_color=MUTED))

    # Table
    tbl_data = fr[['CustomerID','Segment','Recency','Frequency','Monetary']].head(20).round(1)
    table = dash_table.DataTable(
        data=tbl_data.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in tbl_data.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'backgroundColor': CARD_BG, 'color': TEXT, 'border': f'1px solid {BORDER}',
                    'textAlign': 'left', 'padding': '10px', 'fontSize': '13px'},
        style_header={'backgroundColor': '#2d0a3e', 'color': PINK, 'fontWeight': '700',
                      'border': f'1px solid {BORDER}'},
        style_data_conditional=[
            {'if': {'filter_query': '{Segment} = "Loyal Champions"'}, 'color': PINK},
            {'if': {'filter_query': '{Segment} = "Lost / At-Risk"'}, 'color': '#C2185B'},
            {'if': {'filter_query': '{Segment} = "New / Promising"'}, 'color': LIGHT},
        ],
        page_size=10,
    )

    return pie, scatter, bar, trend, cat_chart, box, s3d, heatmap, table


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Fashion Segmentation Dashboard")
    print("  Open: http://127.0.0.1:8050")
    print("="*50 + "\n")
    app.run(debug=False)
