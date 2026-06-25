import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration & Fluid Layout Activation
st.set_page_config(page_title="N-Flix Insights", page_icon="🎬", layout="wide")

# 2. Premium Black & Red Studio Layout (CSS Injection)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
    
    /* Global Base Reset */
    .stApp {
        background-color: #060608;
        color: #94A3B8;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Full-Screen Edge Width Utility */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 3.5rem !important;
        padding-right: 3.5rem !important;
        max-width: 100% !important;
    }
    
    /* Elegant Black/Red Studio Title Header */
    .studio-header {
        border-bottom: 1px solid #221215;
        padding-bottom: 20px;
        margin-bottom: 35px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .studio-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    .studio-title span {
        color: #E50914;
        font-weight: 700;
    }
    .live-indicator {
        background: rgba(229, 9, 20, 0.07);
        border: 1px solid rgba(229, 9, 20, 0.3);
        color: #E50914;
        padding: 5px 14px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Crisp Dark Studio KPI Metric Blocks */
    .metric-tile {
        background: #111115;
        border: 1px solid #1A1A22;
        border-radius: 8px;
        padding: 22px;
        transition: border-color 0.3s ease;
    }
    .metric-tile:hover {
        border-color: rgba(229, 9, 20, 0.4);
    }
    .metric-tile-label {
        color: #64748B;
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-tile-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 6px;
    }
    
    /* Navigation Tab Overrides - Premium Red Accents */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #1A1A22;
        margin-bottom: 25px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        color: #64748B !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #E50914 !important;
        border-bottom: 2px solid #E50914 !important;
    }
    
    /* Premium Grid Item Cards with Deep Red Focus Trim */
    .content-panel {
        background: #111115;
        border: 1px solid #1A1A22;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 16px;
    }
    .content-panel-focus {
        border-left: 4px solid #E50914;
        background: linear-gradient(90deg, #161113 0%, #111115 100%);
    }
    .panel-meta-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .panel-item-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
    }
    .confidence-tag {
        font-family: 'Space Grotesk', sans-serif;
        color: #F87171;
        font-weight: 700;
        font-size: 0.85rem;
        background: rgba(229, 9, 20, 0.1);
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid rgba(229, 9, 20, 0.2);
    }
    .pill-container {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        margin-top: 14px;
    }
    .custom-pill {
        background: #1A1A22;
        color: #94A3B8;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Studio Header Implementation
st.markdown("""
    <div class="studio-header">
        <div class="studio-title">N-Flix Insights</div>
        <div class="live-indicator">GLOBAL DATABASE ONLINE</div>
    </div>
""", unsafe_allow_html=True)

# 4. Data Infrastructure Pipeline (Weighted Attribute Engineering)
@st.cache_data
def process_data_source():
    data = pd.read_csv("netflix_titles.csv", encoding='latin1')
    data['rating'] = data['rating'].fillna('TV-MA')
    data['country'] = data['country'].fillna('United States')
    data['director'] = data['director'].fillna('')
    data['cast'] = data['cast'].fillna('')
    data['description'] = data['description'].fillna('')
    data['listed_in'] = data['listed_in'].fillna('')
    return data

df = process_data_source()

@st.cache_resource
def compute_isolated_matrices(_df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    genre_mat = vectorizer.fit_transform(_df['listed_in'])
    dir_mat = vectorizer.fit_transform(_df['director'])
    cast_mat = vectorizer.fit_transform(_df['cast'])
    
    return {
        "genre": cosine_similarity(genre_mat, genre_mat),
        "director": cosine_similarity(dir_mat, dir_mat),
        "cast": cosine_similarity(cast_mat, cast_mat)
    }

matrices = compute_isolated_matrices(df)

# 5. Fixed Summary KPI Blocks Row
kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.markdown(f'<div class="metric-tile"><div class="metric-tile-label">Catalog Size</div><div class="metric-tile-value">{len(df):,}</div></div>', unsafe_allow_html=True)
with kpi_cols[1]:
    st.markdown(f'<div class="metric-tile"><div class="metric-tile-label">Feature Length Films</div><div class="metric-tile-value">{len(df[df["type"] == "Movie"]):,}</div></div>', unsafe_allow_html=True)
with kpi_cols[2]:
    st.markdown(f'<div class="metric-tile"><div class="metric-tile-label">TV Show Series</div><div class="metric-tile-value">{len(df[df["type"] == "TV Show"]):,}</div></div>', unsafe_allow_html=True)
with kpi_cols[3]:
    regions_count = df['country'].str.split(', ').explode().nunique()
    st.markdown(f'<div class="metric-tile"><div class="metric-tile-label">Global Source Markets</div><div class="metric-tile-value">{regions_count}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 6. Navigation Tabs Layout Allocation
tab_telemetry, tab_engine = st.tabs(["GLOBAL DISTRIBUTION DATA", "MATCH INFERENCE MATRICES"])

with tab_telemetry:
    # Row 1 Charts
    layout_col1, layout_col2 = st.columns([1, 1])
    
    with layout_col1:
        type_matrix = df['type'].value_counts().reset_index()
        fig_donut = go.Figure(data=[go.Pie(labels=type_matrix['type'], values=type_matrix['count'], hole=.68,
                                          marker=dict(colors=['#E50914', '#1A1A22']), textinfo='percent')])
        fig_donut.update_layout(
            title="Catalog Core Volume Breakdown", title_font_color='#FFFFFF', title_font_size=13,
            paper_bgcolor='rgba(0,0,0,0)', font_color='#64748B',
            showlegend=True, legend=dict(font=dict(color='#94A3B8')), margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with layout_col2:
        timeline = df.groupby(['release_year', 'type']).size().reset_index(name='volume')
        timeline = timeline[timeline['release_year'] >= 2012]
        fig_line = px.line(timeline, x='release_year', y='volume', color='type',
                           color_discrete_map={'Movie': '#E50914', 'TV Show': '#5A1216'})
        fig_line.update_layout(
            title="Longitudinal Release Tracking (Post-2012)", title_font_color='#FFFFFF', title_font_size=13,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#64748B', xaxis_showgrid=False, yaxis_showgrid=True,
            yaxis_gridcolor='#1A1A22', margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
    # Row 2 Chart
    st.markdown("<br>", unsafe_allow_html=True)
    global_distribution = df['country'].str.split(', ').explode().value_counts().head(10).reset_index()
    fig_global = px.bar(global_distribution, x='country', y='count',
                        title="Top 10 Global Production Origin Markets",
                        color_discrete_sequence=['#E50914'])
    fig_global.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#64748B', title_font_color='#FFFFFF', title_font_size=13,
        xaxis_showgrid=False, yaxis_showgrid=True, yaxis_gridcolor='#1A1A22',
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig_global, use_container_width=True)

with tab_engine:
    engine_col1, engine_col2 = st.columns([1, 2])
    
    with engine_col1:
        st.markdown("<div style='color:#FFFFFF; font-size:0.85rem; font-weight:600; margin-bottom:10px;'>SEARCH BASELINE ITEM</div>", unsafe_allow_html=True)
        title_options = list(df['title'].sort_values().values)
        
        # Enforce empty baseline selection point on initialization
        selected_title = st.selectbox(
            "Choose baseline catalog item:", 
            options=[""] + title_options, 
            index=0,
            label_visibility="collapsed"
        )
        
        if selected_title:
            asset_idx = df[df['title'] == selected_title].index[0]
            target = df.iloc[asset_idx]
            
            st.markdown(f"""
                <div class="content-panel content-panel-focus" style="margin-top:15px;">
                    <div style="color:#64748B; font-size:0.7rem; font-weight:700; text-transform:uppercase;">Selected Profile</div>
                    <div style="font-size:1.25rem; font-weight:700; color:#FFFFFF; margin:4px 0 10px 0;">{target['title']}</div>
                    <p style="color:#94A3B8; font-size:0.85rem; line-height:1.5; margin:0;">{target['description']}</p>
                    <div class="pill-container">
                        <span class="custom-pill" style="background:rgba(229, 9, 20, 0.15); color:#E50914; font-weight:600;">{target['type']}</span>
                        <span class="custom-pill">{target['rating']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="border: 1px dashed #221215; border-radius:8px; padding:35px; text-align:center; margin-top:15px; color:#64748B; font-size:0.85rem;">
                    Awaiting query baseline selection to initialize attribute matching.
                </div>
            """, unsafe_allow_html=True)
            
    with engine_col2:
        st.markdown("<div style='color:#FFFFFF; font-size:0.85rem; font-weight:600; margin-bottom:12px;'>HIGH-ACCURACY MATCH PROFILES & DISTRIBUTION LOCATIONS</div>", unsafe_allow_html=True)
        
        if selected_title:
            # Linear Attribute Weighting Execution (Genres: 40%, Directors: 40%, Cast: 20%)
            genre_scores = matrices['genre'][asset_idx]
            director_scores = matrices['director'][asset_idx]
            cast_scores = matrices['cast'][asset_idx]
            composite_scores = (0.4 * genre_scores) + (0.4 * director_scores) + (0.2 * cast_scores)
            
            scores_list = list(enumerate(composite_scores))
            sorted_scores = sorted(scores_list, key=lambda x: x[1], reverse=True)[1:4] # Slice top 3
            
            for index, score in sorted_scores:
                match_row = df.iloc[index]
                match_confidence = int(score * 100)
                primary_market = match_row['country'].split(',')[0]
                
                st.markdown(f"""
                    <div class="content-panel">
                        <div class="panel-meta-row">
                            <div class="card-title" style="font-size:1.1rem; font-weight:700; color:#FFF;">{match_row['title']}</div>
                            <span class="confidence-tag">{match_confidence}% Score Match</span>
                        </div>
                        <p style="color:#94A3B8; font-size:0.85rem; line-height:1.5; margin:0 0 12px 0;">{match_row['description']}</p>
                        <div style="border-top:1px solid #1A1A22; padding-top:10px; display:flex; justify-content:between; align-items:center;">
                            <div style="font-size:0.78rem; color:#64748B;">
                                Primary Distribution Region: <span style="color:#E50914; font-weight:600;">{primary_market}</span>
                            </div>
                            <div class="pill-container" style="margin-top:0; margin-left:auto;">
                                <span class="custom-pill">{match_row['type']}</span>
                                <span class="custom-pill">{match_row['rating']}</span>
                                <span class="custom-pill">{match_row['release_year']}</span>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="border: 1px dashed #1A1A22; border-radius:8px; padding:35px; text-align:center; color:#64748B; font-size:0.85rem;">
                    Select an entry item within the left workspace dropdown to populate calculation nodes.
                </div>
            """, unsafe_allow_html=True)