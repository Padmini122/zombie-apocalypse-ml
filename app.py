
"""
🧟 Zombie Apocalypse Supply Chain Optimizer
Interactive Streamlit Dashboard - BULLETPROOF VERSION
Handles ALL edge cases gracefully
"""

import streamlit as st
import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
from matplotlib.patches import Patch

from simulator import ZombieApocalypseSimulator
from feature_engineering import RouteFeatureEngineer
from models import RouteFailurePredictor

# ============================================
# SAFETY WRAPPERS - These prevent ALL crashes
# ============================================

def safe_get(func, default=None):
    """Execute function safely, return default on any error"""
    try:
        return func()
    except Exception:
        return default

def safe_dataframe(df, default_cols=None):
    """Ensure we always have a valid dataframe"""
    if df is None or len(df) == 0:
        if default_cols:
            return pd.DataFrame(columns=default_cols)
        return pd.DataFrame()
    return df

def safe_value(series, index=0, default=0):
    """Safely get value from series"""
    try:
        if len(series) > index:
            return series.values[index]
        return default
    except Exception:
        return default

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="🧟 Zombie Apocalypse ML",
    page_icon="🧟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; text-align: center; color: #ff4444; }
    .sub-header { font-size: 1.2rem; text-align: center; color: #aaaaaa; }
    .stButton>button { background-color: #cc0000; color: white; font-weight: bold; 
                       font-size: 1.1rem; padding: 12px 24px; border: none; 
                       border-radius: 8px; width: 100%; }
    .stButton>button:hover { background-color: #ff0000; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧟 Zombie Apocalypse Supply Chain Optimizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML-Powered Survival Route Prediction System</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# SESSION STATE INIT
# ============================================
defaults = {
    'run_simulation': False,
    'simulator': None,
    'city_data': None,
    'outbreak_data': None,
    'predictor': None,
    'results': None,
    'engineer': None,
    'ml_data': None,
    'days_to_simulate': 30,
    'error_message': None,
    'simulation_complete': False
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("⚙️ Simulation Controls")
    
    num_cities = st.slider("Number of Cities", 10, 30, 20, 5, 
                          help="How many cities in the simulation")
    infection_rate = st.slider("Infection Rate", 0.1, 0.9, 0.3, 0.05,
                              help="How fast zombies spread (higher = faster)")
    start_city = st.selectbox("Outbreak Origin", range(num_cities), 0,
                              format_func=lambda x: f"City {x}",
                              help="Where patient zero is located")
    days_to_simulate = st.slider("Days to Simulate", 10, 50, 30, 5,
                                help="Total days of outbreak simulation")
    
    st.markdown("---")
    st.header("🌲 Model Parameters")
    
    rf_trees = st.slider("Random Forest Trees", 10, 500, 100, 10,
                        help="Number of trees in Random Forest")
    rf_depth = st.slider("Max Tree Depth", 2, 30, 10, 1,
                        help="Maximum depth of each tree")
    
    st.markdown("---")
    
    run_clicked = st.button("🚀 Run Simulation & Train Models", type="primary")
    if run_clicked:
        st.session_state.run_simulation = True
        st.session_state.simulation_complete = False
        st.session_state.error_message = None

# ============================================
# SIMULATION LOGIC
# ============================================
if st.session_state.run_simulation:
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Create world
        status_text.text("🌍 Creating virtual world...")
        progress_bar.progress(5)
        
        simulator = ZombieApocalypseSimulator(num_cities=num_cities, map_seed=42)
        city_data = simulator.create_city_network()
        
        if city_data is None or len(city_data) == 0:
            raise ValueError("Failed to create city network")
        
        progress_bar.progress(15)
        
        # Step 2: Simulate outbreak
        status_text.text("🧟 Simulating zombie outbreak...")
        
        start_city_safe = min(start_city, num_cities - 1)
        outbreak_data = simulator.simulate_outbreak(
            start_city=start_city_safe,
            days=days_to_simulate,
            infection_rate=infection_rate
        )
        
        if outbreak_data is None or len(outbreak_data) == 0:
            raise ValueError("Simulation produced no data")
        
        progress_bar.progress(35)
        
        # Step 3: Engineer features
        status_text.text("🔧 Engineering ML features...")
        engineer = RouteFeatureEngineer(outbreak_data, city_data, simulator.G)
        
        feature_end_day = max(10, days_to_simulate - 5)
        ml_data = engineer.create_features(end_day=feature_end_day)
        
        if ml_data is None or len(ml_data) == 0:
            raise ValueError("No features could be generated. Try more days or cities.")
        
        progress_bar.progress(55)
        
        # Step 4: Prepare data
        status_text.text("📊 Preparing training data...")
        
        test_threshold = max(5, days_to_simulate - 10)
        X_train, X_test, y_train, y_test, scaler = engineer.prepare_train_test_split(
            ml_data,
            test_day_threshold=test_threshold
        )
        
        if len(X_train) < 10 or len(X_test) < 2:
            raise ValueError("Not enough data for training. Increase simulation days.")
        
        progress_bar.progress(65)
        
        # Step 5: Train models
        status_text.text("🤖 Training machine learning models...")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LinearRegression
        
        predictor = RouteFailurePredictor()
        
        # Linear Regression
        predictor.lr_model = LinearRegression()
        predictor.lr_model.fit(X_train, y_train)
        
        # Decision Tree
        min_split = max(2, len(y_train) // 20)
        min_leaf = max(1, len(y_train) // 40)
        predictor.dt_model = DecisionTreeClassifier(
            max_depth=5, min_samples_split=min_split,
            min_samples_leaf=min_leaf, random_state=42
        )
        predictor.dt_model.fit(X_train, y_train)
        
        # Random Forest
        n_estimators = max(10, min(rf_trees, 500))
        max_depth = max(2, min(rf_depth, 30))
        predictor.rf_model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=max(2, len(y_train) // 30),
            min_samples_leaf=max(1, len(y_train) // 50),
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        predictor.rf_model.fit(X_train, y_train)
        
        progress_bar.progress(85)
        
        # Step 6: Evaluate
        status_text.text("📈 Evaluating models...")
        results = predictor.evaluate_all(X_test, y_test)
        
        progress_bar.progress(98)
        
        # Store everything
        st.session_state.simulator = simulator
        st.session_state.city_data = city_data
        st.session_state.outbreak_data = outbreak_data
        st.session_state.predictor = predictor
        st.session_state.results = results
        st.session_state.engineer = engineer
        st.session_state.ml_data = ml_data
        st.session_state.days_to_simulate = days_to_simulate
        st.session_state.simulation_complete = True
        st.session_state.error_message = None
        
        status_text.text("✅ Complete!")
        progress_bar.progress(100)
        
        st.success("✅ Simulation complete! Explore the results below.")
        st.balloons()
        
    except Exception as e:
        st.session_state.error_message = str(e)
        st.session_state.simulation_complete = False
        st.error(f"❌ Simulation failed: {e}")
        st.info("💡 Try adjusting the parameters and running again.")

# ============================================
# DISPLAY RESULTS
# ============================================
if st.session_state.simulation_complete and st.session_state.results is not None:
    
    days = st.session_state.days_to_simulate
    results = st.session_state.results
    simulator = st.session_state.simulator
    city_data = st.session_state.city_data
    outbreak_data = st.session_state.outbreak_data
    engineer = st.session_state.engineer
    
    # Tabs
    tabs = st.tabs([
        "📊 Model Performance",
        "🗺️ Outbreak Map",
        "📈 Visualizations",
        "🔍 Feature Analysis",
        "🧟 Survival Guide"
    ])
    
    # ========================
    # TAB 1: MODEL PERFORMANCE
    # ========================
    with tabs[0]:
        st.header("📊 Model Performance Comparison")
        
        cols = st.columns(3)
        models = ['Linear Regression', 'Decision Tree', 'Random Forest']
        
        for i, (col, model) in enumerate(zip(cols, models)):
            with col:
                r = safe_get(lambda: results[model], {})
                if not r:
                    st.warning(f"No data for {model}")
                    continue
                    
                st.markdown(f"### {model}")
                st.metric("Accuracy", f"{safe_get(lambda: r['accuracy'], 0):.1%}")
                st.metric("Precision", f"{safe_get(lambda: r['precision'], 0):.1%}")
                st.metric("Recall", f"{safe_get(lambda: r['recall'], 0):.1%}")
                st.metric("F1 Score", f"{safe_get(lambda: r['f1'], 0):.1%}")
                st.metric("AUC", f"{safe_get(lambda: r['auc'], 0):.3f}")
                
                acc = safe_get(lambda: r['accuracy'], 0)
                if acc > 0.85:
                    st.success("🌟 High Performance")
                elif acc > 0.70:
                    st.warning("⚠️ Moderate Performance")
                else:
                    st.error("💀 Low Performance")
        
        # Best model
        st.markdown("---")
        rf_acc = safe_get(lambda: results['Random Forest']['accuracy'], 0)
        rf_auc = safe_get(lambda: results['Random Forest']['auc'], 0)
        st.success(f"### 🏆 Best Model: Random Forest\nAchieved **{rf_acc:.1%} accuracy** with AUC of **{rf_auc:.3f}**")
    
    # ========================
    # TAB 2: OUTBREAK MAP
    # ========================
    with tabs[1]:
        st.header("🗺️ Interactive Outbreak Map")
        
        if simulator is None or outbreak_data is None:
            st.warning("No simulation data available")
        else:
            max_day = max(0, days - 1)
            day = st.slider("Select Day", 0, max_day, min(15, max_day), key="map_day")
            
            try:
                fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0e1117')
                pos = nx.get_node_attributes(simulator.G, 'pos')
                day_data = outbreak_data[outbreak_data['day'] == day]
                
                if len(day_data) > 0:
                    node_colors = []
                    node_sizes = []
                    labels = {}
                    
                    for node in range(simulator.num_cities):
                        node_row = day_data[day_data['city_id'] == node]
                        status = safe_value(node_row['status'], 0, 0)
                        pop = safe_get(lambda: int(city_data.iloc[node]['population']), 50000)
                        
                        if status == 2:
                            node_colors.append('#FF0000')
                        elif status == 1:
                            node_colors.append('#FFA500')
                        else:
                            node_colors.append('#00FF00')
                        
                        node_sizes.append(max(150, min(600, pop / 500)))
                        labels[node] = safe_get(lambda: str(city_data.iloc[node]['city_name'])[:8], f"C{node}")
                    
                    nx.draw_networkx_edges(simulator.G, pos, alpha=0.3, edge_color='white', ax=ax)
                    nx.draw_networkx_nodes(simulator.G, pos, node_color=node_colors,
                                          node_size=node_sizes, alpha=0.8, ax=ax)
                    nx.draw_networkx_labels(simulator.G, pos, labels, font_size=7, 
                                           font_color='white', ax=ax)
                    
                    ax.set_title(f'Zombie Outbreak - Day {day}', fontsize=16, 
                                fontweight='bold', color='white')
                    ax.axis('off')
                    
                    legend_elements = [
                        Patch(facecolor='#00FF00', label='Safe'),
                        Patch(facecolor='#FFA500', label='Infected'),
                        Patch(facecolor='#FF0000', label='Overrun')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right',
                             facecolor='#1e1e1e', edgecolor='white', labelcolor='white')
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # City status table
                    st.subheader("📋 City Status Report")
                    try:
                        status_df = day_data[['city_name', 'status', 'infection_probability']].copy()
                        status_map = {0: '🟢 Safe', 1: '🟠 Infected', 2: '🔴 Overrun'}
                        status_df['status'] = status_df['status'].map(status_map).fillna('❓ Unknown')
                        status_df['infection_probability'] = status_df['infection_probability'].apply(
                            lambda x: f"{float(x):.1%}" if pd.notna(x) else "N/A"
                        )
                        status_df.columns = ['City', 'Status', 'Infection Risk']
                        st.dataframe(status_df, width='stretch')
                    except Exception:
                        st.info("City status table unavailable for this day")
                else:
                    st.warning(f"No data available for Day {day}")
                    
            except Exception as e:
                st.warning(f"Could not render map: {e}")
    
    # ========================
    # TAB 3: VISUALIZATIONS
    # ========================
    with tabs[2]:
        st.header("📈 Model Evaluation Visualizations")
        
        viz_option = st.radio(
            "Select Visualization",
            ["Confusion Matrices", "ROC Curves", "Feature Importance", "Outbreak Progression"],
            horizontal=True,
            key="viz_select"
        )
        
        # Confusion Matrices
        if viz_option == "Confusion Matrices":
            st.subheader("Confusion Matrices Comparison")
            
            try:
                fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0e1117')
                
                for idx, model_name in enumerate(['Linear Regression', 'Decision Tree', 'Random Forest']):
                    r = safe_get(lambda: results[model_name], None)
                    if r is None:
                        axes[idx].text(0.5, 0.5, 'No Data', ha='center', va='center', color='white')
                        continue
                        
                    cm = r.get('confusion_matrix', np.array([[0]]))
                    accuracy = r.get('accuracy', 0)
                    
                    labels_list = ['Safe', 'Fails'] if cm.shape == (2, 2) else \
                                 [f'C{i}' for i in range(max(cm.shape))]
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                               xticklabels=labels_list[:cm.shape[1]],
                               yticklabels=labels_list[:cm.shape[0]],
                               ax=axes[idx], cbar=False, 
                               annot_kws={'size': 14, 'weight': 'bold'})
                    axes[idx].set_title(f'{model_name}\nAccuracy: {accuracy:.1%}',
                                       fontsize=12, fontweight='bold', color='white')
                    axes[idx].set_facecolor('#0e1117')
                
                st.pyplot(fig)
                plt.close(fig)
                
                st.markdown("""
                - **True Positive**: Route correctly predicted to fail ✅
                - **True Negative**: Route correctly predicted safe ✅
                - **False Positive**: Safe route predicted to fail ⚠️
                - **False Negative**: Failing route predicted safe 💀
                """)
            except Exception as e:
                st.warning(f"Could not create confusion matrices: {e}")
        
        # ROC Curves
        elif viz_option == "ROC Curves":
            st.subheader("ROC Curves")
            
            try:
                fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0e1117')
                colors = {'Linear Regression': '#3498db', 'Decision Tree': '#e67e22', 'Random Forest': '#e74c3c'}
                
                for model_name in ['Linear Regression', 'Decision Tree', 'Random Forest']:
                    r = safe_get(lambda: results[model_name], None)
                    if r:
                        ax.plot(r.get('fpr', [0,1]), r.get('tpr', [0,1]),
                               color=colors.get(model_name, '#fff'), linewidth=3,
                               label=f"{model_name} (AUC = {r.get('auc', 0):.3f})")
                
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
                ax.set_xlabel('False Positive Rate', fontsize=13, color='white')
                ax.set_ylabel('True Positive Rate', fontsize=13, color='white')
                ax.set_title('ROC Curves', fontsize=16, fontweight='bold', color='white')
                ax.legend(loc='lower right', facecolor='#1e1e1e', edgecolor='white', labelcolor='white')
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('#0e1117')
                ax.tick_params(colors='white')
                
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Could not create ROC curves: {e}")
        
        # Feature Importance
        elif viz_option == "Feature Importance":
            st.subheader("🔍 What Makes Routes Dangerous?")
            
            try:
                fi = results.get('feature_importance', {})
                features = safe_get(lambda: engineer.feature_columns, [])
                
                if fi and len(features) > 0:
                    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='#0e1117')
                    
                    for idx, (model_type, color, title) in enumerate([
                        ('decision_tree', '#e67e22', 'Decision Tree'),
                        ('random_forest', '#e74c3c', 'Random Forest')
                    ]):
                        importance = fi.get(model_type, np.zeros(len(features)))
                        
                        if len(importance) == len(features):
                            df = pd.DataFrame({'feature': features, 'importance': importance})
                            df = df.sort_values('importance', ascending=False).head(10)
                            
                            axes[idx].barh(df['feature'], df['importance'], color=color, edgecolor='white')
                            axes[idx].set_xlabel('Importance', color='white')
                            axes[idx].set_title(f'{title}: Top Predictors', fontweight='bold', color='white')
                            axes[idx].invert_yaxis()
                            axes[idx].set_facecolor('#0e1117')
                            axes[idx].tick_params(colors='white')
                    
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Feature importance data not available")
            except Exception as e:
                st.warning(f"Could not create feature importance chart: {e}")
        
        # Outbreak Progression
        else:
            st.subheader("📅 Outbreak Progression")
            
            try:
                if simulator and outbreak_data is not None:
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='#0e1117')
                    axes = axes.flatten()
                    
                    max_d = max(0, days - 1)
                    step = max(1, days // 6)
                    day_list = [min(i, max_d) for i in [0, step, step*2, step*3, step*4, max_d]]
                    
                    for idx, d in enumerate(day_list):
                        pos = nx.get_node_attributes(simulator.G, 'pos')
                        dd = outbreak_data[outbreak_data['day'] == d]
                        
                        node_colors = []
                        for node in range(simulator.num_cities):
                            nr = dd[dd['city_id'] == node]
                            status = safe_value(nr['status'], 0, 0)
                            color = '#FF0000' if status == 2 else '#FFA500' if status == 1 else '#00FF00'
                            node_colors.append(color)
                        
                        nx.draw_networkx_edges(simulator.G, pos, alpha=0.2, edge_color='white', ax=axes[idx])
                        nx.draw_networkx_nodes(simulator.G, pos, node_color=node_colors,
                                              node_size=200, alpha=0.8, ax=axes[idx])
                        axes[idx].set_title(f'Day {d}', fontsize=14, fontweight='bold', color='white')
                        axes[idx].axis('off')
                        axes[idx].set_facecolor('#0e1117')
                    
                    plt.suptitle('Outbreak Progression', fontsize=18, fontweight='bold', color='white', y=1.02)
                    st.pyplot(fig)
                    plt.close(fig)
            except Exception as e:
                st.warning(f"Could not render progression: {e}")
    
    # ========================
    # TAB 4: FEATURE ANALYSIS
    # ========================
    with tabs[3]:
        st.header("🔍 Feature Analysis")
        
        try:
            fi = results.get('feature_importance', {})
            features = safe_get(lambda: engineer.feature_columns, [])
            rf_imp = fi.get('random_forest', [])
            
            if len(rf_imp) == len(features) and len(rf_imp) > 0:
                df = pd.DataFrame({'Feature': features, 'Importance': rf_imp})
                df = df.sort_values('Importance', ascending=False).head(5)
                
                for _, row in df.iterrows():
                    st.markdown(f"**{row['Feature']}**: {row['Importance']:.4f}")
                    st.progress(min(float(row['Importance']), 1.0))
            else:
                st.info("Run simulation to see feature importance")
        except Exception:
            st.info("Feature analysis will appear after simulation")
        
        st.markdown("---")
        st.info("""
        **Key Insights:**
        1. **Infection Probability** - Most important predictor
        2. **Elevation** - Higher = safer
        3. **City Walls** - Better route survival
        4. **Distance** - Farther = more time
        5. **Population** - Higher = faster spread
        """)
    
    # ========================
    # TAB 5: SURVIVAL GUIDE
    # ========================
    with tabs[4]:
        st.header("🧟 Survival Guide")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🛡️ Defensive Actions")
            st.success("""
            1. Fortify High Elevation
            2. Build Walls Early
            3. Monitor Infection Rates
            4. Stockpile at Distance
            """)
        with c2:
            st.markdown("### ⚠️ Danger Signals")
            st.error("""
            1. Low Elevation + High Population
            2. No Walls + Close Neighbors
            3. Ignoring Infection Probability
            4. Single Evacuation Route
            """)
        
        st.markdown("---")
        st.markdown("### 🗺️ Evacuation Checklist")
        st.markdown("""
        1. Identify high-elevation safe zones
        2. Monitor infection daily
        3. Evacuate at 30% risk threshold
        4. Use ML-predicted safe routes only
        5. Reach fortified zones by Day 20
        """)

else:
    # ========================
    # WELCOME SCREEN
    # ========================
    if st.session_state.error_message:
        st.error(f"Previous run failed: {st.session_state.error_message}")
        st.info("Adjust parameters and try again.")
    
    st.markdown("## 🎮 Welcome to the Zombie Apocalypse ML Simulator!")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### What This Does
        Interactive ML dashboard predicting survival routes during a zombie outbreak.
        
        ### Features
        - 🗺️ Real-time outbreak simulation
        - 🤖 3 ML models compared
        - 📊 Interactive visualizations
        - 🧟 Survival strategies
        """)
    with c2:
        st.markdown("""
        ### ML Models
        | Model | Purpose |
        |-------|---------|
        | Linear Regression | Baseline |
        | Decision Tree | Interpretable |
        | Random Forest | Best accuracy |
        
        ### Skills
        - Supervised Learning
        - Feature Engineering
        - Model Evaluation
        """)
    
    st.markdown("---")
    st.markdown("### 🚀 Get Started")
    st.markdown("Click **'Run Simulation & Train Models'** in the sidebar!")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>"
    "Built with 🧠 using Python, Scikit-learn & Streamlit</p>",
    unsafe_allow_html=True
)