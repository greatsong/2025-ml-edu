import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Advanced Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Step 4: Advanced Analysis and Insights")
st.markdown("### Deep analysis of model performance and finding improvement directions")

# Check data and models
if 'data' not in st.session_state or st.session_state.data is None:
    st.error("⚠️ No data found. Please start with data preparation.")
    if st.button("Go to Data Preparation"):
        st.switch_page("pages/1_data_preparation.py")
    st.stop()

if 'models' not in st.session_state or not st.session_state.models:
    st.warning("⚠️ No trained models found. Please train a model first.")
    if st.button("Go to Model Training"):
        st.switch_page("pages/3_model_training.py")
    st.stop()

# Educational content
with st.expander("📚 Importance of Advanced Analysis", expanded=True):
    st.markdown("""
    ### 🎯 Why do we need advanced analysis?
    
    Simple accuracy metrics alone cannot tell us the real performance of a model.
    
    **What we can learn through advanced analysis:**
    
    1. **Model weaknesses**
       - When does it make mistakes?
       - Does performance drop for specific classes or ranges?
    
    2. **Improvement potential**
       - Do we need more data?
       - Do we need different features?
       - Should we adjust model complexity?
    
    3. **Deployment readiness**
       - How reliable are the predictions?
       - What is the cost of errors?
       - Are the predictions explainable?
    
    4. **Business insights**
       - Which factors are most important?
       - Are there unexpected patterns?
       - What actionable insights exist?
    """)

st.divider()

# Analysis tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Error Analysis",
    "📊 Learning Curves",
    "🎲 Prediction Uncertainty",
    "🔄 Cross Validation",
    "💡 Improvement Suggestions"
])

# Data preparation
df = st.session_state.data.copy()
target_col = st.session_state.target_column
problem_type = st.session_state.data_type

# Prepare data for analysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

X = df.drop(columns=[target_col])
y = df[target_col]

# Encode target if needed
if problem_type == 'classification' and y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Select numeric columns only
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X = X[numeric_cols]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if problem_type == 'classification' else None
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with tab1:
    st.markdown("## 🔬 Error Analysis")
    st.markdown("Find patterns in model mistakes")
    
    # Model selection
    model_name = st.selectbox(
        "Select model to analyze",
        list(st.session_state.models.keys()),
        key="error_analysis_model"
    )
    
    if model_name and 'model' in st.session_state.models[model_name]:
        model = st.session_state.models[model_name]['model']
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        if problem_type == 'classification':
            st.markdown("### 🎯 Classification Error Analysis")
            
            # Find misclassified samples
            misclassified_mask = y_test != y_pred
            misclassified_indices = np.where(misclassified_mask)[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Test Samples", len(y_test))
            with col2:
                st.metric("Misclassified Samples", len(misclassified_indices))
            with col3:
                error_rate = len(misclassified_indices) / len(y_test) * 100
                st.metric("Error Rate", f"{error_rate:.1f}%")
            
            if len(misclassified_indices) > 0:
                # Misclassification patterns
                st.markdown("#### Misclassification Patterns")
                
                misclass_df = pd.DataFrame({
                    'Actual': y_test[misclassified_mask],
                    'Predicted': y_pred[misclassified_mask]
                })
                
                # Count misclassification pairs
                confusion_pairs = misclass_df.groupby(['Actual', 'Predicted']).size().reset_index(name='Count')
                confusion_pairs = confusion_pairs.sort_values('Count', ascending=False)
                
                # Create bar chart
                if len(confusion_pairs) > 0:
                    pairs_labels = []
                    for _, row in confusion_pairs.head(10).iterrows():
                        pairs_labels.append(f"{row['Actual']} → {row['Predicted']}")
                    
                    fig = px.bar(
                        x=confusion_pairs.head(10)['Count'].values,
                        y=pairs_labels,
                        orientation='h',
                        title="Most Frequent Misclassification Patterns (Actual → Predicted)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:  # Regression
            st.markdown("### 📉 Regression Error Analysis")
            
            errors = np.abs(y_test - y_pred)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Absolute Error", f"{errors.mean():.3f}")
            with col2:
                st.metric("Max Error", f"{errors.max():.3f}")
            with col3:
                st.metric("Error Std Dev", f"{errors.std():.3f}")
            
            # Error distribution
            fig = px.histogram(
                errors,
                nbins=30,
                title="Error Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("## 📊 Learning Curve Analysis")
    st.markdown("Analyze the impact of data size and model complexity on performance")
    
    analysis_type = st.radio(
        "Analysis Type",
        ["Training Data Size", "Model Complexity"]
    )
    
    if analysis_type == "Training Data Size":
        st.markdown("### 📈 Learning Curve")
        st.info("Will more data improve performance?")
        
        selected_model = st.selectbox(
            "Select model to analyze",
            list(st.session_state.models.keys()),
            key="learning_curve_model"
        )
        
        if st.button("Generate Learning Curve"):
            with st.spinner("Generating learning curve..."):
                from sklearn.model_selection import learning_curve
                
                model = st.session_state.models[selected_model]['model']
                
                # Calculate learning curve
                train_sizes = np.linspace(0.1, 1.0, 10)
                
                # Create new instance of the model
                model_class = type(model)
                new_model = model_class()
                
                train_sizes_abs, train_scores, val_scores = learning_curve(
                    new_model,
                    X_train_scaled, y_train,
                    train_sizes=train_sizes,
                    cv=5,
                    scoring='accuracy' if problem_type == 'classification' else 'r2',
                    n_jobs=-1
                )
                
                # Calculate mean and std
                train_mean = train_scores.mean(axis=1)
                train_std = train_scores.std(axis=1)
                val_mean = val_scores.mean(axis=1)
                val_std = val_scores.std(axis=1)
                
                # Visualization
                fig = go.Figure()
                
                # Training scores
                fig.add_trace(go.Scatter(
                    x=train_sizes_abs,
                    y=train_mean,
                    mode='lines+markers',
                    name='Training Score',
                    line=dict(color='blue'),
                    error_y=dict(
                        type='data',
                        array=train_std,
                        visible=True
                    )
                ))
                
                # Validation scores
                fig.add_trace(go.Scatter(
                    x=train_sizes_abs,
                    y=val_mean,
                    mode='lines+markers',
                    name='Validation Score',
                    line=dict(color='red'),
                    error_y=dict(
                        type='data',
                        array=val_std,
                        visible=True
                    )
                ))
                
                fig.update_layout(
                    title="Learning Curve: Performance vs Training Data Size",
                    xaxis_title="Training Data Size",
                    yaxis_title="Score",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                gap = train_mean[-1] - val_mean[-1]
                
                if gap > 0.1:
                    st.warning("""
                    ⚠️ **Signs of Overfitting**
                    - Large gap between training and validation scores
                    - Solutions: Increase regularization, get more data, simplify model
                    """)
                elif val_mean[-1] < 0.7:
                    st.warning("""
                    ⚠️ **Signs of Underfitting**
                    - Overall performance is low
                    - Solutions: Increase model complexity, add features, try different algorithms
                    """)
                else:
                    st.success("""
                    ✅ **Good Learning**
                    - Training and validation scores are converging
                    - Slight improvement possible with more data
                    """)

with tab3:
    st.markdown("## 🎲 Prediction Uncertainty Analysis")
    st.markdown("Analyze how confident the model is in its predictions")
    
    # Models with probability predictions
    prob_models = []
    for m_name, m_info in st.session_state.models.items():
        if 'model' in m_info and hasattr(m_info['model'], 'predict_proba'):
            prob_models.append(m_name)
    
    if prob_models and problem_type == 'classification':
        selected_model = st.selectbox(
            "Select model to analyze",
            prob_models,
            key="uncertainty_model"
        )
        
        if selected_model:
            model = st.session_state.models[selected_model]['model']
            
            # Probability predictions
            y_proba = model.predict_proba(X_test_scaled)
            y_pred = model.predict(X_test_scaled)
            
            # Maximum probability (confidence)
            max_proba = y_proba.max(axis=1)
            
            st.markdown("### 📊 Prediction Confidence Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence histogram
                fig = px.histogram(
                    max_proba,
                    nbins=30,
                    title="Prediction Confidence Distribution",
                    labels={'value': 'Confidence', 'count': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Accuracy by confidence
                st.markdown("#### Actual Accuracy by Confidence")
                st.info("Higher confidence should mean higher accuracy")
    
    else:
        st.info("Train a classification model with probability predictions to analyze uncertainty")

with tab4:
    st.markdown("## 🔄 Cross Validation Analysis")
    st.markdown("Evaluate model stability with different validation strategies")
    
    cv_model = st.selectbox(
        "Select model to analyze",
        list(st.session_state.models.keys()),
        key="cv_model"
    )
    
    cv_strategy = st.selectbox(
        "Cross Validation Strategy",
        ["K-Fold", "Stratified K-Fold"]
    )
    
    n_splits = st.slider("Number of Folds", 3, 10, 5)
    
    if st.button("Run Cross Validation"):
        with st.spinner("Running cross validation..."):
            from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
            
            # Select CV strategy
            if cv_strategy == "K-Fold":
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            else:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            model = st.session_state.models[cv_model]['model']
            
            # Define scoring metrics
            if problem_type == 'classification':
                scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            else:
                scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
            
            # Create new model instance
            model_class = type(model)
            new_model = model_class()
            
            # Run cross validation
            cv_results = cross_validate(
                new_model,
                X_train_scaled, y_train,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            # Display results
            st.markdown("### 📊 Cross Validation Results")
            
            # Summary statistics
            summary_data = []
            for scorer in scoring:
                train_scores = cv_results[f'train_{scorer}']
                test_scores = cv_results[f'test_{scorer}']
                
                summary_data.append({
                    'Metric': scorer.replace('_', ' ').title(),
                    'Train Mean': f"{train_scores.mean():.3f}",
                    'Train Std': f"{train_scores.std():.3f}",
                    'Test Mean': f"{test_scores.mean():.3f}",
                    'Test Std': f"{test_scores.std():.3f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

with tab5:
    st.markdown("## 💡 Improvement Suggestions")
    st.markdown("Model improvement recommendations based on analysis results")
    
    if st.session_state.models:
        st.markdown("### 🎯 Comprehensive Analysis and Suggestions")
        
        # Collect performance of all models
        model_performances = []
        
        for name, info in st.session_state.models.items():
            if 'model' in info:
                model = info['model']
                y_pred = model.predict(X_test_scaled)
                
                if problem_type == 'classification':
                    from sklearn.metrics import accuracy_score
                    score = accuracy_score(y_test, y_pred)
                else:
                    from sklearn.metrics import r2_score
                    score = r2_score(y_test, y_pred)
                
                model_performances.append((name, score))
        
        if model_performances:
            # Best and worst performing models
            model_performances.sort(key=lambda x: x[1], reverse=True)
            best_model = model_performances[0]
            worst_model = model_performances[-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"**Best Model**: {best_model[0]} ({best_model[1]:.3f})")
            with col2:
                st.warning(f"**Worst Model**: {worst_model[0]} ({worst_model[1]:.3f})")
            
            # Generate improvement suggestions
            st.markdown("### 📝 Improvement Recommendations")
            
            suggestions = []
            
            # Performance-based suggestions
            if best_model[1] < 0.7:
                suggestions.append("• Try more complex models (ensemble, neural networks)")
                suggestions.append("• Perform feature engineering")
                suggestions.append("• Collect more training data")
                suggestions.append("• Optimize hyperparameters more thoroughly")
            elif best_model[1] < 0.85:
                suggestions.append("• Fine-tune hyperparameters")
                suggestions.append("• Try ensemble methods")
                suggestions.append("• Add more relevant features")
            else:
                suggestions.append("• Model performance is good!")
                suggestions.append("• Consider model interpretability")
                suggestions.append("• Test on more diverse data")
            
            # Data-based suggestions
            if len(df) < 1000:
                suggestions.append("• Dataset is small - consider data augmentation")
                suggestions.append("• Use cross-validation for reliable estimates")
            
            if len(numeric_cols) < 5:
                suggestions.append("• Limited features - consider feature engineering")
                suggestions.append("• Look for additional data sources")
            
            for suggestion in suggestions:
                st.write(suggestion)
            
            # Next steps
            st.markdown("### 🗺️ Next Steps")
            
            st.info("""
            **Short term (immediate)**:
            - Fine-tune hyperparameters
            - Try different scaling methods
            - Experiment with feature selection
            
            **Medium term (1-2 weeks)**:
            - Deep feature engineering
            - Build ensemble models
            - Collect more training data
            
            **Long term (1+ month)**:
            - Collaborate with domain experts
            - Build automated ML pipeline
            - Deploy and monitor in production
            """)

# Sidebar
with st.sidebar:
    st.markdown("### 📈 Analysis Progress")
    
    if 'models' in st.session_state and st.session_state.models:
        st.success("✅ Models trained")
        st.info(f"Total models: {len(st.session_state.models)}")
    
    st.markdown("### 💡 Analysis Tips")
    with st.expander("Effective Analysis"):
        st.markdown("""
        1. **Find error patterns**
           - Look for repeated mistakes
           - Check data quality issues
        
        2. **Use learning curves**
           - Diagnose overfitting/underfitting
           - Estimate data requirements
        
        3. **Manage uncertainty**
           - Be careful with low confidence predictions
           - Adjust thresholds for better precision
        
        4. **Continuous improvement**
           - Accumulate small improvements
           - Keep systematic experiment records
        """)
