import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_curve, auc, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# 분류 모델들
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# 회귀 모델들
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# 페이지 설정
st.set_page_config(
    page_title="모델 학습",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Step 3: 모델 학습 및 하이퍼파라미터 튜닝")
st.markdown("### 데이터로부터 패턴을 학습하는 AI 모델 만들기")

# 데이터 체크
if 'data' not in st.session_state or st.session_state.data is None:
    st.error("⚠️ 먼저 데이터를 준비해주세요!")
    if st.button("데이터 준비 페이지로 이동"):
        st.switch_page("pages/1_📊_데이터_준비.py")
    st.stop()

# 세션 상태 초기화
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []

df = st.session_state.data.copy()
target_col = st.session_state.target_column
problem_type = st.session_state.data_type

# 교육 콘텐츠
with st.expander("📚 머신러닝 모델 학습이란?", expanded=True):
    st.markdown("""
    ### 🎯 모델 학습 (Model Training)
    
    > **"모델은 데이터로부터 패턴을 찾아내는 과정을 거쳐 예측 능력을 갖추게 됩니다"**
    
    ### 📊 핵심 개념들:
    
    **1. 훈련/검증/테스트 분할**
    - **훈련 세트 (70%)**: 모델이 패턴을 학습
    - **검증 세트 (15%)**: 하이퍼파라미터 조정
    - **테스트 세트 (15%)**: 최종 성능 평가
    
    **2. 과적합 vs 과소적합**
    - **과적합**: 훈련 데이터에 너무 최적화 → 새 데이터에서 성능 저하
    - **과소적합**: 패턴을 제대로 학습 못함 → 전반적으로 낮은 성능
    
    **3. 하이퍼파라미터 튜닝**
    - 모델의 학습 방식을 제어하는 설정값
    - 최적값을 찾으면 성능이 크게 향상
    
    **4. 교차 검증 (Cross-Validation)**
    - 데이터를 여러 방식으로 나누어 평가
    - 더 안정적이고 신뢰할 수 있는 성능 측정
    """)

st.divider()

# 데이터 준비
X = df.drop(columns=[target_col])
y = df[target_col]

# 타겟 인코딩 (분류 문제에서 문자열인 경우)
if problem_type == 'classification' and y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)
    st.session_state.label_encoder = le

# 수치형 컬럼만 선택
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X = X[numeric_cols]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, 
    stratify=y if problem_type == 'classification' else None
)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 탭 구성
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 빠른 학습", 
    "🔧 하이퍼파라미터 튜닝", 
    "⚔️ 모델 비교", 
    "📊 상세 평가"
])

# 모델 정의
if problem_type == 'classification':
    MODELS = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'liblinear']
            },
            'description': '선형 결정 경계를 사용하는 확률 기반 분류기'
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'description': 'if-then 규칙 기반의 트리 구조 분류기'
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            },
            'description': '여러 결정 트리의 앙상블로 과적합을 방지'
        },
        'SVM': {
            'model': SVC(probability=True),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'description': '고차원 공간에서 최적의 결정 경계를 찾는 모델'
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            },
            'description': '가까운 이웃들의 투표로 분류하는 모델'
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {},
            'description': '베이즈 정리 기반의 확률적 분류기'
        },
        'Neural Network': {
            'model': MLPClassifier(max_iter=1000),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'learning_rate_init': [0.001, 0.01]
            },
            'description': '인공 신경망 기반의 딥러닝 모델'
        }
    }
else:  # 회귀
    MODELS = {
        'Linear Regression': {
            'model': LinearRegression(),
            'params': {},
            'description': '가장 기본적인 선형 회귀 모델'
        },
        'Ridge Regression': {
            'model': Ridge(),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
            },
            'description': 'L2 정규화를 적용한 선형 회귀'
        },
        'Lasso Regression': {
            'model': Lasso(),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1]
            },
            'description': 'L1 정규화를 적용한 선형 회귀 (특성 선택 효과)'
        },
        'Decision Tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            'description': '트리 구조 기반의 회귀 모델'
        },
        'Random Forest': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None]
            },
            'description': '여러 트리의 평균으로 예측하는 앙상블 모델'
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            },
            'description': 'Support Vector Machine의 회귀 버전'
        }
    }

with tab1:
    st.markdown("## 🎯 빠른 모델 학습")
    st.markdown("기본 설정으로 빠르게 모델을 학습시켜 봅니다.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "학습할 모델 선택",
            list(MODELS.keys()),
            help="각 모델의 특징을 확인하고 선택하세요"
        )
    
    with col2:
        cv_folds = st.number_input("교차 검증 폴드 수", 3, 10, 5)
    
    # 모델 설명
    st.info(f"**{selected_model}**: {MODELS[selected_model]['description']}")
    
    # 학습 버튼
    if st.button("🚀 모델 학습 시작", type="primary"):
        with st.spinner(f"{selected_model} 학습 중..."):
            # 프로그레스 바
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 모델 생성 및 학습
            model = MODELS[selected_model]['model']
            
            # 교차 검증
            status_text.text("교차 검증 수행 중...")
            progress_bar.progress(30)
            
            if problem_type == 'classification':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                           cv=cv_folds, scoring='accuracy')
            else:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                           cv=cv_folds, scoring='r2')
            
            # 모델 학습
            status_text.text("모델 학습 중...")
            progress_bar.progress(60)
            
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            # 예측
            status_text.text("예측 및 평가 중...")
            progress_bar.progress(90)
            
            y_pred = model.predict(X_test_scaled)
            
            # 결과 저장
            st.session_state.models[selected_model] = {
                'model': model,
                'cv_scores': cv_scores,
                'y_pred': y_pred,
                'training_time': training_time
            }
            
            progress_bar.progress(100)
            status_text.text("학습 완료!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
        
        # 결과 표시
        st.success(f"✅ {selected_model} 학습 완료!")
        
        col1, col2, col3, col4 = st.columns(4)
        
        if problem_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            col1.metric("정확도", f"{accuracy:.3f}")
            col2.metric("정밀도", f"{precision:.3f}")
            col3.metric("재현율", f"{recall:.3f}")
            col4.metric("F1 Score", f"{f1:.3f}")
        else:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            col1.metric("R² Score", f"{r2:.3f}")
            col2.metric("RMSE", f"{rmse:.3f}")
            col3.metric("MAE", f"{mae:.3f}")
            col4.metric("학습 시간", f"{training_time:.2f}초")
        
        # 교차 검증 결과
        st.markdown("### 📊 교차 검증 결과")
        
        cv_df = pd.DataFrame({
            'Fold': [f'Fold {i+1}' for i in range(len(cv_scores))],
            'Score': cv_scores
        })
        
        fig = px.bar(cv_df, x='Fold', y='Score', 
                     title=f"교차 검증 점수 (평균: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("## 🔧 하이퍼파라미터 튜닝")
    st.markdown("최적의 모델 성능을 위해 하이퍼파라미터를 조정합니다.")
    
    tuning_model = st.selectbox(
        "튜닝할 모델 선택",
        [m for m in MODELS.keys() if MODELS[m]['params']],
        help="하이퍼파라미터가 있는 모델만 표시됩니다"
    )
    
    tuning_method = st.radio(
        "튜닝 방법",
        ["Grid Search", "Random Search", "수동 조정"]
    )
    
    if tuning_method == "수동 조정":
        st.markdown("### 수동 하이퍼파라미터 조정")
        
        # 파라미터별 위젯 생성
        params = {}
        model_params = MODELS[tuning_model]['params']
        
        if model_params:
            for param_name, param_values in model_params.items():
                if isinstance(param_values[0], (int, float)):
                    # 수치형 파라미터
                    if len(param_values) > 1:
                        params[param_name] = st.select_slider(
                            f"{param_name}",
                            options=param_values,
                            value=param_values[len(param_values)//2]
                        )
                else:
                    # 범주형 파라미터
                    params[param_name] = st.selectbox(
                        f"{param_name}",
                        param_values
                    )
            
            if st.button("선택한 파라미터로 학습"):
                with st.spinner("모델 학습 중..."):
                    # 모델 생성
                    model_class = type(MODELS[tuning_model]['model'])
                    model = model_class(**params)
                    
                    # 학습
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    # 결과 표시
                    if problem_type == 'classification':
                        score = accuracy_score(y_test, y_pred)
                        st.success(f"✅ 정확도: {score:.3f}")
                    else:
                        score = r2_score(y_test, y_pred)
                        st.success(f"✅ R² Score: {score:.3f}")
                    
                    # 저장
                    st.session_state.models[f"{tuning_model}_manual"] = {
                        'model': model,
                        'params': params,
                        'score': score
                    }
    
    elif tuning_method == "Grid Search":
        st.markdown("### Grid Search 하이퍼파라미터 튜닝")
        st.info("모든 가능한 조합을 체계적으로 탐색합니다.")
        
        # 탐색할 파라미터 선택
        param_grid = {}
        model_params = MODELS[tuning_model]['params']
        
        if model_params:
            st.markdown("#### 탐색할 파라미터 범위 설정")
            
            for param_name, param_values in model_params.items():
                if st.checkbox(f"{param_name} 탐색", value=True):
                    param_grid[param_name] = param_values
            
            if param_grid and st.button("Grid Search 시작", type="primary"):
                with st.spinner(f"Grid Search 진행 중... (조합 수: {np.prod([len(v) for v in param_grid.values()])})"):
                    # Grid Search 수행
                    model = MODELS[tuning_model]['model']
                    
                    grid_search = GridSearchCV(
                        model, param_grid, 
                        cv=5,
                        scoring='accuracy' if problem_type == 'classification' else 'r2',
                        n_jobs=-1
                    )
                    
                    grid_search.fit(X_train_scaled, y_train)
                    
                    # 결과 저장
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    best_score = grid_search.best_score_
                    
                    st.session_state.best_model = best_model
                    st.session_state.models[f"{tuning_model}_grid"] = {
                        'model': best_model,
                        'params': best_params,
                        'score': best_score,
                        'cv_results': grid_search.cv_results_
                    }
                
                # 결과 표시
                st.success(f"✅ 최적 파라미터를 찾았습니다!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 최적 파라미터")
                    for param, value in best_params.items():
                        st.write(f"**{param}**: {value}")
                
                with col2:
                    st.markdown("#### 성능")
                    st.metric("최고 CV 점수", f"{best_score:.3f}")
                    
                    # 테스트 세트 평가
                    y_pred = best_model.predict(X_test_scaled)
                    if problem_type == 'classification':
                        test_score = accuracy_score(y_test, y_pred)
                        st.metric("테스트 정확도", f"{test_score:.3f}")
                    else:
                        test_score = r2_score(y_test, y_pred)
                        st.metric("테스트 R²", f"{test_score:.3f}")
                
                # Grid Search 결과 시각화
                st.markdown("### 📊 Grid Search 결과 분석")
                
                results_df = pd.DataFrame(grid_search.cv_results_)
                
                # 파라미터별 성능 히트맵 (2개 파라미터인 경우)
                if len(param_grid) == 2:
                    param_names = list(param_grid.keys())
                    pivot_table = results_df.pivot_table(
                        values='mean_test_score',
                        index=f'param_{param_names[0]}',
                        columns=f'param_{param_names[1]}'
                    )
                    
                    fig = px.imshow(
                        pivot_table,
                        labels=dict(x=param_names[1], y=param_names[0], color="Score"),
                        title="파라미터 조합별 성능 히트맵"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    else:  # Random Search
        st.markdown("### Random Search 하이퍼파라미터 튜닝")
        st.info("무작위로 선택한 조합을 탐색합니다. Grid Search보다 빠릅니다.")
        
        n_iter = st.slider("탐색할 조합 수", 10, 100, 20)
        
        if MODELS[tuning_model]['params'] and st.button("Random Search 시작", type="primary"):
            with st.spinner(f"Random Search 진행 중... ({n_iter}개 조합)"):
                model = MODELS[tuning_model]['model']
                param_distributions = MODELS[tuning_model]['params']
                
                random_search = RandomizedSearchCV(
                    model, param_distributions,
                    n_iter=n_iter,
                    cv=5,
                    scoring='accuracy' if problem_type == 'classification' else 'r2',
                    n_jobs=-1,
                    random_state=42
                )
                
                random_search.fit(X_train_scaled, y_train)
                
                # 결과 저장
                best_model = random_search.best_estimator_
                best_params = random_search.best_params_
                best_score = random_search.best_score_
                
                st.session_state.models[f"{tuning_model}_random"] = {
                    'model': best_model,
                    'params': best_params,
                    'score': best_score
                }
            
            st.success(f"✅ Random Search 완료!")
            st.write("**최적 파라미터:**", best_params)
            st.metric("최고 CV 점수", f"{best_score:.3f}")

with tab3:
    st.markdown("## ⚔️ 모델 비교")
    st.markdown("여러 모델을 동시에 학습시켜 성능을 비교합니다.")
    
    # 비교할 모델 선택
    models_to_compare = st.multiselect(
        "비교할 모델들을 선택하세요",
        list(MODELS.keys()),
        default=list(MODELS.keys())[:3]
    )
    
    if st.button("🏁 모델 비교 시작", type="primary") and models_to_compare:
        comparison_results = []
        
        # 프로그레스 바
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_name in enumerate(models_to_compare):
            status_text.text(f"{model_name} 학습 중...")
            progress_bar.progress((i + 1) / len(models_to_compare))
            
            # 모델 학습
            model = MODELS[model_name]['model']
            
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            # 예측
            y_pred = model.predict(X_test_scaled)
            
            # 평가
            if problem_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                comparison_results.append({
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'Training Time (s)': training_time
                })
            else:
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                comparison_results.append({
                    'Model': model_name,
                    'R² Score': r2,
                    'RMSE': np.sqrt(mse),
                    'MAE': mae,
                    'Training Time (s)': training_time
                })
        
        progress_bar.empty()
        status_text.empty()
        
        # 결과 데이터프레임
        comparison_df = pd.DataFrame(comparison_results)
        
        # 최고 성능 모델 찾기
        if problem_type == 'classification':
            best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
            st.success(f"🏆 최고 성능 모델: **{best_model_name}**")
        else:
            best_model_name = comparison_df.loc[comparison_df['R² Score'].idxmax(), 'Model']
            st.success(f"🏆 최고 성능 모델: **{best_model_name}**")
        
        # 결과 테이블
        st.markdown("### 📊 성능 비교 테이블")
        
        # 스타일링된 데이터프레임
        styled_df = comparison_df.style.highlight_max(
            axis=0, 
            subset=[col for col in comparison_df.columns if col not in ['Model', 'Training Time (s)']],
            color='lightgreen'
        ).highlight_min(
            axis=0,
            subset=['Training Time (s)'],
            color='lightblue'
        ).format({col: '{:.3f}' for col in comparison_df.columns if col != 'Model'})
        
        st.dataframe(styled_df, use_container_width=True)
        
        # 레이더 차트 (분류 문제)
        if problem_type == 'classification':
            st.markdown("### 📈 성능 레이더 차트")
            
            fig = go.Figure()
            
            categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            
            for _, row in comparison_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row[cat] for cat in categories],
                    theta=categories,
                    fill='toself',
                    name=row['Model']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="모델별 성능 지표 비교"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 막대 그래프
        st.markdown("### 📊 성능 막대 그래프")
        
        if problem_type == 'classification':
            metric_to_plot = st.selectbox("표시할 지표", categories)
        else:
            metric_to_plot = st.selectbox("표시할 지표", ['R² Score', 'RMSE', 'MAE'])
        
        fig = px.bar(
            comparison_df,
            x='Model',
            y=metric_to_plot,
            title=f"{metric_to_plot} 비교",
            color=metric_to_plot,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("## 📊 상세 모델 평가")
    
    # 평가할 모델 선택
    trained_models = list(st.session_state.models.keys())
    
    if trained_models:
        eval_model_name = st.selectbox(
            "평가할 모델 선택",
            trained_models
        )
        
        if eval_model_name:
            model_info = st.session_state.models[eval_model_name]
            model = model_info['model']
            
            # 예측
            y_pred = model.predict(X_test_scaled)
            
            if problem_type == 'classification':
                st.markdown("### 분류 모델 상세 평가")
                
                # Confusion Matrix
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig = px.imshow(
                        cm,
                        labels=dict(x="예측", y="실제", color="개수"),
                        text_auto=True,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(title="혼동 행렬")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 정규화된 Confusion Matrix")
                    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    
                    fig = px.imshow(
                        cm_normalized,
                        labels=dict(x="예측", y="실제", color="비율"),
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r'
                    )
                    fig.update_layout(title="정규화된 혼동 행렬")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Classification Report
                st.markdown("#### 📋 분류 리포트")
                
                if hasattr(st.session_state, 'label_encoder'):
                    target_names = st.session_state.label_encoder.classes_
                else:
                    target_names = [str(i) for i in np.unique(y_test)]
                
                report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                
                st.dataframe(
                    report_df.style.format({'precision': '{:.3f}', 'recall': '{:.3f}', 
                                           'f1-score': '{:.3f}', 'support': '{:.0f}'}),
                    use_container_width=True
                )
                
                # ROC Curve (이진 분류)
                if len(np.unique(y_test)) == 2:
                    st.markdown("#### ROC Curve")
                    
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test_scaled)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        roc_auc = auc(fpr, tpr)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines',
                            name=f'ROC Curve (AUC = {roc_auc:.3f})',
                            line=dict(color='darkblue', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Random Classifier',
                            line=dict(color='red', width=1, dash='dash')
                        ))
                        fig.update_layout(
                            title='ROC Curve',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info(f"""
                        💡 **ROC-AUC 해석:**
                        - AUC = {roc_auc:.3f}
                        - 1.0: 완벽한 분류기
                        - 0.5: 무작위 분류기
                        - 0.7-0.8: 적절한 성능
                        - 0.8-0.9: 좋은 성능
                        - 0.9+: 매우 좋은 성능
                        """)
            
            else:  # 회귀 모델 평가
                st.markdown("### 회귀 모델 상세 평가")
                
                # 예측 vs 실제 플롯
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 예측 vs 실제")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=y_test, y=y_pred,
                        mode='markers',
                        name='예측값',
                        marker=dict(color='blue', opacity=0.5)
                    ))
                    
                    # 완벽한 예측 선
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='완벽한 예측',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='예측값 vs 실제값',
                        xaxis_title='실제값',
                        yaxis_title='예측값'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 잔차 플롯")
                    
                    residuals = y_test - y_pred
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=y_pred, y=residuals,
                        mode='markers',
                        name='잔차',
                        marker=dict(color='green', opacity=0.5)
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    
                    fig.update_layout(
                        title='잔차 플롯',
                        xaxis_title='예측값',
                        yaxis_title='잔차 (실제 - 예측)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 잔차 분포
                st.markdown("#### 잔차 분포 분석")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 히스토그램
                    fig = px.histogram(
                        residuals,
                        nbins=30,
                        title="잔차 히스토그램"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Q-Q plot
                    from scipy import stats
                    
                    fig = go.Figure()
                    
                    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
                    sample_quantiles = np.sort(residuals)
                    
                    fig.add_trace(go.Scatter(
                        x=theoretical_quantiles,
                        y=sample_quantiles,
                        mode='markers',
                        name='잔차 Q-Q'
                    ))
                    
                    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
                    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='정규분포',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="잔차 Q-Q Plot",
                        xaxis_title="이론적 분위수",
                        yaxis_title="샘플 분위수"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 평가 메트릭
                st.markdown("#### 📊 회귀 평가 지표")
                
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                metrics_col1.metric("R² Score", f"{r2:.4f}")
                metrics_col2.metric("RMSE", f"{np.sqrt(mse):.4f}")
                metrics_col3.metric("MAE", f"{mae:.4f}")
                metrics_col4.metric("MAPE", f"{mape:.2f}%")
                
                with st.expander("📚 평가 지표 설명"):
                    st.markdown("""
                    - **R² Score**: 모델이 설명하는 분산의 비율 (1에 가까울수록 좋음)
                    - **RMSE**: 평균 제곱근 오차 (낮을수록 좋음, 큰 오차에 민감)
                    - **MAE**: 평균 절대 오차 (낮을수록 좋음, 해석이 쉬움)
                    - **MAPE**: 평균 절대 백분율 오차 (스케일에 무관한 비교)
                    """)
            
            # 특성 중요도 (가능한 경우)
            if hasattr(model, 'feature_importances_'):
                st.markdown("### 🎯 특성 중요도")
                
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig = px.bar(
                    feature_importance.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 10 중요 특성"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                💡 **특성 중요도 활용법:**
                - 가장 중요한 특성들에 집중
                - 불필요한 특성 제거 고려
                - 도메인 지식과 비교하여 검증
                """)
    else:
        st.info("먼저 모델을 학습시켜주세요.")

# 모델 저장 섹션
st.divider()
st.markdown("## 💾 모델 저장 및 배포")

if st.session_state.models:
    col1, col2 = st.columns(2)
    
    with col1:
        save_model = st.selectbox(
            "저장할 모델 선택",
            list(st.session_state.models.keys())
        )
        
        if st.button("모델 저장", type="primary"):
            import pickle
            
            model_to_save = st.session_state.models[save_model]['model']
            
            # 모델 정보 딕셔너리
            model_package = {
                'model': model_to_save,
                'scaler': scaler,
                'features': list(X.columns),
                'problem_type': problem_type,
                'model_name': save_model,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            # pickle로 저장
            model_bytes = pickle.dumps(model_package)
            
            st.download_button(
                label="📥 모델 다운로드 (.pkl)",
                data=model_bytes,
                file_name=f"model_{save_model}_{model_package['timestamp']}.pkl",
                mime="application/octet-stream"
            )
            
            st.success("✅ 모델이 준비되었습니다!")
    
    with col2:
        st.markdown("### 📝 모델 사용 코드")
        
        code = f"""
# 모델 불러오기
import pickle
import pandas as pd

with open('model_{save_model}.pkl', 'rb') as f:
    model_package = pickle.load(f)

model = model_package['model']
scaler = model_package['scaler']
features = model_package['features']

# 새로운 데이터로 예측
new_data = pd.DataFrame({{
    # 특성 값 입력
    {', '.join([f"'{col}': [값]" for col in X.columns[:3]])}
}})

# 스케일링
new_data_scaled = scaler.transform(new_data[features])

# 예측
prediction = model.predict(new_data_scaled)
print(f"예측 결과: {{prediction[0]}}")
"""
        st.code(code, language='python')

# 다음 단계 안내
st.divider()
st.markdown("## 🎯 다음 단계")

col1, col2, col3 = st.columns(3)

with col2:
    if st.session_state.models:
        st.success("""
        ### ✅ 모델 학습 완료!
        
        이제 다음을 할 수 있습니다:
        - 모델 성능 심화 분석
        - 하이퍼파라미터 추가 튜닝
        - 새로운 데이터로 예측
        - 모델 배포 준비
        """)
        
        if st.button("📈 심화 분석 페이지로", type="primary", use_container_width=True):
            st.switch_page("pages/4_📈_심화_분석.py")
    else:
        st.info("모델을 학습시켜 성능을 확인해보세요!")

# 사이드바
with st.sidebar:
    st.markdown("### 🤖 학습된 모델들")
    
    if st.session_state.models:
        for model_name, model_info in st.session_state.models.items():
            with st.expander(model_name):
                if 'score' in model_info:
                    st.write(f"점수: {model_info['score']:.3f}")
                if 'params' in model_info:
                    st.write("파라미터:", model_info['params'])
                if 'training_time' in model_info:
                    st.write(f"학습 시간: {model_info['training_time']:.2f}초")
    else:
        st.info("아직 학습된 모델이 없습니다.")
    
    st.markdown("### 💡 모델 선택 가이드")
    
    with st.expander("어떤 모델을 선택할까?"):
        st.markdown("""
        **선형 모델 (Logistic/Linear)**
        - ✅ 해석이 쉬움
        - ✅ 빠른 학습
        - ❌ 비선형 패턴 못잡음
        
        **트리 기반 (RF, GB)**
        - ✅ 비선형 패턴
        - ✅ 특성 중요도
        - ❌ 과적합 위험
        
        **SVM**
        - ✅ 고차원 데이터
        - ✅ 강건한 성능
        - ❌ 느린 학습
        
        **신경망**
        - ✅ 복잡한 패턴
        - ✅ 높은 성능
        - ❌ 많은 데이터 필요
        """)
