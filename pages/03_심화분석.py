import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="심화 분석",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Step 4: 심화 분석 및 인사이트")
st.markdown("### 모델 성능을 깊이 있게 분석하고 개선 방향을 찾아봅니다")

# 데이터 및 모델 체크
if 'data' not in st.session_state or st.session_state.data is None:
    st.error("⚠️ 데이터가 없습니다. 데이터 준비부터 시작해주세요.")
    if st.button("데이터 준비 페이지로 이동"):
        st.switch_page("pages/1_📊_데이터_준비.py")
    st.stop()

if 'models' not in st.session_state or not st.session_state.models:
    st.warning("⚠️ 학습된 모델이 없습니다. 먼저 모델을 학습시켜주세요.")
    if st.button("모델 학습 페이지로 이동"):
        st.switch_page("pages/3_🤖_모델_학습.py")
    st.stop()

# 교육 콘텐츠
with st.expander("📚 심화 분석의 중요성", expanded=True):
    st.markdown("""
    ### 🎯 왜 심화 분석이 필요한가?
    
    단순한 정확도 수치만으로는 모델의 진짜 성능을 알 수 없습니다.
    
    **심화 분석을 통해 알 수 있는 것들:**
    
    1. **모델의 약점**
       - 어떤 상황에서 틀리는가?
       - 특정 클래스나 범위에서 성능이 떨어지는가?
    
    2. **개선 가능성**
       - 더 많은 데이터가 필요한가?
       - 다른 특성이 필요한가?
       - 모델 복잡도 조정이 필요한가?
    
    3. **실제 배포 가능성**
       - 예측의 신뢰도는 어느 정도인가?
       - 오류의 비용은 얼마나 되는가?
       - 설명 가능한 예측인가?
    
    4. **비즈니스 인사이트**
       - 어떤 요인이 가장 중요한가?
       - 예상과 다른 패턴이 있는가?
       - 실행 가능한 통찰이 있는가?
    """)

st.divider()

# 분석 탭
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 오류 분석",
    "📊 학습 곡선",
    "🎲 예측 불확실성",
    "🔄 교차 검증 심화",
    "💡 개선 제안"
])

# 데이터 준비
df = st.session_state.data.copy()
target_col = st.session_state.target_column
problem_type = st.session_state.data_type

with tab1:
    st.markdown("## 🔬 오류 분석 (Error Analysis)")
    st.markdown("모델이 어떤 경우에 틀리는지 패턴을 찾아봅니다.")
    
    # 모델 선택
    model_name = st.selectbox(
        "분석할 모델",
        list(st.session_state.models.keys()),
        key="error_analysis_model"
    )
    
    if model_name:
        model_info = st.session_state.models[model_name]
        
        # 예측값이 있는지 확인
        if 'y_pred' in model_info:
            y_pred = model_info['y_pred']
            
            # 실제 테스트 데이터 필요 (재생성)
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            if problem_type == 'classification' and y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            X = X[numeric_cols]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if problem_type == 'classification' else None
            )
            
            if problem_type == 'classification':
                st.markdown("### 🎯 분류 오류 분석")
                
                # 오분류 샘플 찾기
                misclassified_mask = y_test != y_pred
                misclassified_indices = np.where(misclassified_mask)[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("전체 테스트 샘플", len(y_test))
                with col2:
                    st.metric("오분류 샘플", len(misclassified_indices))
                with col3:
                    st.metric("오류율", f"{len(misclassified_indices)/len(y_test)*100:.1f}%")
                
                if len(misclassified_indices) > 0:
                    # 오분류 패턴 분석
                    st.markdown("#### 오분류 패턴")
                    
                    misclass_df = pd.DataFrame({
                        '실제': y_test[misclassified_mask],
                        '예측': y_pred[misclassified_mask]
                    })
                    
                    # 오분류 매트릭스
                    confusion_pairs = misclass_df.groupby(['실제', '예측']).size().reset_index(name='횟수')
                    confusion_pairs = confusion_pairs.sort_values('횟수', ascending=False)
                    
                    fig = px.bar(
                        confusion_pairs.head(10),
                        x='횟수',
                        y=[f"{row['실제']} → {row['예측']}" for _, row in confusion_pairs.head(10).iterrows()],
                        orientation='h',
                        title="가장 빈번한 오분류 패턴 (실제 → 예측)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 오분류 샘플의 특성 분석
                    st.markdown("#### 오분류 샘플의 특성")
                    
                    feature_to_analyze = st.selectbox(
                        "분석할 특성",
                        numeric_cols
                    )
                    
                    if feature_to_analyze:
                        # 정확/오분류 샘플의 특성 분포 비교
                        fig = go.Figure()
                        
                        correct_values = X_test[~misclassified_mask][feature_to_analyze]
                        incorrect_values = X_test[misclassified_mask][feature_to_analyze]
                        
                        fig.add_trace(go.Box(
                            y=correct_values,
                            name="정확한 예측",
                            marker_color='green'
                        ))
                        
                        fig.add_trace(go.Box(
                            y=incorrect_values,
                            name="잘못된 예측",
                            marker_color='red'
                        ))
                        
                        fig.update_layout(
                            title=f"{feature_to_analyze}의 분포: 정확 vs 오분류",
                            yaxis_title=feature_to_analyze
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 통계적 차이
                        from scipy import stats
                        
                        if len(correct_values) > 0 and len(incorrect_values) > 0:
                            t_stat, p_value = stats.ttest_ind(correct_values, incorrect_values)
                            
                            if p_value < 0.05:
                                st.warning(f"⚠️ {feature_to_analyze}에서 정확/오분류 그룹 간 유의미한 차이가 있습니다 (p={p_value:.4f})")
                            else:
                                st.info(f"ℹ️ {feature_to_analyze}에서 정확/오분류 그룹 간 유의미한 차이가 없습니다 (p={p_value:.4f})")
            
            else:  # 회귀
                st.markdown("### 📉 회귀 오차 분석")
                
                errors = np.abs(y_test - y_pred)
                
                # 오차 분포
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("평균 절대 오차", f"{errors.mean():.3f}")
                with col2:
                    st.metric("최대 오차", f"{errors.max():.3f}")
                with col3:
                    st.metric("오차 표준편차", f"{errors.std():.3f}")
                
                # 큰 오차 샘플 분석
                threshold = st.slider(
                    "큰 오차 기준 (상위 %)",
                    5, 30, 10
                )
                
                error_threshold = np.percentile(errors, 100 - threshold)
                large_error_mask = errors > error_threshold
                
                st.markdown(f"#### 상위 {threshold}% 오차 샘플 분석")
                
                # 특성별 분석
                feature_to_analyze = st.selectbox(
                    "분석할 특성",
                    numeric_cols,
                    key="regression_error_feature"
                )
                
                if feature_to_analyze:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=X_test[feature_to_analyze],
                        y=errors,
                        mode='markers',
                        marker=dict(
                            color=large_error_mask,
                            colorscale=['blue', 'red'],
                            size=8,
                            opacity=0.6
                        ),
                        text=[f"오차: {e:.2f}" for e in errors],
                        name="오차"
                    ))
                    
                    fig.add_hline(
                        y=error_threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"상위 {threshold}% 기준선"
                    )
                    
                    fig.update_layout(
                        title=f"{feature_to_analyze} vs 예측 오차",
                        xaxis_title=feature_to_analyze,
                        yaxis_title="절대 오차"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("## 📊 학습 곡선 분석")
    st.markdown("데이터 크기와 모델 복잡도가 성능에 미치는 영향을 분석합니다.")
    
    analysis_type = st.radio(
        "분석 유형",
        ["학습 데이터 크기", "모델 복잡도"]
    )
    
    if analysis_type == "학습 데이터 크기":
        st.markdown("### 📈 학습 곡선 (Learning Curve)")
        st.info("데이터가 더 많으면 성능이 향상될까요?")
        
        selected_model = st.selectbox(
            "분석할 모델",
            list(st.session_state.models.keys()),
            key="learning_curve_model"
        )
        
        if st.button("학습 곡선 생성"):
            with st.spinner("학습 곡선 생성 중..."):
                from sklearn.model_selection import learning_curve
                
                model = st.session_state.models[selected_model]['model']
                
                # 학습 곡선 계산
                train_sizes = np.linspace(0.1, 1.0, 10)
                
                train_sizes_abs, train_scores, val_scores = learning_curve(
                    model.__class__(),  # 새 인스턴스 생성
                    X_train, y_train,
                    train_sizes=train_sizes,
                    cv=5,
                    scoring='accuracy' if problem_type == 'classification' else 'r2',
                    n_jobs=-1
                )
                
                # 평균과 표준편차
                train_mean = train_scores.mean(axis=1)
                train_std = train_scores.std(axis=1)
                val_mean = val_scores.mean(axis=1)
                val_std = val_scores.std(axis=1)
                
                # 시각화
                fig = go.Figure()
                
                # 훈련 점수
                fig.add_trace(go.Scatter(
                    x=train_sizes_abs,
                    y=train_mean,
                    mode='lines+markers',
                    name='훈련 점수',
                    line=dict(color='blue'),
                    error_y=dict(
                        type='data',
                        array=train_std,
                        visible=True
                    )
                ))
                
                # 검증 점수
                fig.add_trace(go.Scatter(
                    x=train_sizes_abs,
                    y=val_mean,
                    mode='lines+markers',
                    name='검증 점수',
                    line=dict(color='red'),
                    error_y=dict(
                        type='data',
                        array=val_std,
                        visible=True
                    )
                ))
                
                fig.update_layout(
                    title="학습 곡선: 데이터 크기에 따른 성능",
                    xaxis_title="훈련 데이터 크기",
                    yaxis_title="점수",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 해석
                gap = train_mean[-1] - val_mean[-1]
                
                if gap > 0.1:
                    st.warning("""
                    ⚠️ **과적합 징후**
                    - 훈련과 검증 점수 차이가 큽니다
                    - 해결책: 정규화 강화, 더 많은 데이터, 모델 단순화
                    """)
                elif val_mean[-1] < 0.7:
                    st.warning("""
                    ⚠️ **과소적합 징후**
                    - 전반적으로 성능이 낮습니다
                    - 해결책: 모델 복잡도 증
