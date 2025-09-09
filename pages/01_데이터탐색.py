import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# 페이지 설정
st.set_page_config(
    page_title="데이터 탐색 (EDA)",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Step 2: 탐색적 데이터 분석 (EDA)")
st.markdown("### 데이터 속 숨겨진 패턴을 발견하는 여정")

# 데이터 체크
if 'data' not in st.session_state or st.session_state.data is None:
    st.error("⚠️ 먼저 데이터를 준비해주세요!")
    if st.button("데이터 준비 페이지로 이동"):
        st.switch_page("pages/1_📊_데이터_준비.py")
    st.stop()

df = st.session_state.data.copy()
target_col = st.session_state.target_column
problem_type = st.session_state.data_type

# 교육 콘텐츠
with st.expander("📚 EDA란 무엇이고 왜 중요한가요?", expanded=True):
    st.markdown("""
    ### 🎯 탐색적 데이터 분석 (Exploratory Data Analysis, EDA)
    
    > **"데이터를 이해하지 못하면, 모델도 이해할 수 없습니다"**
    
    EDA는 데이터를 **시각화**하고 **요약**하여 다음을 발견하는 과정입니다:
    
    1. **데이터의 분포와 패턴**
       - 데이터가 어떻게 퍼져있는가?
       - 정규분포를 따르는가?
       - 치우침(skewness)이 있는가?
    
    2. **변수 간의 관계**
       - 어떤 특성들이 서로 관련이 있는가?
       - 타겟과 가장 관련 깊은 특성은?
       - 다중공선성 문제는 없는가?
    
    3. **데이터 품질 문제**
       - 이상치(outlier)가 있는가?
       - 결측값은 어떻게 분포하는가?
       - 데이터 입력 오류는 없는가?
    
    4. **특성 엔지니어링 아이디어**
       - 새로운 특성을 만들 수 있는가?
       - 특성 변환이 필요한가?
       - 불필요한 특성은 무엇인가?
    """)

st.divider()

# 메인 분석 섹션
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 단변량 분석", 
    "🔗 다변량 분석", 
    "🎯 타겟 분석", 
    "⚠️ 이상치 탐지",
    "🔧 데이터 전처리"
])

with tab1:
    st.markdown("## 📊 단변량 분석 (Univariate Analysis)")
    st.markdown("각 변수를 개별적으로 분석하여 분포와 특성을 파악합니다.")
    
    # 수치형과 범주형 변수 구분
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    # 수치형 변수 분석
    if numeric_cols:
        st.markdown("### 수치형 변수 분석")
        
        selected_numeric = st.selectbox(
            "분석할 수치형 변수 선택",
            numeric_cols,
            help="각 변수의 분포를 상세히 분석합니다"
        )
        
        if selected_numeric:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("평균", f"{df[selected_numeric].mean():.2f}")
            with col2:
                st.metric("중앙값", f"{df[selected_numeric].median():.2f}")
            with col3:
                st.metric("표준편차", f"{df[selected_numeric].std():.2f}")
            with col4:
                skewness = df[selected_numeric].skew()
                st.metric("왜도", f"{skewness:.2f}")
                if abs(skewness) > 1:
                    st.caption("⚠️ 치우친 분포")
            
            # 분포 시각화
            col1, col2 = st.columns(2)
            
            with col1:
                # 히스토그램과 KDE
                fig = go.Figure()
                
                # 히스토그램
                fig.add_trace(go.Histogram(
                    x=df[selected_numeric],
                    name='Histogram',
                    nbinsx=30,
                    histnorm='probability density',
                    marker_color='lightblue'
                ))
                
                # KDE 곡선 추가
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(df[selected_numeric].dropna())
                x_range = np.linspace(df[selected_numeric].min(), df[selected_numeric].max(), 100)
                kde_values = kde(x_range)
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=kde_values,
                    mode='lines',
                    name='KDE',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title=f"{selected_numeric}의 분포",
                    xaxis_title=selected_numeric,
                    yaxis_title="밀도",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 박스플롯
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=df[selected_numeric],
                    name=selected_numeric,
                    boxpoints='outliers',
                    marker_color='darkblue'
                ))
                fig.update_layout(
                    title=f"{selected_numeric}의 박스플롯",
                    yaxis_title=selected_numeric
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 정규성 검정
            with st.expander("📊 정규성 검정"):
                # Shapiro-Wilk 검정
                if len(df[selected_numeric]) <= 5000:
                    stat, p_value = stats.shapiro(df[selected_numeric].dropna())
                    test_name = "Shapiro-Wilk"
                else:
                    stat, p_value = stats.normaltest(df[selected_numeric].dropna())
                    test_name = "D'Agostino-Pearson"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"{test_name} 통계량", f"{stat:.4f}")
                with col2:
                    st.metric("p-value", f"{p_value:.4f}")
                
                if p_value > 0.05:
                    st.success("✅ 정규분포를 따른다고 볼 수 있습니다 (p > 0.05)")
                else:
                    st.warning("⚠️ 정규분포를 따르지 않습니다 (p ≤ 0.05)")
                    st.info("""
                    💡 **정규분포가 아닐 때 할 수 있는 것:**
                    - 로그 변환 (양의 치우침)
                    - 제곱근 변환 (완만한 치우침)
                    - Box-Cox 변환 (자동 최적 변환)
                    - 트리 기반 모델 사용 (변환 불필요)
                    """)
    
    # 범주형 변수 분석
    if categorical_cols:
        st.markdown("### 범주형 변수 분석")
        
        selected_categorical = st.selectbox(
            "분석할 범주형 변수 선택",
            categorical_cols,
            help="각 범주의 분포를 확인합니다"
        )
        
        if selected_categorical:
            value_counts = df[selected_categorical].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 막대 그래프
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"{selected_categorical}의 분포",
                    labels={'x': selected_categorical, 'y': '빈도'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 파이 차트
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"{selected_categorical}의 비율"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 범주별 통계
            st.markdown("#### 범주별 통계")
            stats_df = pd.DataFrame({
                '범주': value_counts.index,
                '빈도': value_counts.values,
                '비율(%)': (value_counts.values / len(df) * 100).round(2)
            })
            st.dataframe(stats_df, use_container_width=True)

with tab2:
    st.markdown("## 🔗 다변량 분석 (Multivariate Analysis)")
    st.markdown("변수들 간의 관계를 분석하여 패턴을 발견합니다.")
    
    analysis_type = st.radio(
        "분석 유형 선택",
        ["상관관계 분석", "산점도 매트릭스", "특성 간 관계"]
    )
    
    if analysis_type == "상관관계 분석":
        if numeric_cols:
            st.markdown("### 상관관계 히트맵")
            
            # 상관계수 계산
            corr_matrix = df[numeric_cols + [target_col] if target_col in df.select_dtypes(include=[np.number]).columns else numeric_cols].corr()
            
            # 히트맵
            fig = px.imshow(
                corr_matrix,
                labels=dict(x="변수", y="변수", color="상관계수"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # 상관관계 해석
            with st.expander("📚 상관관계 해석 가이드"):
                st.markdown("""
                **상관계수(r)의 해석:**
                - **r = 1**: 완벽한 양의 상관관계
                - **0.7 ≤ r < 1**: 강한 양의 상관관계
                - **0.3 ≤ r < 0.7**: 중간 양의 상관관계
                - **-0.3 < r < 0.3**: 약한 또는 무상관
                - **-0.7 < r ≤ -0.3**: 중간 음의 상관관계
                - **-1 < r ≤ -0.7**: 강한 음의 상관관계
                - **r = -1**: 완벽한 음의 상관관계
                
                ⚠️ **주의사항:**
                - 상관관계 ≠ 인과관계
                - 비선형 관계는 포착하지 못함
                - 이상치에 민감함
                """)
            
            # 타겟과의 상관관계
            if target_col in corr_matrix.columns:
                st.markdown("### 타겟 변수와의 상관관계")
                target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
                
                # 막대 그래프
                fig = go.Figure(go.Bar(
                    x=target_corr.values,
                    y=target_corr.index,
                    orientation='h',
                    marker_color=['green' if x > 0 else 'red' for x in target_corr.values]
                ))
                fig.update_layout(
                    title=f"{target_col}과의 상관관계",
                    xaxis_title="상관계수",
                    yaxis_title="특성",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 높은 상관관계 경고
                high_corr = target_corr[abs(target_corr) > 0.8]
                if len(high_corr) > 0:
                    st.warning(f"⚠️ 다음 특성들이 타겟과 매우 높은 상관관계를 보입니다: {', '.join(high_corr.index)}")
                    st.info("💡 너무 높은 상관관계는 데이터 유출(data leakage)의 신호일 수 있습니다.")
    
    elif analysis_type == "산점도 매트릭스":
        if len(numeric_cols) >= 2:
            st.markdown("### 산점도 매트릭스 (Scatter Plot Matrix)")
            
            # 변수 선택 (최대 5개)
            selected_vars = st.multiselect(
                "분석할 변수 선택 (최대 5개)",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))]
            )
            
            if len(selected_vars) >= 2:
                # 타겟 포함 여부
                include_target = st.checkbox("타겟 변수로 색상 구분", value=True)
                
                if include_target and problem_type == 'classification':
                    fig = px.scatter_matrix(
                        df,
                        dimensions=selected_vars,
                        color=target_col,
                        title="변수 간 관계 시각화"
                    )
                else:
                    fig = px.scatter_matrix(
                        df,
                        dimensions=selected_vars,
                        title="변수 간 관계 시각화"
                    )
                
                fig.update_traces(diagonal_visible=False)
                fig.update_layout(height=800)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("최소 2개 이상의 변수를 선택해주세요.")
    
    else:  # 특성 간 관계
        st.markdown("### 특성 간 관계 분석")
        
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("X축 변수", numeric_cols if numeric_cols else [None])
        with col2:
            y_var = st.selectbox("Y축 변수", [col for col in numeric_cols if col != x_var] if numeric_cols else [None])
        
        if x_var and y_var:
            # 산점도 with 회귀선
            fig = px.scatter(
                df, x=x_var, y=y_var,
                color=target_col if problem_type == 'classification' else None,
                trendline="ols",
                title=f"{x_var} vs {y_var}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 회귀 통계
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(df[x_var].dropna(), df[y_var].dropna())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R²", f"{r_value**2:.3f}")
            with col2:
                st.metric("기울기", f"{slope:.3f}")
            with col3:
                st.metric("p-value", f"{p_value:.4f}")

with tab3:
    st.markdown("## 🎯 타겟 변수 분석")
    st.markdown(f"예측 대상인 **{target_col}** 변수를 심층 분석합니다.")
    
    if problem_type == 'classification':
        # 분류 문제의 타겟 분석
        st.markdown("### 클래스 분포")
        
        target_counts = df[target_col].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 막대 그래프
            fig = px.bar(
                x=target_counts.index,
                y=target_counts.values,
                title="클래스별 샘플 수",
                labels={'x': '클래스', 'y': '샘플 수'},
                color=target_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 파이 차트
            fig = px.pie(
                values=target_counts.values,
                names=target_counts.index,
                title="클래스 비율",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 클래스 불균형 확인
        imbalance_ratio = target_counts.min() / target_counts.max()
        
        if imbalance_ratio < 0.1:
            st.error(f"⚠️ 심각한 클래스 불균형! (비율: {imbalance_ratio:.2f})")
            st.markdown("""
            **해결 방법:**
            - **오버샘플링**: 소수 클래스를 증강 (SMOTE)
            - **언더샘플링**: 다수 클래스를 줄임
            - **클래스 가중치**: 모델에 가중치 부여
            - **앙상블**: 불균형에 강한 모델 사용
            """)
        elif imbalance_ratio < 0.3:
            st.warning(f"⚠️ 클래스 불균형 존재 (비율: {imbalance_ratio:.2f})")
        else:
            st.success(f"✅ 클래스가 비교적 균형잡혀 있습니다 (비율: {imbalance_ratio:.2f})")
        
        # 특성별 클래스 분포
        st.markdown("### 특성별 클래스 분포")
        
        selected_feature = st.selectbox(
            "분석할 특성 선택",
            numeric_cols if numeric_cols else []
        )
        
        if selected_feature:
            # 클래스별 분포 비교
            fig = go.Figure()
            for class_val in df[target_col].unique():
                class_data = df[df[target_col] == class_val][selected_feature]
                fig.add_trace(go.Box(
                    y=class_data,
                    name=str(class_val),
                    boxpoints='outliers'
                ))
            
            fig.update_layout(
                title=f"{selected_feature}의 클래스별 분포",
                yaxis_title=selected_feature,
                xaxis_title=target_col
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # 회귀 문제
        st.markdown("### 타겟 변수 분포")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("최소값", f"{df[target_col].min():.2f}")
        with col2:
            st.metric("최대값", f"{df[target_col].max():.2f}")
        with col3:
            st.metric("평균", f"{df[target_col].mean():.2f}")
        with col4:
            st.metric("중앙값", f"{df[target_col].median():.2f}")
        
        # 분포 시각화
        col1, col2 = st.columns(2)
        
        with col1:
            # 히스토그램
            fig = px.histogram(
                df, x=target_col,
                nbins=30,
                title=f"{target_col}의 분포"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Q-Q plot
            fig = go.Figure()
            
            # 정규 Q-Q plot 생성
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(df[target_col])))
            sample_quantiles = np.sort(df[target_col])
            
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='Q-Q Plot'
            ))
            
            # 대각선 추가
            min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
            max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='정규분포 기준선',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="Q-Q Plot (정규성 확인)",
                xaxis_title="이론적 분위수",
                yaxis_title="샘플 분위수"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("## ⚠️ 이상치 탐지")
    st.markdown("데이터의 이상치를 찾고 처리 방법을 결정합니다.")
    
    if numeric_cols:
        detection_method = st.radio(
            "이상치 탐지 방법",
            ["IQR 방법", "Z-Score 방법", "Isolation Forest"]
        )
        
        outlier_results = {}
        
        if detection_method == "IQR 방법":
            st.markdown("""
            ### IQR (Interquartile Range) 방법
            - **이상치 기준**: Q1 - 1.5×IQR 미만 또는 Q3 + 1.5×IQR 초과
            - **장점**: 간단하고 직관적
            - **단점**: 정규분포 가정
            """)
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_results[col] = len(outliers)
                
        elif detection_method == "Z-Score 방법":
            st.markdown("""
            ### Z-Score 방법
            - **이상치 기준**: |Z-Score| > 3
            - **장점**: 통계적 근거 명확
            - **단점**: 정규분포 가정 필요
            """)
            
            threshold = st.slider("Z-Score 임계값", 2.0, 4.0, 3.0, 0.1)
            
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = df[z_scores > threshold]
                outlier_results[col] = len(outliers)
        
        else:  # Isolation Forest
            st.markdown("""
            ### Isolation Forest
            - **원리**: 이상치는 고립시키기 쉬운 점
            - **장점**: 비선형 패턴 감지 가능
            - **단점**: 해석이 어려움
            """)
            
            from sklearn.ensemble import IsolationForest
            
            contamination = st.slider("오염도 (예상 이상치 비율)", 0.01, 0.2, 0.1)
            
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outliers_pred = iso_forest.fit_predict(df[numeric_cols].dropna())
            outlier_results["전체"] = (outliers_pred == -1).sum()
        
        # 결과 표시
        st.markdown("### 이상치 탐지 결과")
        
        outlier_df = pd.DataFrame({
            '특성': list(outlier_results.keys()),
            '이상치 개수': list(outlier_results.values()),
            '비율(%)': [v/len(df)*100 for v in outlier_results.values()]
        })
        
        # 막대 그래프
        fig = px.bar(
            outlier_df,
            x='특성',
            y='이상치 개수',
            title="특성별 이상치 개수",
            color='비율(%)',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(outlier_df, use_container_width=True)
        
        # 처리 방법
        with st.expander("💡 이상치 처리 방법"):
            st.markdown("""
            **1. 제거**
            - 명백한 오류인 경우
            - 샘플이 충분한 경우
            
            **2. 대체**
            - 중앙값/평균값으로 대체
            - 경계값으로 클리핑
            
            **3. 유지**
            - 실제 극단값인 경우
            - 트리 기반 모델 사용시
            
            **4. 별도 처리**
            - 이상치용 별도 모델
            - 규칙 기반 처리
            """)

with tab5:
    st.markdown("## 🔧 데이터 전처리")
    st.markdown("모델 학습을 위한 데이터 변환과 정제")
    
    preprocessing_type = st.radio(
        "전처리 작업 선택",
        ["결측값 처리", "스케일링", "인코딩", "특성 생성"]
    )
    
    if preprocessing_type == "결측값 처리":
        st.markdown("### 결측값 처리")
        
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if missing_cols:
            st.warning(f"결측값이 있는 컬럼: {', '.join(missing_cols)}")
            
            strategy = st.selectbox(
                "처리 방법",
                ["평균값 대체", "중앙값 대체", "최빈값 대체", "삭제", "전방 채우기", "후방 채우기"]
            )
            
            if st.button("결측값 처리 적용"):
                df_processed = df.copy()
                
                if strategy == "평균값 대체":
                    for col in missing_cols:
                        if col in numeric_cols:
                            df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                elif strategy == "중앙값 대체":
                    for col in missing_cols:
                        if col in numeric_cols:
                            df_processed[col].fillna(df_processed[col].median(), inplace=True)
                elif strategy == "최빈값 대체":
                    for col in missing_cols:
                        df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else df_processed[col].iloc[0], inplace=True)
                elif strategy == "삭제":
                    df_processed = df_processed.dropna()
                elif strategy == "전방 채우기":
                    df_processed = df_processed.fillna(method='ffill')
                else:  # 후방 채우기
                    df_processed = df_processed.fillna(method='bfill')
                
                st.session_state.data = df_processed
                st.success(f"✅ 결측값 처리 완료! (방법: {strategy})")
                st.rerun()
        else:
            st.success("✅ 결측값이 없습니다!")
    
    elif preprocessing_type == "스케일링":
        st.markdown("### 특성 스케일링")
        st.info("""
        💡 **스케일링이 필요한 이유:**
        - 특성들의 단위가 다를 때 (예: 나이 vs 연봉)
        - 거리 기반 알고리즘 사용시 (KNN, SVM)
        - 경사하강법 수렴 속도 향상
        """)
        
        scaler_type = st.selectbox(
            "스케일링 방법",
            ["StandardScaler (표준화)", "MinMaxScaler (정규화)", "RobustScaler (강건한 스케일링)"]
        )
        
        if st.button("스케일링 적용"):
            df_scaled = df.copy()
            
            if scaler_type.startswith("Standard"):
                scaler = StandardScaler()
                df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                st.success("✅ 표준화 완료 (평균=0, 표준편차=1)")
            elif scaler_type.startswith("MinMax"):
                scaler = MinMaxScaler()
                df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                st.success("✅ 정규화 완료 (범위: 0~1)")
            else:
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                st.success("✅ 강건한 스케일링 완료 (이상치에 강함)")
            
            st.session_state.data = df_scaled
            st.session_state.scaler = scaler  # 나중에 역변환을 위해 저장
            st.rerun()
    
    elif preprocessing_type == "인코딩":
        st.markdown("### 범주형 변수 인코딩")
        
        if categorical_cols:
            encoding_method = st.selectbox(
                "인코딩 방법",
                ["Label Encoding", "One-Hot Encoding"]
            )
            
            selected_cols = st.multiselect(
                "인코딩할 컬럼 선택",
                categorical_cols,
                default=categorical_cols
            )
            
            if st.button("인코딩 적용") and selected_cols:
                df_encoded = df.copy()
                
                if encoding_method == "Label Encoding":
                    for col in selected_cols:
                        le = LabelEncoder()
                        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    st.success("✅ Label Encoding 완료!")
                else:  # One-Hot Encoding
                    df_encoded = pd.get_dummies(df_encoded, columns=selected_cols)
                    st.success("✅ One-Hot Encoding 완료!")
                
                st.session_state.data = df_encoded
                st.rerun()
        else:
            st.info("범주형 변수가 없습니다.")
    
    else:  # 특성 생성
        st.markdown("### 특성 엔지니어링")
        st.info("새로운 특성을 생성하여 모델 성능을 향상시킵니다.")
        
        feature_type = st.selectbox(
            "생성할 특성 유형",
            ["다항식 특성", "교호작용", "비율/차이"]
        )
        
        if feature_type == "다항식 특성" and len(numeric_cols) >= 1:
            degree = st.slider("차수", 2, 3, 2)
            if st.button("다항식 특성 생성"):
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                poly_features = poly.fit_transform(df[numeric_cols[:2]])  # 처음 2개 특성만
                
                feature_names = poly.get_feature_names_out(numeric_cols[:2])
                for i, name in enumerate(feature_names[len(numeric_cols[:2]):]):  # 새 특성만
                    df[name] = poly_features[:, len(numeric_cols[:2]) + i]
                
                st.session_state.data = df
                st.success(f"✅ {len(feature_names) - len(numeric_cols[:2])}개의 다항식 특성 생성!")
                st.rerun()

# 진행 상황 저장
if st.button("💾 전처리 완료 및 다음 단계로", type="primary"):
    st.session_state.preprocessed = True
    st.success("✅ 데이터 탐색 및 전처리 완료!")
    st.info("이제 모델 학습 단계로 넘어갈 수 있습니다.")
    
    if st.button("🤖 모델 학습 페이지로 이동"):
        st.switch_page("pages/3_🤖_모델_학습.py")

# 사이드바
with st.sidebar:
    st.markdown("### 📊 현재 데이터 상태")
    
    if 'data' in st.session_state:
        df_info = st.session_state.data
        st.metric("샘플 수", len(df_info))
        st.metric("특성 수", len(df_info.columns) - 1)
        st.metric("타겟", target_col)
        
        # 진행 체크리스트
        st.markdown("### ✅ 진행 상황")
        st.checkbox("데이터 불러오기", value=True, disabled=True)
        st.checkbox("단변량 분석", value=False)
        st.checkbox("다변량 분석", value=False)
        st.checkbox("이상치 탐지", value=False)
        st.checkbox("전처리 완료", value='preprocessed' in st.session_state)
    
    with st.expander("💡 EDA 팁"):
        st.markdown("""
        1. **시각화를 활용하세요**
           - 그래프가 숫자보다 직관적입니다
        
        2. **가설을 세우세요**
           - "이 특성이 중요할 것 같다"
           - "이런 패턴이 있을 것 같다"
        
        3. **이상치를 주의깊게 보세요**
           - 때로는 가장 중요한 정보입니다
        
        4. **도메인 지식을 활용하세요**
           - 데이터의 맥락을 이해하면 더 좋은 인사이트를 얻습니다
        """)
