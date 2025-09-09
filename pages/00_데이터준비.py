import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# 페이지 설정
st.set_page_config(
    page_title="데이터 준비",
    page_icon="📊",
    layout="wide"
)

# 세션 상태 초기화
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_type' not in st.session_state:
    st.session_state.data_type = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

st.title("📊 Step 1: 데이터 준비")
st.markdown("### 모든 머신러닝 프로젝트의 시작점")

# 교육 콘텐츠
with st.expander("📚 왜 데이터 준비가 중요한가요?", expanded=True):
    st.markdown("""
    ### 🎯 데이터 준비의 중요성
    
    > **"Garbage In, Garbage Out"** - 나쁜 데이터로는 좋은 모델을 만들 수 없습니다.
    
    머신러닝 프로젝트의 **80%는 데이터 준비**에 소요됩니다. 왜일까요?
    
    1. **데이터의 품질이 모델의 성능을 결정합니다**
       - 잘못된 데이터는 잘못된 패턴을 학습시킵니다
       - 불완전한 데이터는 예측의 신뢰도를 떨어뜨립니다
    
    2. **실제 세계의 데이터는 지저분합니다**
       - 결측값 (비어있는 데이터)
       - 이상치 (극단적으로 다른 값)
       - 형식 불일치 (같은 의미, 다른 표현)
    
    3. **목적에 맞는 데이터 선택이 필요합니다**
       - 예측하려는 것(target)이 무엇인가?
       - 어떤 정보(features)가 예측에 도움이 될까?
    """)

st.divider()

# 데이터 소스 선택
st.markdown("## 🗂️ 데이터 소스 선택")

tab1, tab2, tab3 = st.tabs(["📁 내장 데이터셋", "📤 파일 업로드", "🔨 데이터 생성"])

with tab1:
    st.markdown("### 학습용 데이터셋")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌺 Iris 데이터셋 (분류)")
        st.markdown("""
        **붓꽃 품종 분류 데이터**
        - 150개 샘플, 3가지 품종
        - 4개 특성: 꽃잎/꽃받침 길이와 너비
        - 입문자에게 완벽한 데이터셋
        """)
        
        if st.button("Iris 데이터 불러오기", type="primary", use_container_width=True):
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df['target'] = iris.target
            df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
            
            st.session_state.data = df
            st.session_state.data_type = 'classification'
            st.session_state.target_column = 'species'
            st.success("✅ Iris 데이터셋을 불러왔습니다!")
    
    with col2:
        st.markdown("#### 🌡️ 기온 예측 데이터 (회귀)")
        st.markdown("""
        **합성 기상 데이터**
        - 365일 시뮬레이션 데이터
        - 습도, 기압, 풍속 등으로 기온 예측
        - 시계열 패턴 포함
        """)
        
        if st.button("기온 데이터 생성하기", type="primary", use_container_width=True):
            # 합성 기온 데이터 생성
            np.random.seed(42)
            days = 365
            dates = pd.date_range(start='2024-01-01', periods=days)
            
            # 계절성을 포함한 기온 생성
            seasonal_temp = 15 + 10 * np.sin(2 * np.pi * np.arange(days) / 365 - np.pi/2)
            daily_variation = np.random.normal(0, 3, days)
            temperature = seasonal_temp + daily_variation
            
            # 다른 기상 요소들
            humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(days) / 365) + np.random.normal(0, 10, days)
            humidity = np.clip(humidity, 20, 100)
            
            pressure = 1013 + 10 * np.cos(2 * np.pi * np.arange(days) / 365) + np.random.normal(0, 5, days)
            
            wind_speed = np.abs(np.random.gamma(2, 2, days))
            
            # 기온과 상관관계가 있는 특성 추가
            solar_radiation = 200 + 100 * np.sin(2 * np.pi * np.arange(days) / 365 - np.pi/2) + np.random.normal(0, 20, days)
            solar_radiation = np.clip(solar_radiation, 50, 350)
            
            df = pd.DataFrame({
                'date': dates,
                'humidity': humidity,
                'pressure': pressure,
                'wind_speed': wind_speed,
                'solar_radiation': solar_radiation,
                'temperature': temperature
            })
            
            st.session_state.data = df
            st.session_state.data_type = 'regression'
            st.session_state.target_column = 'temperature'
            st.success("✅ 기온 예측 데이터를 생성했습니다!")

with tab2:
    st.markdown("### 자신의 데이터 업로드")
    
    st.info("""
    💡 **지원 형식**: CSV, Excel (xlsx, xls)
    
    **데이터 요구사항**:
    - 첫 번째 행은 컬럼명이어야 합니다
    - 최소 50개 이상의 샘플 권장
    - 예측할 타겟 컬럼이 있어야 합니다
    """)
    
    uploaded_file = st.file_uploader(
        "파일을 선택하세요",
        type=['csv', 'xlsx', 'xls'],
        help="CSV 또는 Excel 파일을 업로드하세요"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ 파일을 성공적으로 불러왔습니다! ({len(df)} 행, {len(df.columns)} 열)")
            
            # 데이터 미리보기
            st.markdown("#### 데이터 미리보기")
            st.dataframe(df.head())
            
            # 타겟 컬럼 선택
            st.markdown("#### 예측할 타겟 선택")
            target_col = st.selectbox(
                "어떤 값을 예측하고 싶으신가요?",
                df.columns,
                help="모델이 예측할 목표 변수를 선택하세요"
            )
            
            # 문제 유형 자동 판단
            unique_values = df[target_col].nunique()
            if unique_values <= 20:
                problem_type = st.radio(
                    "문제 유형",
                    ["classification", "regression"],
                    format_func=lambda x: "분류 (카테고리 예측)" if x == "classification" else "회귀 (숫자 예측)",
                    help=f"타겟 컬럼에 {unique_values}개의 고유값이 있습니다"
                )
            else:
                problem_type = "regression"
                st.info(f"타겟 컬럼에 {unique_values}개의 고유값이 있어 회귀 문제로 설정됩니다")
            
            if st.button("이 데이터 사용하기", type="primary"):
                st.session_state.data = df
                st.session_state.data_type = problem_type
                st.session_state.target_column = target_col
                st.success("✅ 데이터를 저장했습니다!")
                
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {str(e)}")

with tab3:
    st.markdown("### 학습용 합성 데이터 생성")
    
    st.markdown("""
    실험을 위한 다양한 패턴의 데이터를 생성할 수 있습니다.
    각 패턴은 특정 머신러닝 개념을 학습하는데 최적화되어 있습니다.
    """)
    
    pattern_type = st.selectbox(
        "데이터 패턴 선택",
        ["선형 분리 가능 (Linear Separable)", 
         "비선형 패턴 (Non-linear)",
         "클러스터 (Clusters)",
         "나선형 (Spiral)",
         "동심원 (Concentric Circles)"]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("샘플 수", 100, 1000, 300)
        noise = st.slider("노이즈 레벨", 0.0, 0.5, 0.1)
    
    with col2:
        if pattern_type in ["클러스터 (Clusters)"]:
            n_classes = st.slider("클래스 수", 2, 5, 3)
        else:
            n_classes = 2
        
        random_state = st.number
