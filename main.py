import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="머신러닝 교육 플랫폼",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 메인 타이틀
st.markdown('<h1 class="main-header">🎓 머신러닝 교육 플랫폼</h1>', unsafe_allow_html=True)
st.markdown("### 데이터에서 인사이트까지, 한 걸음씩 배워가는 머신러닝 여정")

# 소개 섹션
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    > **"데이터는 21세기의 원유입니다. 하지만 정제하지 않으면 쓸모가 없죠."**
    > 
    > 이 플랫폼은 여러분이 데이터를 의미 있는 지식으로 변환하는 과정을 
    > 단계별로 체험하고 학습할 수 있도록 설계되었습니다.
    """)

st.divider()

# 학습 로드맵
st.markdown("## 🗺️ 학습 로드맵")

# 탭으로 구성
tab1, tab2, tab3 = st.tabs(["🚀 빠른 시작", "📚 학습 경로", "💡 핵심 개념"])

with tab1:
    st.markdown("### 5분 안에 첫 모델 만들기!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        #### 📊 Step 1: 데이터 준비
        - 내장된 Iris 데이터셋으로 시작
        - 또는 자신의 CSV 파일 업로드
        - 데이터의 형태와 의미 이해하기
        
        #### 🔍 Step 2: 데이터 탐색 (EDA)
        - 시각화로 패턴 발견하기
        - 상관관계 분석
        - 이상치와 결측값 확인
        """)
    
    with col2:
        st.markdown("""
        #### 🤖 Step 3: 모델 학습
        - 다양한 알고리즘 중 선택
        - 하이퍼파라미터 조정
        - 실시간 학습 과정 관찰
        
        #### 📈 Step 4: 결과 평가
        - 성능 지표 이해하기
        - 예측 결과 시각화
        - 개선 방향 찾기
        """)
    
    st.info("💡 **추천**: 처음이라면 '1_📊_데이터_준비' 페이지부터 순서대로 진행하세요!")

with tab2:
    st.markdown("### 체계적인 학습 경로")
    
    # 학습 경로 시각화
    learning_path = {
        "단계": ["1. 기초", "2. 데이터 이해", "3. 전처리", "4. 모델링", "5. 평가", "6. 고급"],
        "주요 내용": [
            "머신러닝이란?",
            "EDA 기법",
            "특성 공학",
            "알고리즘 선택",
            "성능 평가",
            "앙상블 & 튜닝"
        ],
        "예상 시간": [10, 20, 15, 30, 20, 25],
        "난이도": [1, 2, 3, 4, 3, 5]
    }
    
    df_path = pd.DataFrame(learning_path)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_path["단계"],
        y=df_path["예상 시간"],
        mode='lines+markers',
        name='학습 시간(분)',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_path["단계"],
        y=df_path["난이도"] * 5,
        mode='lines+markers',
        name='난이도',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=10),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="학습 경로별 시간과 난이도",
        xaxis_title="학습 단계",
        yaxis_title="예상 시간 (분)",
        yaxis2=dict(
            title="난이도",
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 단계별 상세 설명
    for i, row in df_path.iterrows():
        with st.expander(f"{row['단계']}: {row['주요 내용']}"):
            if i == 0:
                st.markdown("""
                **학습 목표**: 머신러닝의 기본 개념 이해
                - 지도학습 vs 비지도학습
                - 분류 vs 회귀
                - 훈련 데이터와 테스트 데이터
                
                **핵심 질문**: "컴퓨터가 어떻게 데이터로부터 학습할까?"
                """)
            elif i == 1:
                st.markdown("""
                **학습 목표**: 데이터를 다각도로 분석하는 능력
                - 기술통계량 해석
                - 분포 시각화
                - 변수 간 관계 파악
                
                **핵심 질문**: "이 데이터에서 어떤 패턴을 발견할 수 있을까?"
                """)
            elif i == 2:
                st.markdown("""
                **학습 목표**: 모델이 학습하기 좋은 데이터 만들기
                - 결측값 처리
                - 스케일링과 정규화
                - 범주형 변수 인코딩
                
                **핵심 질문**: "왜 데이터를 전처리해야 할까?"
                """)
            elif i == 3:
                st.markdown("""
                **학습 목표**: 적절한 알고리즘 선택과 적용
                - 각 알고리즘의 장단점
                - 하이퍼파라미터의 역할
                - 과적합과 과소적합
                
                **핵심 질문**: "어떤 상황에서 어떤 모델을 선택해야 할까?"
                """)
            elif i == 4:
                st.markdown("""
                **학습 목표**: 모델 성능을 정확히 평가하기
                - 평가 지표 선택
                - 교차 검증
                - 오류 분석
                
                **핵심 질문**: "이 모델을 실제로 사용해도 될까?"
                """)
            else:
                st.markdown("""
                **학습 목표**: 성능 극대화를 위한 고급 기법
                - 앙상블 방법
                - 자동 하이퍼파라미터 튜닝
                - 특성 중요도 분석
                
                **핵심 질문**: "어떻게 하면 더 나은 성능을 얻을 수 있을까?"
                """)

with tab3:
    st.markdown("### 꼭 알아야 할 핵심 개념들")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h4 class="sub-header">🎯 분류 vs 회귀</h4>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>분류 (Classification)</b><br>
        • 범주를 예측하는 문제<br>
        • 예: 이메일이 스팸인지 아닌지<br>
        • 붓꽃의 종류 구분<br>
        • 평가: 정확도, 정밀도, 재현율
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <b>회귀 (Regression)</b><br>
        • 연속적인 값을 예측하는 문제<br>
        • 예: 내일의 기온 예측<br>
        • 주택 가격 예측<br>
        • 평가: R², MAE, RMSE
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h4 class="sub-header">⚖️ 편향 vs 분산</h4>', unsafe_allow_html=True)
        st.markdown("""
        <div class="warning-box">
        <b>과적합 (Overfitting)</b><br>
        • 훈련 데이터에 너무 최적화<br>
        • 새로운 데이터에서 성능 저하<br>
        • 해결: 정규화, 더 많은 데이터
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <b>과소적합 (Underfitting)</b><br>
        • 모델이 너무 단순함<br>
        • 패턴을 제대로 학습 못함<br>
        • 해결: 복잡한 모델, 더 많은 특성
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h4 class="sub-header">📊 데이터 분할</h4>', unsafe_allow_html=True)
        
        # 데이터 분할 시각화
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['전체 데이터'],
            y=[100],
            name='전체',
            marker_color='lightgray'
        ))
        fig.add_trace(go.Bar(
            x=['훈련'],
            y=[70],
            name='훈련 세트 (70%)',
            marker_color='#1f77b4'
        ))
        fig.add_trace(go.Bar(
            x=['검증'],
            y=[15],
            name='검증 세트 (15%)',
            marker_color='#ff7f0e'
        ))
        fig.add_trace(go.Bar(
            x=['테스트'],
            y=[15],
            name='테스트 세트 (15%)',
            marker_color='#2ca02c'
        ))
        
        fig.update_layout(
            title="데이터 분할 비율",
            yaxis_title="비율 (%)",
            showlegend=True,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="success-box">
        <b>왜 데이터를 나누나요?</b><br>
        • <b>훈련 세트</b>: 모델이 패턴을 학습<br>
        • <b>검증 세트</b>: 하이퍼파라미터 조정<br>
        • <b>테스트 세트</b>: 최종 성능 평가<br><br>
        💡 테스트 세트는 마지막에 단 한 번만 사용!
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h4 class="sub-header">🔄 교차 검증</h4>', unsafe_allow_html=True)
        st.markdown("""
        데이터를 여러 번 다르게 나누어 평가하는 방법:
        - **더 신뢰할 수 있는** 성능 평가
        - **데이터를 효율적으로** 활용
        - **과적합 위험** 감소
        """)

st.divider()

# 실습 환경 체크
st.markdown("## 🛠️ 실습 환경 체크")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Python 버전", "3.8+", "✅")
    st.metric("Streamlit 버전", st.__version__, "✅")

with col2:
    st.metric("데이터셋", "2개 내장", "✅")
    st.metric("알고리즘", "10+ 종류", "✅")

with col3:
    st.metric("시각화 도구", "준비됨", "✅")
    st.metric("실시간 피드백", "활성화", "✅")

# 시작 안내
st.divider()
st.markdown("## 🚀 시작하기")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    ### 준비되셨나요?
    
    왼쪽 사이드바에서 원하는 페이지를 선택하여 학습을 시작하세요!
    
    1. **데이터 준비** → 데이터 불러오기 및 이해
    2. **데이터 탐색** → 시각화와 패턴 발견
    3. **모델 학습** → 알고리즘 선택과 학습
    4. **평가 및 개선** → 성능 평가와 튜닝
    
    각 단계마다 **교육적 설명**과 **인터랙티브 실습**이 준비되어 있습니다.
    """)
    
    if st.button("🎯 첫 번째 실습 시작하기", type="primary", use_container_width=True):
        st.switch_page("pages/1_📊_데이터_준비.py")

# 팁 섹션
with st.expander("💡 학습 팁"):
    st.markdown("""
    ### 효과적인 학습을 위한 팁
    
    1. **실험을 두려워하지 마세요**: 파라미터를 바꿔가며 결과가 어떻게 변하는지 관찰하세요.
    
    2. **왜?라고 질문하세요**: 각 단계가 왜 필요한지, 결과가 왜 그렇게 나왔는지 생각해보세요.
    
    3. **시각화를 활용하세요**: 그래프와 차트는 복잡한 개념을 이해하는 지름길입니다.
    
    4. **작은 것부터 시작하세요**: 간단한 모델부터 시작해 점진적으로 복잡도를 높이세요.
    
    5. **문서화하세요**: 실험 결과와 배운 점을 기록하면 나중에 큰 도움이 됩니다.
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Made with ❤️ for ML Education | 궁금한 점이 있다면 언제든 실험해보세요!</p>
</div>
""", unsafe_allow_html=True)
