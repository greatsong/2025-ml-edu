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
    
    > **"모델은 데이터로부터 패턴을 찾아내는 과정을 거
