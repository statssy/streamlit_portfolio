# main.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="데이터 엔지니어링 파이프라인 시각화",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .pipeline-step {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    """샘플 데이터셋 생성"""
    np.random.seed(42)
    n_samples = 1000
    
    # 고객 데이터 생성
    customer_data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(35, 12, n_samples).astype(int),
        'income': np.random.normal(50000, 15000, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples).astype(int),
        'years_with_company': np.random.exponential(3, n_samples),
        'previous_purchases': np.random.poisson(5, n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'employment_status': np.random.choice(['Employed', 'Self-employed', 'Unemployed'], n_samples, p=[0.7, 0.2, 0.1])
    }
    
    df = pd.DataFrame(customer_data)
    
    # 타겟 변수 생성 (고객 만족도 점수 및 구매 여부)
    satisfaction_score = (
        0.3 * (df['income'] / 1000) +
        0.2 * df['credit_score'] / 10 +
        0.1 * df['age'] +
        0.2 * df['years_with_company'] +
        0.2 * df['previous_purchases'] +
        np.random.normal(0, 10, n_samples)
    )
    
    df['satisfaction_score'] = np.clip(satisfaction_score, 0, 100)
    df['will_purchase'] = (df['satisfaction_score'] > 60).astype(int)
    
    return df

def execute_sql_query(df, query):
    """SQL 쿼리 실행 함수"""
    try:
        # 임시 SQLite 데이터베이스 생성
        conn = sqlite3.connect(':memory:')
        df.to_sql('customer_data', conn, index=False, if_exists='replace')
        
        # 쿼리 실행
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        return result, None
    except Exception as e:
        return None, str(e)

def create_pipeline_diagram():
    """데이터 파이프라인 다이어그램 생성"""
    fig = go.Figure()
    
    # 파이프라인 단계들
    steps = [
        "원본 데이터",
        "SQL 전처리",
        "특성 엔지니어링",
        "모델 학습",
        "예측 결과"
    ]
    
    # 노드 위치
    x_positions = [1, 2, 3, 4, 5]
    y_positions = [1, 1, 1, 1, 1]
    
    # 노드 추가
    for i, (step, x, y) in enumerate(zip(steps, x_positions, y_positions)):
        color = '#1f77b4' if i < 3 else '#ff7f0e'
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=80, color=color, symbol='circle'),
            text=step,
            textposition="middle center",
            textfont=dict(color='white', size=10, family='Arial Black'),
            showlegend=False,
            hovertext=f"단계 {i+1}: {step}",
            hoverinfo="text"
        ))
    
    # 화살표 추가
    for i in range(len(x_positions) - 1):
        fig.add_annotation(
            x=x_positions[i+1], y=y_positions[i+1],
            ax=x_positions[i], ay=y_positions[i],
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowsize=2, arrowwidth=2,
            arrowcolor="#cccccc"
        )
    
    fig.update_layout(
        title="데이터 엔지니어링 파이프라인",
        xaxis=dict(range=[0.5, 5.5], showticklabels=False, showgrid=False),
        yaxis=dict(range=[0.5, 1.5], showticklabels=False, showgrid=False),
        plot_bgcolor='white',
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def main():
    # 메인 헤더
    st.markdown('<h1 class="main-header">🔧 데이터 엔지니어링 파이프라인 시각화</h1>', unsafe_allow_html=True)
    
    # 사이드바
    st.sidebar.title("📊 제어판")
    
    # 데이터 생성
    if 'original_data' not in st.session_state:
        st.session_state.original_data = create_sample_data()
        st.session_state.processed_data = st.session_state.original_data.copy()
    
    # 파이프라인 다이어그램 표시
    st.plotly_chart(create_pipeline_diagram(), use_container_width=True)
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 원본 데이터", 
        "🔍 SQL 전처리", 
        "⚙️ 특성 엔지니어링", 
        "🤖 머신러닝 모델", 
        "📈 예측 결과"
    ])
    
    with tab1:
        st.markdown('<div class="pipeline-step">', unsafe_allow_html=True)
        st.header("📁 원본 데이터")
        st.write("고객 데이터셋 (1,000개 샘플)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 고객 수", len(st.session_state.original_data))
        with col2:
            st.metric("평균 연령", f"{st.session_state.original_data['age'].mean():.1f}세")
        with col3:
            st.metric("평균 소득", f"${st.session_state.original_data['income'].mean():.0f}")
        with col4:
            st.metric("평균 신용점수", f"{st.session_state.original_data['credit_score'].mean():.0f}")
        
        st.dataframe(st.session_state.original_data.head(100), height=300)
        
        # 데이터 분포 시각화
        col1, col2 = st.columns(2)
        with col1:
            fig_age = px.histogram(st.session_state.original_data, x='age', title="연령 분포")
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            fig_income = px.histogram(st.session_state.original_data, x='income', title="소득 분포")
            st.plotly_chart(fig_income, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="pipeline-step">', unsafe_allow_html=True)
        st.header("🔍 SQL 전처리")
        st.write("SQL 쿼리를 사용하여 데이터를 필터링하고 변환하세요.")
        
        # 미리 정의된 쿼리 예시
        st.subheader("📝 쿼리 예시")
        query_examples = {
            "전체 데이터": "SELECT * FROM customer_data",
            "고소득 고객": "SELECT * FROM customer_data WHERE income > 60000",
            "젊은 고객": "SELECT * FROM customer_data WHERE age BETWEEN 25 AND 35",
            "우수 신용고객": "SELECT * FROM customer_data WHERE credit_score > 700",
            "지역별 평균": "SELECT region, AVG(income) as avg_income, AVG(age) as avg_age FROM customer_data GROUP BY region"
        }
        
        selected_example = st.selectbox("예시 쿼리 선택:", list(query_examples.keys()))
        
        # SQL 쿼리 입력
        sql_query = st.text_area(
            "SQL 쿼리 입력:",
            value=query_examples[selected_example],
            height=100,
            help="customer_data 테이블을 사용하세요. 사용 가능한 컬럼: customer_id, age, income, credit_score, years_with_company, previous_purchases, region, education, employment_status, satisfaction_score, will_purchase"
        )
        
        if st.button("🚀 쿼리 실행", type="primary"):
            result, error = execute_sql_query(st.session_state.original_data, sql_query)
            
            if error:
                st.error(f"SQL 오류: {error}")
            else:
                st.session_state.processed_data = result
                st.success(f"✅ 쿼리 실행 완료! {len(result)}개 행이 반환되었습니다.")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("처리된 행 수", len(result))
                with col2:
                    st.metric("컬럼 수", len(result.columns))
                with col3:
                    reduction = (1 - len(result)/len(st.session_state.original_data)) * 100
                    st.metric("데이터 감소율", f"{reduction:.1f}%")
                
                st.dataframe(result.head(100), height=300)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="pipeline-step">', unsafe_allow_html=True)
        st.header("⚙️ 특성 엔지니어링")
        
        if len(st.session_state.processed_data) > 0:
            df = st.session_state.processed_data.copy()
            
            st.subheader("📊 데이터 분포 및 상관관계")
            
            # 수치형 컬럼만 선택
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # 상관관계 히트맵
                    if len(numeric_cols) > 1:
                        corr_matrix = df[numeric_cols].corr()
                        fig_corr = px.imshow(
                            corr_matrix, 
                            title="특성 간 상관관계",
                            color_continuous_scale="RdBu",
                            aspect="auto"
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                
                with col2:
                    # 특성 중요도 (satisfaction_score 기준)
                    if 'satisfaction_score' in df.columns:
                        feature_importance = {}
                        for col in numeric_cols:
                            if col != 'satisfaction_score':
                                corr = abs(df[col].corr(df['satisfaction_score']))
                                feature_importance[col] = corr
                        
                        importance_df = pd.DataFrame(
                            list(feature_importance.items()), 
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=True)
                        
                        fig_importance = px.bar(
                            importance_df, 
                            x='Importance', 
                            y='Feature',
                            title="특성 중요도 (상관계수 기준)",
                            orientation='h'
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
            
            # 새로운 특성 생성
            st.subheader("🔧 파생 특성 생성")
            
            if st.checkbox("소득 대비 신용점수 비율 추가"):
                if 'income' in df.columns and 'credit_score' in df.columns:
                    df['income_credit_ratio'] = df['credit_score'] / (df['income'] / 1000)
                    st.success("✅ income_credit_ratio 특성이 추가되었습니다.")
            
            if st.checkbox("연령 그룹 카테고리 추가"):
                if 'age' in df.columns:
                    df['age_group'] = pd.cut(df['age'], 
                                           bins=[0, 25, 35, 45, 55, 100], 
                                           labels=['~25', '26-35', '36-45', '46-55', '55+'])
                    st.success("✅ age_group 특성이 추가되었습니다.")
            
            if st.checkbox("고가치 고객 여부 추가"):
                if 'income' in df.columns and 'credit_score' in df.columns:
                    df['high_value_customer'] = (
                        (df['income'] > df['income'].quantile(0.75)) & 
                        (df['credit_score'] > df['credit_score'].quantile(0.75))
                    ).astype(int)
                    st.success("✅ high_value_customer 특성이 추가되었습니다.")
            
            st.session_state.processed_data = df
            st.dataframe(df.head(100), height=300)
            
        else:
            st.warning("먼저 SQL 전처리 탭에서 데이터를 처리해주세요.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="pipeline-step">', unsafe_allow_html=True)
        st.header("🤖 머신러닝 모델")
        
        if len(st.session_state.processed_data) > 0:
            df = st.session_state.processed_data.copy()
            
            # 예측 유형 선택
            prediction_type = st.radio(
                "예측 유형 선택:",
                ["회귀 (만족도 점수 예측)", "분류 (구매 여부 예측)"]
            )
            
            # 수치형 특성만 선택
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if prediction_type == "회귀 (만족도 점수 예측)" and 'satisfaction_score' in df.columns:
                target = 'satisfaction_score'
                features = [col for col in numeric_features if col not in [target, 'customer_id', 'will_purchase']]
                
                if len(features) > 0:
                    selected_features = st.multiselect(
                        "예측에 사용할 특성 선택:",
                        features,
                        default=features[:min(5, len(features))]
                    )
                    
                    if len(selected_features) > 0 and st.button("🎯 회귀 모델 학습"):
                        # 데이터 준비
                        X = df[selected_features].fillna(df[selected_features].mean())
                        y = df[target].fillna(df[target].mean())
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # 모델 학습
                        models = {
                            'Linear Regression': LinearRegression(),
                            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
                        }
                        
                        results = {}
                        for name, model in models.items():
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            results[name] = {'model': model, 'mse': mse, 'predictions': y_pred}
                        
                        # 결과 표시
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 모델 성능")
                            for name, result in results.items():
                                st.metric(f"{name} MSE", f"{result['mse']:.2f}")
                        
                        with col2:
                            # 예측 vs 실제 산점도
                            best_model_name = min(results.keys(), key=lambda x: results[x]['mse'])
                            fig_pred = px.scatter(
                                x=y_test, 
                                y=results[best_model_name]['predictions'],
                                title=f"예측 vs 실제 ({best_model_name})",
                                labels={'x': '실제 값', 'y': '예측 값'}
                            )
                            fig_pred.add_shape(
                                type="line", line=dict(dash="dash"),
                                x0=y_test.min(), y0=y_test.min(),
                                x1=y_test.max(), y1=y_test.max()
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # 최고 모델 저장
                        st.session_state.best_model = results[best_model_name]['model']
                        st.session_state.feature_names = selected_features
                        st.session_state.model_type = 'regression'
                        
                        st.success(f"✅ 최고 성능 모델: {best_model_name} (MSE: {results[best_model_name]['mse']:.2f})")
            
            elif prediction_type == "분류 (구매 여부 예측)" and 'will_purchase' in df.columns:
                target = 'will_purchase'
                features = [col for col in numeric_features if col not in [target, 'customer_id', 'satisfaction_score']]
                
                if len(features) > 0:
                    selected_features = st.multiselect(
                        "예측에 사용할 특성 선택:",
                        features,
                        default=features[:min(5, len(features))]
                    )
                    
                    if len(selected_features) > 0 and st.button("🎯 분류 모델 학습"):
                        # 데이터 준비
                        X = df[selected_features].fillna(df[selected_features].mean())
                        y = df[target].fillna(0)
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # 모델 학습
                        models = {
                            'Logistic Regression': LogisticRegression(random_state=42),
                            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
                        }
                        
                        results = {}
                        for name, model in models.items():
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            results[name] = {'model': model, 'accuracy': accuracy, 'predictions': y_pred}
                        
                        # 결과 표시
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 모델 성능")
                            for name, result in results.items():
                                st.metric(f"{name} 정확도", f"{result['accuracy']:.3f}")
                        
                        with col2:
                            # 혼동 행렬
                            best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
                            from sklearn.metrics import confusion_matrix
                            cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
                            
                            fig_cm = px.imshow(
                                cm, 
                                title=f"혼동 행렬 ({best_model_name})",
                                labels=dict(x="예측", y="실제", color="Count"),
                                x=['구매 안함', '구매함'],
                                y=['구매 안함', '구매함']
                            )
                            st.plotly_chart(fig_cm, use_container_width=True)
                        
                        # 최고 모델 저장
                        st.session_state.best_model = results[best_model_name]['model']
                        st.session_state.feature_names = selected_features
                        st.session_state.model_type = 'classification'
                        
                        st.success(f"✅ 최고 성능 모델: {best_model_name} (정확도: {results[best_model_name]['accuracy']:.3f})")
        
        else:
            st.warning("먼저 SQL 전처리 탭에서 데이터를 처리해주세요.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="pipeline-step">', unsafe_allow_html=True)
        st.header("📈 예측 결과")
        
        if 'best_model' in st.session_state:
            st.subheader("🔮 새로운 고객 예측")
            
            # 예측을 위한 입력 폼
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                input_data = {}
                feature_names = st.session_state.feature_names
                
                for i, feature in enumerate(feature_names):
                    col = col1 if i % 2 == 0 else col2
                    
                    if feature == 'age':
                        input_data[feature] = col.slider("연령", 18, 80, 35)
                    elif feature == 'income':
                        input_data[feature] = col.slider("소득 ($)", 20000, 100000, 50000)
                    elif feature == 'credit_score':
                        input_data[feature] = col.slider("신용점수", 300, 850, 650)
                    elif feature == 'years_with_company':
                        input_data[feature] = col.slider("회사 근속연수", 0.0, 20.0, 3.0)
                    elif feature == 'previous_purchases':
                        input_data[feature] = col.slider("이전 구매 횟수", 0, 20, 5)
                    else:
                        # 기본값 사용
                        mean_val = st.session_state.processed_data[feature].mean()
                        input_data[feature] = col.number_input(
                            f"{feature}", 
                            value=float(mean_val),
                            help=f"평균값: {mean_val:.2f}"
                        )
                
                submitted = st.form_submit_button("🚀 예측 실행", type="primary")
                
                if submitted:
                    # 예측 수행
                    input_df = pd.DataFrame([input_data])
                    prediction = st.session_state.best_model.predict(input_df)[0]
                    
                    if st.session_state.model_type == 'regression':
                        st.success(f"🎯 예측된 고객 만족도 점수: **{prediction:.1f}점**")
                        
                        # 만족도 게이지 차트
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = prediction,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "고객 만족도 점수"},
                            delta = {'reference': 60},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 40], 'color': "lightgray"},
                                    {'range': [40, 60], 'color': "yellow"},
                                    {'range': [60, 80], 'color': "orange"},
                                    {'range': [80, 100], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 60
                                }
                            }
                        ))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                    else:  # classification
                        prob = st.session_state.best_model.predict_proba(input_df)[0]
                        prediction_text = "구매할 것" if prediction == 1 else "구매하지 않을 것"
                        confidence = max(prob) * 100
                        
                        st.success(f"🎯 예측 결과: 이 고객은 **{prediction_text}**입니다 (신뢰도: {confidence:.1f}%)")
                        
                        # 확률 바 차트
                        prob_df = pd.DataFrame({
                            '결과': ['구매 안함', '구매함'],
                            '확률': prob
                        })
                        
                        fig_prob = px.bar(
                            prob_df, 
                            x='결과', 
                            y='확률',
                            title="구매 확률 예측",
                            color='확률',
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig_prob, use_container_width=True)
            
            # 배치 예측
            st.subheader("📊 배치 예측")
            if st.button("전체 데이터셋에 대한 예측 수행"):
                df = st.session_state.processed_data.copy()
                X = df[st.session_state.feature_names].fillna(df[st.session_state.feature_names].mean())
                
                predictions = st.session_state.best_model.predict(X)
                
                if st.session_state.model_type == 'regression':
                    df['predicted_satisfaction'] = predictions
                    
                    # 예측 분포
                    fig_dist = px.histogram(
                        df, 
                        x='predicted_satisfaction',
                        title="예측된 만족도 점수 분포",
                        nbins=50
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # 상위/하위 고객
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🔝 만족도가 높을 것으로 예측되는 고객 (상위 10명)**")
                        top_customers = df.nlargest(10, 'predicted_satisfaction')[['customer_id', 'predicted_satisfaction'] + st.session_state.feature_names[:3]]
                        st.dataframe(top_customers)
                    
                    with col2:
                        st.write("**⚠️ 만족도가 낮을 것으로 예측되는 고객 (하위 10명)**")
                        bottom_customers = df.nsmallest(10, 'predicted_satisfaction')[['customer_id', 'predicted_satisfaction'] + st.session_state.feature_names[:3]]
                        st.dataframe(bottom_customers)
                
                else:  # classification
                    df['predicted_purchase'] = predictions
                    df['purchase_probability'] = st.session_state.best_model.predict_proba(X)[:, 1]
                    
                    # 예측 결과 분포
                    purchase_counts = df['predicted_purchase'].value_counts()
                    fig_pie = px.pie(
                        values=purchase_counts.values,
                        names=['구매 안함', '구매함'],
                        title="예측된 구매 의향 분포"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # 구매 확률이 높은 고객
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🛒 구매 확률이 높은 고객 (상위 10명)**")
                        likely_buyers = df.nlargest(10, 'purchase_probability')[['customer_id', 'purchase_probability'] + st.session_state.feature_names[:3]]
                        st.dataframe(likely_buyers)
                    
                    with col2:
                        st.write("**❌ 구매 확률이 낮은 고객 (하위 10명)**")
                        unlikely_buyers = df.nsmallest(10, 'purchase_probability')[['customer_id', 'purchase_probability'] + st.session_state.feature_names[:3]]
                        st.dataframe(unlikely_buyers)
        
        else:
            st.warning("먼저 머신러닝 모델 탭에서 모델을 학습해주세요.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 사이드바에 프로젝트 정보
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 프로젝트 정보")
    st.sidebar.markdown(
        """
        **데이터 엔지니어링 파이프라인 시각화**
        
        이 애플리케이션은 다음 기능을 제공합니다:
        
        - 🔹 샘플 고객 데이터 생성
        - 🔹 SQL 쿼리를 통한 데이터 전처리
        - 🔹 특성 엔지니어링 및 파생 변수 생성
        - 🔹 머신러닝 모델 학습 (회귀/분류)
        - 🔹 실시간 예측 및 배치 예측
        - 🔹 파이프라인 시각화
        """
    )
    
    # 데이터 통계
    if len(st.session_state.processed_data) > 0:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 현재 데이터 통계")
        st.sidebar.metric("처리된 데이터 행 수", len(st.session_state.processed_data))
        st.sidebar.metric("컬럼 수", len(st.session_state.processed_data.columns))
        
        if 'best_model' in st.session_state:
            st.sidebar.success("✅ 모델 학습 완료")
            st.sidebar.write(f"모델 유형: {st.session_state.model_type}")
            st.sidebar.write(f"사용 특성: {len(st.session_state.feature_names)}개")

if __name__ == "__main__":
    main()