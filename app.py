import streamlit as st
import numpy as np
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 모델 파일이 없으면 자동으로 생성
@st.cache_resource
def load_or_create_model():
    if not os.path.exists("model.pkl"):
        # 모델 훈련
        X, y = load_iris(return_X_y=True)
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        joblib.dump(model, 'model.pkl')
        return model
    else:
        return joblib.load("model.pkl")

# 모델 로드
model = load_or_create_model()

st.title("🌸 꽃 분류기 (Iris Classifier)")
st.write("입력 값을 기반으로 꽃의 종류를 예측합니다.")

# 사이드바에 설명 추가
st.sidebar.header("🌺 Iris 데이터셋 정보")
st.sidebar.write("""
- **Setosa**: 꽃잎이 작고 둥근 품종
- **Versicolor**: 중간 크기의 품종  
- **Virginica**: 꽃잎이 크고 긴 품종
""")

# 메인 입력 영역
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌿 꽃받침 (Sepal)")
    sepal_length = st.slider("꽃받침 길이 (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width = st.slider("꽃받침 너비 (cm)", 2.0, 4.5, 3.5, 0.1)

with col2:
    st.subheader("🌹 꽃잎 (Petal)")
    petal_length = st.slider("꽃잎 길이 (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width = st.slider("꽃잎 너비 (cm)", 0.1, 2.5, 0.2, 0.1)

# 예측 버튼
if st.button("🔍 꽃 종류 예측하기", type="primary"):
    # 입력 데이터 준비
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # 예측
    prediction = model.predict(input_data)
    predicted_class = prediction[0]
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    
    # 예측 확률
    prediction_proba = model.predict_proba(input_data)
    
    # 결과 출력
    st.success(f"🎉 예측 결과: **{class_names[predicted_class]}**")
    
    # 확률 차트
    st.subheader("📊 각 클래스별 예측 확률")
    prob_data = {
        '품종': class_names,
        '확률': prediction_proba[0]
    }
    st.bar_chart(prob_data, x='품종', y='확률')
    
    # 상세 확률 정보
    with st.expander("📈 상세 확률 정보"):
        for i, class_name in enumerate(class_names):
            confidence = prediction_proba[0][i] * 100
            st.write(f"**{class_name}**: {confidence:.1f}%")

# 푸터
st.markdown("---")
st.markdown("💡 **팁**: 슬라이더를 조정해서 다양한 꽃의 특성을 입력해보세요!")
