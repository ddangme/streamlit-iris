# import streamlit as st
# import numpy as np

# # 패키지 import 에러 처리
# try:
#     from sklearn.datasets import load_iris
#     from sklearn.ensemble import RandomForestClassifier
# except ImportError as e:
#     st.error(f"필요한 패키지를 불러올 수 없습니다: {e}")
#     st.error("requirements.txt 파일을 확인해주세요.")
#     st.stop()

# # 모델 훈련 (매번 새로 훈련 - 테스트용)
# @st.cache_resource
# def create_model():
#     X, y = load_iris(return_X_y=True)
#     model = RandomForestClassifier(random_state=42)
#     model.fit(X, y)
#     return model

# # 모델 로드
# model = create_model()

# st.title("🌸 꽃 분류기 (Iris Classifier)")
# st.write("입력 값을 기반으로 꽃의 종류를 예측합니다.")

# # 사용자 입력
# sepal_length = st.slider("꽃받침 길이 (cm)", 4.0, 8.0, 5.1)
# sepal_width = st.slider("꽃받침 너비 (cm)", 2.0, 4.5, 3.5)
# petal_length = st.slider("꽃잎 길이 (cm)", 1.0, 7.0, 1.4)
# petal_width = st.slider("꽃잎 너비 (cm)", 0.1, 2.5, 0.2)

# # 예측
# input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
# prediction = model.predict(input_data)
# predicted_class = prediction[0]
# class_names = ['Setosa', 'Versicolor', 'Virginica']

# st.subheader("🎉 예측 결과:")
# st.write(f"**{class_names[predicted_class]}**")

# # 확률 표시
# prediction_proba = model.predict_proba(input_data)
# st.subheader("📊 각 클래스별 확률:")
# for i, class_name in enumerate(class_names):
#     st.write(f"{class_name}: {prediction_proba[0][i]:.3f}")


# app.py
import streamlit as st
import numpy as np
import joblib
# 모 델 불 러 오 기
model = joblib.load("model.pkl")
st.title(" ☆ 꽃 분 류 기 ( I r i s C l a s s i f i e r ) ")
st.write(" 입 력 값 을 기 반 으 로 꽃 의 종 류 를 예 측 합 니 다 . ")
# 사 용 자 입 력 받 기
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
# 예 측
prediction = model.predict(input_data)
predicted_class = prediction[0]
class_names = ['Setosa', 'Versicolor', 'Virginical']
# 예측
prediction = model.predict(input_data)
predicted_class = prediction[0]
class_names = ['Setosa', 'Versicolor', 'Virginica']
st.subheader( " ® 예 측 결 과 : " )
st.write(f"→ {class_names[predicted_class]}")
