# app.py
import streamlit as st
import numpy as np
import joblib

# 모 델 불 러 오 기
model = joblib.load("model.pkl")
st.title(" ☆ 꽃 분 류 기 ( I r i s C l a s s i f i e r ) ")
st.write(" 입 력 값 을 기 반 으 로 꽃 의 종 류 를 예 측 합 니 다 . ")

# 사 용 자 입 력 받 기
sepal_length = st.slider("Sepal Length (cm)", 4.0 8.0, 5.1)
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
