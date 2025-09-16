import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression

st.title('🚢 Titanic Survival Prediction App')


Titanic = pd.read_csv(r"C:\Users\jeshw\Desktop\Titanic_train_cleaned.csv")

feature_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
                   'Sex_female', 'Sex_male',
                   'Embarked_C', 'Embarked_Q', 'Embarked_S']


X = Titanic[feature_columns]
Y = Titanic['Survived']
clf = LogisticRegression()
clf.fit(X, Y)


st.sidebar.header('Enter passenger details:')
def passenger_information():
    Pclass = int(st.sidebar.selectbox('Passenger class', (1, 2, 3)))
    Age = float(st.sidebar.number_input("Age", min_value=0.0, max_value=100.0, value=30.0))
    SibSp = int(st.sidebar.selectbox('Siblings/Spouses aboard', range(0, 9)))
    Parch = int(st.sidebar.selectbox('Parents/Children aboard', range(0, 7)))
    Fare = float(st.sidebar.number_input("Fare paid (£)", min_value=0.0, value=50.0))

    sex = st.sidebar.radio("Sex", ['male', 'female'])
    Sex_female = 1 if sex == 'female' else 0
    Sex_male = 1 if sex == 'male' else 0

    embark = st.sidebar.radio("Port of Embarkation", ['Cherbourg', 'Queenstown', 'Southampton'])
    Embarked_C = 1 if embark == 'Cherbourg' else 0
    Embarked_Q = 1 if embark == 'Queenstown' else 0
    Embarked_S = 1 if embark == 'Southampton' else 0

    data = {
        'Pclass': Pclass,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Sex_female': Sex_female,
        'Sex_male': Sex_male,
        'Embarked_C': Embarked_C,
        'Embarked_Q': Embarked_Q,
        'Embarked_S': Embarked_S
    }
    return pd.DataFrame([data])


df = passenger_information()
df = df[feature_columns]  
st.subheader('Passenger Information')
st.write(df)


prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)[0][1]


st.subheader('Predicted Result')
st.write('✅ Survived' if prediction == 1 else '❌ Did not survive')

st.subheader('Prediction Probability')
st.write(f"Survival Probability: {prediction_proba:.2%}")
