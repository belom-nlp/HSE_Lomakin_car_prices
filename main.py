import streamlit as st
import pandas as pd
from scikit-learn.preprocessing import OneHotEncoder
from scikit-learn.compose import ColumnTransformer
from scikit-learn.preprocessing import StandardScaler

# Название
st.title("Predict Car Prices")
st.write('Choose parameters and find out optimal car price!')
st.sidebar.header('Input Parameters')
DATASET_PATH = "https://raw.githubusercontent.com/evgpat/edu_stepik_from_idea_to_mvp/main/datasets/cars.csv"

#processing dataset
df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=['mileage'])
df = df.drop(columns=['name'])
df['age'] = 2021 - df['year']
df['mileage_count'] = df['mileage'].apply(lambda x: float(x.split(' ')[0]) if type(x) == str else None)
df['mileage_count'][df.fuel=='Petrol'] = df['mileage_count'][df.fuel=='Petrol'].apply(lambda x: x * 0.75)
df['mileage_count'][df.fuel=='Diesel'] = df['mileage_count'][df.fuel=='Diesel'].apply(lambda x: x * 0.85)
df['engine_volume'] = df['engine'].apply(lambda x: float(x.split(' ')[0]) if type(x) == str else None)
df['max_power_count'] = df['max_power'].apply(lambda x: float(x.strip().split(' ')[0]) if type(x) == str and x!=' bhp' else None)
df['torque'] = df['torque'].apply(lambda x: x.replace('at', '@') if type(x) == str else None)
df['torque'] = df['torque'].apply(lambda x: x.replace('/', '@') if type(x) == str else None)
df['torque_1'] = df.torque.apply(lambda x: x.split('@', maxsplit=1)[0].lower() if type(x) == str else None)
def count_nm(x: str):
  if 'nm' in x:
    x = x.replace('nm', '')
    x = x.strip()
    if '(' in x:
      if '380' in x:
        return 380.0
    else:
      return float(x)
  elif 'kgm' in x:
    x = x.replace('kgm', '')
    x = x.strip()
    return float(x) * 9.8
  else:
    if float(x) > 100:
      return float(x)
    else:
      return float(x) * 9.8

df['torque_1'] = df['torque_1'].apply(lambda x: count_nm(x) if type(x) == str else None )
df['rpm'] = df['torque'].apply(lambda x: x.split('@', maxsplit=1)[1].lower() if type(x) == str and '@' in x else None)
def process_rpm(x: str):
  x = x.replace('(kgm@ rpm)', '')
  x = x.replace('rpm', '')
  x = x.strip()
  x = x.replace(',', '')
  x = x.replace('+@', '')
  x = x.replace('(nm@ )', '')
  if '-' in x:
    rpm = x.split('-')
    rpm[0] = int(rpm[0])
    rpm[1] = int(rpm[1])
  elif '~' in x:
    rpm = x.split('~')
    rpm[0] = int(rpm[0])
    rpm[1] = int(rpm[1])
  else:
    rpm = [int(x), int(x)]
  return rpm

df['rpm'] = df['rpm'].apply(lambda x: process_rpm(x) if type(x) == str else None)
df['rpm_min'] = df['rpm'].apply(lambda x: x[0] if x is not None else None)
df['rpm_max'] = df['rpm'].apply(lambda x: x[1] if x is not None else None)
x = df.drop(columns=['year', 'mileage', 'engine', 'max_power', 'torque', 'rpm'])
x = x.dropna()
y = x['selling_price']
x = x.drop(['selling_price'], axis=1)

categorical = ['seller_type', 'transmission', 'fuel', 'owner']
numeric_features = [col for col in x.columns if col not in categorical]

column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(), categorical),
    ('scaling', StandardScaler(), numeric_features)
])

X_scaled = column_transformer.fit_transform(x)

#model creation
from scikit-learn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor(n_estimators=3000, min_samples_split=16, min_samples_leaf=5, max_depth=12)
etr.fit(X_scaled, y)

def user_input_features():
    year = st.sidebar.slider('Please choose the year of car production', 1980, 2020, 2000)
    km_driven = st.sidebar.slider('How many kilometers has your car driven?', 0, 2500000, 1250000)
    fuel = st.sidebar.selectbox('Choose the type of fuel:', ['Diesel', 'Petrol', 'CNG', 'LPG', 'electric'])
    seller_type = st.sidebar.selectbox('Choose seller type:', ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.sidebar.selectbox('Choose transmission type:', ['Manual', 'Automatic'])
    owner = st.sidebar.selectbox('Chhose the number of owner:', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
    torque = st.sidebar.slider('Choose the value of torque:', [45, 800, 350])
    rpm_max = st.sidebar.slider('Choose maximum number of rotation per minute:', [500, 22000, 10000])
    rpm_min = st.sidebar.slider('Choose minimum number of rotation per minute:', [150, 22000, 10000])
    engine_volume = st.sidebar.slider('Choose the engine volume:', [500, 3800, 2000])
    seats = st.sidebar.selectbox('Choose the number of seats:', [1, 2, 3, 4, 5, 6])
    mileage_count = st.sidebar.slider('Choose mileage:', [0, 50, 25])
    max_power_count = st.sidebar.slider('Choose max power, bhp:', [30, 500, 250])

    data = {'seller_type': seller_type,
    'transmission': transmission,
    'fuel': fuel,
    'owner': owner,
    'km_driven': km_driven,
    'seats': seats,
    'age': 2021 - year,
    'mileage_count': mileage_count,
    'engine_volume': engine_volume,
    'max_power_count': max_power_count,
    'torque_1': torque,
    'rpm_min': rpm_min,
    'rpm_max': rpm_max
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Your car: characteristics')
st.write(df)

df = column_transformer.transform(df)
pred = etr.predict(df)

st.write('Your optimal price is', int(pred))

