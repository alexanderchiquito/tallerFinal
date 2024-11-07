import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

def entrenar_modelo():
    df = pd.read_csv("data/casas_medellin.csv")
    
    columnas_relevantes = ["Precio", "NRO_HABITACIONES", "NRO_BAÑOS", "METROS_CUADRADOS"]
    for columna in columnas_relevantes:
        df[columna] = df[columna].replace("No disponible", pd.NA)
    df = df.dropna(subset=columnas_relevantes)

    df["Precio"] = df["Precio"].replace({"\\$": "", ",": "", " pesos": ""}, regex=True).astype(float)
    df["NRO_HABITACIONES"] = df["NRO_HABITACIONES"].astype(int)
    df["NRO_BAÑOS"] = df["NRO_BAÑOS"].astype(int)
    df["METROS_CUADRADOS"] = df["METROS_CUADRADOS"].astype(float)
    
    X = df[["NRO_HABITACIONES", "NRO_BAÑOS", "METROS_CUADRADOS"]]
    y = df["Precio"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Sequential()
    model.add(Dense(64, input_dim=3, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1)) 
    
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    # Entrenar el modelo
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    model.save('model_casas.h5')
    
    return model, mse  


def predecir_precio(casa):

    try:
        model = tf.keras.models.load_model('model_casas.h5')
    except:
        model, _ = entrenar_modelo()

    datos = np.array([[casa.nro_habitaciones, casa.nro_banos, casa.metros_cuadrados]])
    prediccion = model.predict(datos)
    
    return float(prediccion[0])

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer  # Import SimpleImputer
# from sklearn.preprocessing import LabelEncoder
# import numpy as np
# from models import Casa

# def entrenar_modelo():
    
#     #Cargar el dataset
#     df = pd.read_csv("data/casas_medellin.csv")
#     # 1. Preprocesamiento de datos
#     # Convertir 'Precio' a numérico
#     df.loc[:, 'Precio'] = df['Precio'].str.replace(' pesos', '').str.replace('.', '').astype(float)
    
#     # Imputar valores faltantes en 'Precio' ANTES de la detección de outliers
#     # Use SimpleImputer for 'Precio'
#     imputer_precio = SimpleImputer(strategy='median')
#     df.loc[:, 'Precio'] = imputer_precio.fit_transform(df[['Precio']])
    
#     # Detectar y eliminar outliers en 'Precio'
#     Q1 = df['Precio'].quantile(0.25)
#     Q3 = df['Precio'].quantile(0.75)
#     IQR = Q3 - Q1
#     df = df[(df['Precio'] >= Q1 - 1.5 * IQR) & (df['Precio'] <= Q3 + 1.5 * IQR)]
    
#     # Imputar valores faltantes (puedes usar estrategias más avanzadas si lo deseas)
#     # Convert to numeric before calculating median
#     df.loc[:, 'NRO_HABITACIONES'] = pd.to_numeric(df['NRO_HABITACIONES'], errors='coerce')
#     df.loc[:, 'NRO_BAÑOS'] = pd.to_numeric(df['NRO_BAÑOS'], errors='coerce')
#     df.loc[:, 'METROS_CUADRADOS'] = pd.to_numeric(df['METROS_CUADRADOS'], errors='coerce')

#     # Use SimpleImputer for numerical features
#     num_features = ['NRO_HABITACIONES', 'NRO_BAÑOS', 'METROS_CUADRADOS']
#     imputer_num = SimpleImputer(strategy='median')
#     df.loc[:, num_features] = imputer_num.fit_transform(df[num_features]) # Use .loc for assignment
    
#     # Codificación one-hot para variables categóricas (Barrio, Ciudad, Departamento)
#     # Handle NaN values in categorical columns before encoding
#     df.loc[:, ['BARRIO', 'CIUDAD', 'DEPARTAMENTO']] = df[['BARRIO', 'CIUDAD', 'DEPARTAMENTO']].fillna('Desconocido') # Replace NaN with 'Desconocido'

#     encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # Manejar categorías desconocidas
#     encoded_features = encoder.fit_transform(df[['BARRIO', 'CIUDAD', 'DEPARTAMENTO']])
#     encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['BARRIO', 'CIUDAD', 'DEPARTAMENTO']))
#     df = pd.concat([df, encoded_df], axis=1)
    
#     # 2. Dividir los datos
#     X = df.drop(['Precio', 'Título', 'BARRIO', 'CIUDAD', 'DEPARTAMENTO'], axis=1) # Eliminamos las columnas originales
#     y = df['Precio']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Impute missing values in X_train and y_train 
#     imputer_X_train = SimpleImputer(strategy='median')
#     X_train = imputer_X_train.fit_transform(X_train)
    
#     imputer_y_train = SimpleImputer(strategy='median')
#     # Convert y_train to a NumPy array before reshaping
#     y_train = imputer_y_train.fit_transform(y_train.values.reshape(-1, 1))  # Reshape y_train for SimpleImputer
    
#     y_train = y_train.ravel() # Flatten y_train back to 1D
    
    
#     # 3. Escalar las características numéricas
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
    
#     # 4. Entrenar el modelo (usando RandomForestRegressor)
#     model = RandomForestRegressor(random_state=42)  # Cambiamos el modelo
#     model.fit(X_train, y_train)
    
#     # 5. Evaluar el modelo
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
    
#     return model, mse, r2, scaler

# # Función para predecir el precio de una casa
# def predecir_precio(model, scaler, casa: Casa):
#     # Crear un codificador para las variables categóricas
#     encoder_barrio = LabelEncoder()
#     encoder_ciudad = LabelEncoder()
#     encoder_departamento = LabelEncoder()

#     # Ajustar el encoder con las categorías y transformar las variables
#     casa.barrio = encoder_barrio.fit_transform([casa.barrio])[0]
#     casa.ciudad = encoder_ciudad.fit_transform([casa.ciudad])[0]
#     casa.departamento = encoder_departamento.fit_transform([casa.departamento])[0]
    
#     datos = scaler.transform([[casa.barrio, casa.ciudad, casa.departamento, casa.nro_habitaciones, casa.nro_banos, casa.metros_cuadrados]])  
#     prediccion = model.predict(datos) 
#     return prediccion[0]  

