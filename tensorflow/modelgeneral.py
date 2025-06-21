from prophet import Prophet
import pickle

def train_general_timeseries_model(df, model_path):
    """
    Entrena un modelo Prophet con el historial de ventas.
    Guarda el modelo en model_path.
    """
    if df.empty:
        raise ValueError("No hay datos para entrenar el modelo general.")

    model = Prophet(daily_seasonality=True)
    model.fit(df)
    # Guardar modelo entrenado
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model

def load_general_timeseries_model(model_path):
    """
    Carga un modelo Prophet desde disco.
    """
    import pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model