from prophet import Prophet
import pickle
import os

def train_product_timeseries_models(df, model_dir):
    modelos = {}
    print(f"[LOG][TRAIN] Entrenando modelos Prophet para {df['producto_id'].nunique()} productos únicos")
    for producto_id in df["producto_id"].unique():
        df_prod = df[df["producto_id"] == producto_id][["ds", "y"]]
        ventas_count = len(df_prod)
        print(f"[LOG][TRAIN] Producto {producto_id}: {ventas_count} ventas en el historial para entrenamiento")
        if df_prod.empty:
            print(f"[LOG][TRAIN] Producto {producto_id}: sin ventas, se omite entrenamiento.")
            continue
        print(f"[LOG][TRAIN] Entrenando Prophet para producto {producto_id} ...")
        model = Prophet(daily_seasonality=True)
        model.fit(df_prod)
        modelo_path = os.path.join(model_dir, f"{producto_id}.pkl")
        os.makedirs(model_dir, exist_ok=True)
        with open(modelo_path, "wb") as f:
            pickle.dump(model, f)
        modelos[producto_id] = model
        print(f"[LOG][TRAIN] Modelo guardado en {modelo_path}")
    return modelos

def load_product_timeseries_model(model_dir, producto_id):
    modelo_path = os.path.join(model_dir, f"{producto_id}.pkl")
    print(f"[LOG][LOAD] Cargando modelo Prophet de {modelo_path}")
    with open(modelo_path, "rb") as f:
        model = pickle.load(f)
    return model