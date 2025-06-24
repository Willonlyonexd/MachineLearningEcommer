import pandas as pd


def predict_general_timeseries(df, model, dias_prediccion=90):
    """
    Usa el modelo Prophet para predecir ventas totales por día (general).
    """
    if df.empty:
        return pd.DataFrame(columns=["fecha", "venta_total_predicho"])
    last_date = df["ds"].max()
    future = model.make_future_dataframe(periods=dias_prediccion)
    forecast = model.predict(future)
    pred_futuro = forecast[forecast["ds"] > last_date][["ds", "yhat"]].copy()
    pred_futuro = pred_futuro.rename(columns={"ds": "fecha", "yhat": "venta_total_predicho"})
    pred_futuro["fecha"] = pred_futuro["fecha"].dt.strftime("%Y-%m-%d")
    return pred_futuro

def predict_product_timeseries(df_producto, model, dias_prediccion=7):
    """
    Predice el monto vendido por día para un producto, usando Prophet.
    El historial se rellena hasta hoy (now) con 0 si faltan fechas.
    """
    df_producto = df_producto.copy()
    df_producto["ds"] = pd.to_datetime(df_producto["ds"])
    # Rellenar días faltantes hasta hoy
    hoy = pd.to_datetime(pd.Timestamp.now().date())
    idx = pd.date_range(start=df_producto["ds"].min(), end=hoy, freq="D")
    df_producto = df_producto.set_index("ds").reindex(idx, fill_value=0).reset_index()
    df_producto = df_producto.rename(columns={"index": "ds"})

    future = model.make_future_dataframe(periods=dias_prediccion)
    forecast = model.predict(future)

    last_date = df_producto["ds"].max()
    if not isinstance(last_date, pd.Timestamp):
        last_date_dt = pd.to_datetime(last_date)
    else:
        last_date_dt = last_date

    pred_futuro = forecast[forecast["ds"] > last_date_dt][["ds", "yhat"]].copy()
    pred_futuro = pred_futuro.head(dias_prediccion)
    pred_futuro.rename(columns={"ds": "fecha", "yhat": "prediccion"}, inplace=True)
    pred_futuro["fecha"] = pred_futuro["fecha"].dt.strftime("%Y-%m-%d")
    pred_futuro["prediccion"] = pred_futuro["prediccion"].round(2)
    return pred_futuro