from bd.database import get_mongo_db
import pandas as pd
from bson import ObjectId
from datetime import datetime

def extract_and_transform_general_timeseries(tenant_id):
    """
    Extrae y transforma los datos generales de ventas agrupados por día.
    Retorna un DataFrame con las columnas: ds (fecha), y (ventas totales ese día).
    El historial llega hasta hoy (now), incluso si hay días sin ventas (rellena con 0).
    """
    db = get_mongo_db()
    try:
        ventas_data = list(db.ventas.find({"tenant": ObjectId(tenant_id)}))
    except Exception as e:
        return pd.DataFrame(columns=["ds", "y"])

    if not ventas_data:
        return pd.DataFrame(columns=["ds", "y"])

    df_ventas = pd.DataFrame(ventas_data)
    df_ventas["createdAT"] = pd.to_datetime(df_ventas["createdAT"], errors="coerce")
    df_grouped = (
        df_ventas
        .groupby(df_ventas["createdAT"].dt.date)
        .agg({"total": "sum"})
        .reset_index()
        .rename(columns={"createdAT": "ds", "total": "y"})
    )
    df_grouped["ds"] = pd.to_datetime(df_grouped["ds"])
    # --- NUEVO: completar hasta hoy con días faltantes ---
    if not df_grouped.empty:
        start_date = df_grouped["ds"].min()
        end_date = pd.to_datetime(datetime.now().date())
        full_range = pd.date_range(start=start_date, end=end_date, freq="D")
        # Reindexar para asegurar días sin ventas (rellena con 0)
        df_grouped = df_grouped.set_index("ds").reindex(full_range, fill_value=0).rename_axis("ds").reset_index()
        # Aquí cortamos a los últimos 90 días, siempre
        df_grouped = df_grouped.sort_values("ds").tail(90).reset_index(drop=True)
    return df_grouped