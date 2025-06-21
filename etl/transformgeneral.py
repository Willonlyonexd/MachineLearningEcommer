from bd.database import get_mongo_db
import pandas as pd
from bson import ObjectId

def extract_and_transform_general_timeseries(tenant_id):
    """
    Extrae y transforma los datos generales de ventas agrupados por día.
    Retorna un DataFrame con las columnas: ds (fecha), y (ventas totales ese día).
    """
    db = get_mongo_db()
    try:
        ventas_data = list(db.ventas.find({"tenant": ObjectId(tenant_id)}))
    except Exception as e:
        # Si tenant_id no es un ObjectId válido, retorna DataFrame vacío
        return pd.DataFrame(columns=["ds", "y"])

    if not ventas_data:
        return pd.DataFrame(columns=["ds", "y"])

    df_ventas = pd.DataFrame(ventas_data)
    # Asegura formato datetime
    df_ventas["createdAT"] = pd.to_datetime(df_ventas["createdAT"], errors="coerce")
    # Agrupa por día y suma el total vendido por día
    df_grouped = (
        df_ventas
        .groupby(df_ventas["createdAT"].dt.date)
        .agg({"total": "sum"})
        .reset_index()
        .rename(columns={"createdAT": "ds", "total": "y"})
    )
    # Convierte ds a datetime
    df_grouped["ds"] = pd.to_datetime(df_grouped["ds"])
    return df_grouped