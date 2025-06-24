### etl/transformextra.py

import pandas as pd
from bd.database import get_mongo_db
from bson import ObjectId

def obtener_ventas_recientes(tenant_id: str, dias: int = 15) -> pd.DataFrame:
    db = get_mongo_db()
    ventas_data = list(db.ventas.find({"tenant": ObjectId(tenant_id)}))
    if not ventas_data:
        return pd.DataFrame(columns=["fecha", "n_ventas", "total", "ticket_promedio"])

    df = pd.DataFrame(ventas_data)
    df["createdAT"] = pd.to_datetime(df["createdAT"], errors="coerce")
    df = df.dropna(subset=["createdAT"])
    df["fecha"] = df["createdAT"].dt.date

    resumen = df.groupby("fecha").agg(
        n_ventas=("cliente", "count"),
        total=("total", "sum")
    ).reset_index()

    resumen["ticket_promedio"] = (resumen["total"] / resumen["n_ventas"]).round(2)
    resumen = resumen.sort_values("fecha", ascending=False).head(dias)
    return resumen.sort_values("fecha")


### estadisticas/general.py

def resumen_modelo(predicciones: list, historial_dias: int, dias_prediccion: int) -> dict:
    if not predicciones:
        return {}

    valores = [p["venta_total_predicho"] for p in predicciones]
    fechas = [p["fecha"] for p in predicciones]
    resumen = {
        "historial_usado_dias": historial_dias,
        "dias_proyectados": dias_prediccion,
        "max_venta_predicha": round(max(valores), 2),
        "min_venta_predicha": round(min(valores), 2),
        "fecha_max": fechas[valores.index(max(valores))],
        "fecha_min": fechas[valores.index(min(valores))]
    }
    return resumen


def alerta_tendencia_anomala(predicciones: list, umbral_pendiente: float = -10.0) -> dict:
    if len(predicciones) < 2:
        return {"alerta": False, "detalle": "Predicciones insuficientes"}

    valores = [p["venta_total_predicho"] for p in predicciones[-7:]]
    fechas = [p["fecha"] for p in predicciones[-7:]]

    pendientes = [valores[i+1] - valores[i] for i in range(len(valores) - 1)]
    tendencia = sum(pendientes)

    alerta = tendencia <= umbral_pendiente
    return {
        "alerta": alerta,
        "detalle": f"Tendencia en últimos 7 días: {round(tendencia, 2)}",
        "desde": fechas[0],
        "hasta": fechas[-1]
    }
