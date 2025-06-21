from bd.database import get_mongo_db
import pandas as pd
from bson import ObjectId

def extract_and_transform_product_timeseries(tenant_id):
    print(f"[LOG][ETL] Iniciando extract_and_transform_product_timeseries para tenant_id={tenant_id}")
    db = get_mongo_db()
    try:
        tenant_oid = ObjectId(tenant_id)
    except Exception as e:
        print(f"[LOG][ETL] tenant_id inválido: {tenant_id}, error: {e}")
        return pd.DataFrame(columns=["producto_id", "ds", "y"])

    ventadetalles = list(db.ventadetalles.find({"tenant": tenant_oid}))
    print(f"[LOG][ETL] ventadetalles encontrados: {len(ventadetalles)}")
    if not ventadetalles:
        print("[LOG][ETL] No hay ventadetalles para el tenant.")
        return pd.DataFrame(columns=["producto_id", "ds", "y"])

    vd_df = pd.DataFrame(ventadetalles)
    print(f"[LOG][ETL] DataFrame ventadetalles shape: {vd_df.shape}")
    if vd_df.empty or "producto_variedad" not in vd_df.columns:
        print("[LOG][ETL] DataFrame ventadetalles vacío o sin columna producto_variedad")
        return pd.DataFrame(columns=["producto_id", "ds", "y"])

    # Calcula el monto por línea
    if "precio_unitario" in vd_df.columns:
        vd_df["monto"] = vd_df["cantidad"] * vd_df["precio_unitario"]
    elif "precio" in vd_df.columns:
        vd_df["monto"] = vd_df["cantidad"] * vd_df["precio"]
    else:
        print("[LOG][ETL] No existe columna de precio en ventadetalles!")
        return pd.DataFrame(columns=["producto_id", "ds", "y"])

    vd_df["producto_variedad"] = vd_df["producto_variedad"].apply(lambda x: str(x) if not isinstance(x, str) else x)
    producto_variedad_ids = vd_df["producto_variedad"].unique()
    print(f"[LOG][ETL] producto_variedad_ids únicos: {producto_variedad_ids}")

    def to_oid(x):
        try:
            return ObjectId(x)
        except Exception:
            return x

    producto_variedades = list(
        db.producto_variedads.find({"_id": {"$in": [to_oid(pv) for pv in producto_variedad_ids]}})
    )
    print(f"[LOG][ETL] producto_variedades encontrados: {len(producto_variedades)}")
    if not producto_variedades:
        print("[LOG][ETL] No hay producto_variedades para los ids encontrados.")
        return pd.DataFrame(columns=["producto_id", "ds", "y"])

    pv_df = pd.DataFrame(producto_variedades)
    print(f"[LOG][ETL] DataFrame producto_variedades shape: {pv_df.shape}")
    if pv_df.empty or "producto" not in pv_df.columns:
        print("[LOG][ETL] DataFrame producto_variedades vacío o sin columna producto")
        return pd.DataFrame(columns=["producto_id", "ds", "y"])

    pv_df["_id"] = pv_df["_id"].apply(lambda x: str(x) if not isinstance(x, str) else x)
    pv_df["producto"] = pv_df["producto"].apply(lambda x: str(x) if not isinstance(x, str) else x)

    merged = vd_df.merge(pv_df[["_id", "producto"]], left_on="producto_variedad", right_on="_id", how="left")
    print(f"[LOG][ETL] merged shape: {merged.shape}")

    merged["createdAT"] = pd.to_datetime(merged["createdAT"], errors="coerce")
    merged = merged.dropna(subset=["producto", "createdAT", "monto"])
    print(f"[LOG][ETL] merged shape tras dropna: {merged.shape}")

    # Agrupa por producto y día, suma el monto
    grouped = (
        merged.groupby(["producto", merged["createdAT"].dt.date])
        .agg({"monto": "sum"})
        .reset_index()
        .rename(columns={"producto": "producto_id", "createdAT": "ds", "monto": "y"})
    )
    print(f"[LOG][ETL] grouped shape: {grouped.shape}")
    print(f"[LOG][ETL] grouped columnas: {grouped.columns}")
    if not grouped.empty:
        print(f"[LOG][ETL] Ejemplo grouped: {grouped.head(3)}")
    else:
        print("[LOG][ETL] grouped DataFrame VACÍO.")
    return grouped[["producto_id", "ds", "y"]]