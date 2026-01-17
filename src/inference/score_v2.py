import json
import numpy as np
import joblib
import os

def init():
    global model
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"),
        "g10tfm_model_xgb-realdt2.pkl"
    )
    model = joblib.load(model_path)

def run(raw_data):
    data = json.loads(raw_data)

    # Convertidor seguro para ints y floats
    def to_number(x):
        try:
            return float(x) if "." in str(x) else int(x)
        except:
            return x

    # Identificador del cliente
    customer_id = data.get("customer_id", "N/A")
    monto_solicitado = data.get("monto_solicitado", 0)

    # Variables explícitas exactamente en el orden usado para entrenar el modelo
    variables = [
        to_number(data["edad"]),
        to_number(data["ingresos_declarados"]),
        to_number(data["nivel_endeudamiento"]),
        to_number(data["utilizacion_tarjetas"]),
        to_number(data["score_buro"]),
        to_number(data["morosidad_prev"]),
        to_number(data["consumo_electrico"]),
        to_number(data["pago_serv_publico"]),
        to_number(data["titularidad_serv_publico"]),
        to_number(data["remesas"]),
        to_number(data["ingresos_bancarios"]),
        to_number(data["historial_empresa"]),
        to_number(data["portabilidad"]),
        to_number(data["ultimo_consumo_movil"])
    ]


    # Convertir a numpy
    X = np.array([variables])

    # Predicción del modelo
    prob = model.predict_proba(X)[0][1]
    score = int(round(prob * 100, 0))

    # iniciar variables booleanas
    equipo_autorizado = 0
    cargo_automatico = 0

    # Motor de decisión basado en score (USD)
    if score >= 96:
        riesgo = "Muy Bajo"
        max_autorizable = 100
        # indica si hay un equipo/ terminal autorizado 0 = no, 1 = si
        equipo_autorizado = 1
        # indica si el equipo esta condicionado a cargo automatico 0 = no, 1 = si
        cargo_automatico = 0
    
    elif score >= 86:
        riesgo = "Bajo"
        max_autorizable = 75
        equipo_autorizado = 1
        cargo_automatico = 0
    
    elif score >= 76:
        riesgo = "Medio"
        max_autorizable = 50
        equipo_autorizado = 1
        cargo_automatico = 0
    
    elif score >= 66:
        riesgo = "Alto"
        max_autorizable = 35
        equipo_autorizado = 1
        cargo_automatico = 1
    
    elif score >= 51:
        riesgo = "Muy Alto"
        max_autorizable = 33
        equipo_autorizado = 0
        cargo_automatico = 0
    
    else:
        riesgo = "Extremadamente Alto"
        max_autorizable = 25
        equipo_autorizado = 0
        cargo_automatico = 0

    # Dictamen según monto solicitado
    if monto_solicitado <= max_autorizable:
        dictamen = "APROBADO"
        razon = f"El monto solicitado USD {monto_solicitado} está dentro del límite permitido USD {max_autorizable}."
    else:
        dictamen = "RECHAZADO"
        razon = f"El monto solicitado USD {monto_solicitado} excede el máximo permitido USD {max_autorizable}."

    # Respuesta final del endpoint
    return {
        "customer_id": customer_id,
        "probabilidad_pago": float(prob),
        "score": score,
        "riesgo": riesgo,
        "monto_solicitado": monto_solicitado,
        "monto_recomendado": max_autorizable,
        "equipo_autorizado": equipo_autorizado,
        "cargo_automatico": cargo_automatico,
        "dictamen_crediticio": dictamen,
        "razon_dictamen": razon
    }
