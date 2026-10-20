from fastapi.testclient import TestClient
import pytest
from src import api
from src.api import app

client = TestClient(app)
payload = {
    "timestamp": "2026-03-25T00:00:00", "id_equipamento": "TURB-101",
    "modo_operacao": "Carga_Media", "horas_operacao_acumuladas": 400,
    "temperatura_mancal_c": 65, "vibracao_rms_mms": 2.8,
    "pressao_oleo_bar": 4.1, "rotacao_rpm": 3000,
    "emissao_acustica_db": 52, "corrente_eletrica_amp": 28,
}

@pytest.fixture(autouse=True)
def reset_in_memory_equipment_history():
    api.history.clear()

def test_health_and_single_prediction():
    assert client.get('/health').json()['status'] == 'ok'
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    assert response.json()['tipo_falha'] in [0, 1, 2, 3]

def test_batch_and_schema_validation():
    second = {**payload, 'timestamp': '2026-03-25T00:10:00'}
    response = client.post('/predict-batch', json=[payload, second])
    assert response.status_code == 200
    assert len(response.json()['predictions']) == 2
    bad = {**payload, 'id_equipamento': 'UNKNOWN'}
    assert client.post('/predict', json=bad).status_code == 422
    assert client.post('/predict', json=payload).status_code == 422

def test_missing_sensor_is_imputed_and_batch_order_is_checked():
    partial = {**payload, 'timestamp': '2026-03-25T00:20:00', 'vibracao_rms_mms': None}
    response = client.post('/predict', json=partial)
    assert response.status_code == 200
    unordered = [
        {**payload, 'timestamp': '2026-03-25T00:40:00'},
        {**payload, 'timestamp': '2026-03-25T00:30:00'},
    ]
    assert client.post('/predict-batch', json=unordered).status_code == 422
