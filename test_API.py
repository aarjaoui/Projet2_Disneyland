from fastapi import FastAPI
from fastapi.testclient import TestClient
 
app = FastAPI()

@app.get("/status")
def get_status():
    return {"Service is OK"}
    
client = TestClient(app)
 
def test_get_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"Service is OK"}
