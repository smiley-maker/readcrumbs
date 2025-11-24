import pytest
import requests

API_URL = "http://13.221.193.197:8000"

@pytest.fixture
def test_health_check():
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_get_random():
    response = requests.get(f"{API_URL}/random")
    assert response.status_code == 200
    assert response.json() is not None

def test_predict():
    data = {
        "items": ["The Great Gatsby", "1984", "To Kill a Mockingbird"],
        "userid": "1234567890"
    }
    response = requests.post(f"{API_URL}/predict", json=data)
    assert response.status_code == 200
    assert response.json() is not None
    assert len(response.json()["recs"]) == 10
    
if __name__ == "__main__":
    pytest.main()