from locust import HttpUser, task, between, events
import random

# For stress testing, we'll use a valid token
# You would get this from a real login first
VALID_TOKEN = "YOUR_TOKEN_HERE" 

class VN30APIUser(HttpUser):
    wait_time = between(0.1, 0.5) # Fast requests to simulate 100 RPS
    
    symbols = ["FPT", "TCB", "VNM", "SSI", "HPG", "VIC", "VHM", "MSN", "MWG", "STB"]

    def on_start(self):
        """
        In a real test, we would log in here to get the token.
        For simplicity, we assume the token is provided or we use a mock.
        """
        # Note: You need to put a real token here after starting app.py and calling /token
        self.headers = {"Authorization": f"Bearer {VALID_TOKEN}"}

    @task(3)
    def test_predict(self):
        symbol = random.choice(self.symbols)
        self.client.get(f"/predict/{symbol}", headers=self.headers)

    @task(1)
    def test_health(self):
        self.client.get("/docs") # Swagger UI check
