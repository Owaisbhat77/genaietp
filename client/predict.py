import json
import requests

url = "http://localhost:8501/v1/models/sms_model:predict"
payload = {
    "instances": [
        "Congratulations! You've won a prize.",
        "Let's meet at 5pm."
    ]
}
response = requests.post(url, json=payload)
print(response.json())
