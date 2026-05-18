# genaietp – TensorFlow Text Classification + TF Serving

## 1) Setup Kaggle API
1. Create Kaggle token from https://www.kaggle.com -> Account -> API -> Create New Token  
2. Save `kaggle.json` to:
   - Linux/macOS: `~/.kaggle/kaggle.json`
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`

## 2) Install dependencies
```bash
pip install -r requirements.txt
```

## 3) Train and export model
```bash
python training/train.py
```

Model export path:
```
exported_model/1
```

## 4) Run TensorFlow Serving
```bash
cd serving
docker-compose up
```

## 5) Test inference
```bash
curl -X POST http://localhost:8501/v1/models/sms_model:predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Output will be class probabilities.

## 6) Optional Python client
```bash
python client/predict.py
```
