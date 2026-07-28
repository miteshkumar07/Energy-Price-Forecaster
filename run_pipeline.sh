#!/bin/bash

echo "🚀 Starting Full Energy Optimization Pipeline..."

echo "📥 1. Running ETL..."
python src/etl.py

echo "🧠 2. Retraining Models..."
python src/train.py

echo "⚡ 3. Generating Forecasts & SHAP..."
python src/inference.py

echo "🌐 4. Launching Dashboard..."
streamlit run streamlit_app.py