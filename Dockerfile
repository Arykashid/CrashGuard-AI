FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install streamlit psutil requests numpy pandas plotly tensorflow keras scikit-learn joblib

EXPOSE 8502

CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]