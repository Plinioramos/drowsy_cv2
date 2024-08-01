FROM python:3.11-slim

# Instale CMake e outras dependências do sistema
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Instale as dependências do Python
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Copie o código da aplicação
COPY . /app

# Comando para rodar a aplicação Streamlit
CMD ["streamlit", "run", "DrowsinessNET9_webrtc.py"]
