# Usar una imagen oficial de Python ligera
FROM python:3.10-slim

# Crear directorio de trabajo en el contenedor
WORKDIR /app

# Copiar primero el archivo de requerimientos (aprovecha la caché de Docker)
COPY requirements.txt .

# Instalar las dependencias sin guardar caché temporal para ahorrar espacio
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente de la API (main.py)
COPY main.py .

# Exponer el puerto donde corre FastAPI
EXPOSE 8000

# Comando para ejecutar la API al encender el contenedor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]