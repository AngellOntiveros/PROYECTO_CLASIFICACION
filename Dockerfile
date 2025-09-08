# 1. Usar una imagen base de Python. Se recomienda usar la versión "slim"
# para un tamaño de imagen más pequeño, ya que no incluye extras innecesarios.
FROM python:3.9-slim

# 2. Establecer el directorio de trabajo dentro del contenedor.
# Aquí es donde se copiará y ejecutará el código de tu aplicación.
WORKDIR /app

# 3. Copiar solo el archivo de requerimientos primero.
# Esto permite que Docker use el caché de la capa si los requisitos no cambian,
# acelerando futuras compilaciones.
COPY requirements.txt .

# 4. Instalar las dependencias de Python.
# El comando `--no-cache-dir` ahorra espacio en la imagen final.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Instalar las dependencias de sistema para OpenCV.
# Este es el paso clave para evitar los errores de importación de cv2 en el servidor.
# Estas librerías son necesarias para que OpenCV funcione sin una interfaz gráfica.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1

# 6. Copiar el resto del código de tu aplicación al directorio de trabajo.
# Esto incluye tu script principal y el modelo si ya lo descargaste.
COPY . .

# 7. Exponer el puerto por el cual correrá la aplicación.
# Streamlit por defecto usa el puerto 8501.
EXPOSE 8501

# 8. Comando para ejecutar la aplicación de Streamlit al iniciar el contenedor.
# Reemplaza "app.py" por el nombre de tu archivo principal si es diferente.
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]