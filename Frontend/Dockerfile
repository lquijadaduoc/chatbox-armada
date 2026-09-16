# Usar un servidor web Nginx muy liviano (Alpine Linux)
FROM nginx:alpine

# Borrar los archivos por defecto de Nginx
RUN rm -rf /usr/share/nginx/html/*

# Copiar tu archivo index.html a la carpeta pública de Nginx
COPY . /usr/share/nginx/html/

# Exponer el puerto web estándar
EXPOSE 80

