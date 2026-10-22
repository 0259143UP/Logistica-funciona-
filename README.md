# Optimización de Matrices de Distribución Logística

## 1. Definición del problema
El análisis de costos y tiempos de tránsito en los envíos de carga (tarimas y paquetería) desde un centro de distribución hacia múltiples destinos a nivel nacional genera volúmenes masivos de datos. Este proyecto es relevante porque optimizar la selección de paqueterías reduce costos operativos y mejora los tiempos de entrega.
**Pregunta concreta:** ¿Cuáles son las rutas y paqueterías más eficientes en términos de costo-tiempo basándonos en el código postal de destino?

## 2. Usuario o interesado
**Interesados:** Gerentes de Operaciones y Coordinadores de Logística.
**Decisión a tomar:** Seleccionar de manera automatizada al proveedor de transporte más rentable para cada ruta específica.

## 3. Fuentes de datos
1. **Histórico de Envíos (Mock Data / Local):** Archivo CSV generado con registros de guías de envío, pesos, paquetería, costo en MXN y código postal destino. Periodo: 2026. Formato: CSV.
2. **Zippopotam.us API (Automatizada):** API pública y gratuita que recibe un código postal de México y devuelve coordenadas geográficas y el Estado de destino. Formato: JSON.

## 4. Instrucciones de ejecución
1. Clonar este repositorio.
2. Instalar dependencias: `pip install -r requirements.txt`
3. Ejecutar el notebook ubicado en `notebooks/entregable_1.ipynb`.
