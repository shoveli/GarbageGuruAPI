from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
import logging
from io import BytesIO
from PIL import Image

# Archivo de log por si tira algun error la api
logging.basicConfig(filename='app.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s: %(message)s')

# Inicio aplicacion
app = Flask(__name__)

# En primeras versiones, no se controlaba el tema de si el modelo se cargaba bien o no
# Por las dudas, agrego esa validacion, si no se carga el modelo, no inicia la api y escribo en el archivo log

modelo = None
try:
    modelo = load_model('app/modelo_clasificador_inception_capas_descongeladas.h5')
except Exception as e:
    logging.error(f"Error al cargar el modelo: {str(e)}")
    raise RuntimeError("No se pudo cargar el modelo. Verifique los registros de errores para más detalles.")

# Estas clases son las que tiene el modelo entrenado, para simplificar, las meti en un array aca.

class_names = ['battery', 'biological', 'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Este preprocesamiento es similar al que se hacia en el entrenamiento, es para standarizar un poco
# las imagenes que lleguen.

def preprocesar_imagen(img):
    img = img.resize((224, 224))  # Redimensionar la imagen
    img_array = image.img_to_array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión para el batch
    return img_array

# endpoint propiamente dicho.

@app.route('/clasificar', methods=['POST'])
def clasificar_imagen():
    if modelo is None:
        return jsonify({'error': 'El modelo no está disponible en este momento'}), 500

    if not request.json or 'image_base64' not in request.json:
        return jsonify({'error': 'No se envió ningún archivo en Base64'}), 400

    try:
        # Obtener y decodificar la imagen Base64
        base64_str = request.json['image_base64']

        # Corregir cualquier problema de relleno Base64
        base64_str += "=" * ((4 - len(base64_str) % 4) % 4)
        img_data = base64.b64decode(base64_str)

        # Convertir la imagen decodificada en un objeto de imagen
        img = Image.open(BytesIO(img_data))

        # Preprocesar la imagen
        imagen_preprocesada = preprocesar_imagen(img)

        # Realizar la predicción
        prediccion = modelo.predict(imagen_preprocesada)
        clase_predicha = class_names[np.argmax(prediccion)]
    except Exception as e:
        logging.error(f"Error al procesar la imagen en Base64: {str(e)}")
        return jsonify({'error': f'Error al procesar la imagen en Base64: {str(e)}'}), 500

    # Devolver la clase predicha
    return jsonify({'class': clase_predicha})

# Iniciar la aplicación
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
