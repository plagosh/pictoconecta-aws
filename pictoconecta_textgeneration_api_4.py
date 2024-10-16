import time
import openai
import json
import logging
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from pprint import pformat
import sys
import os
from config import ConfigModel
import traceback
from dotenv import load_dotenv

# Cargar variables de entorno desde un archivo .env
load_dotenv()

# Configuración inicial
config = ConfigModel()
openai.api_key = config.OPENAI_API_KEY
openai.api_request_timeout = 120
print("Clave API de OpenAI:", openai.api_request_timeout)

if not openai.api_key:
    raise ValueError("No se ha proporcionado la clave API de OpenAI. Asegúrate de establecer la variable de entorno OPENAI_API_KEY.")

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa la aplicación Flask
app = Flask(__name__)
CORS(
    app,
    origins="*",
    allow_headers="*",
    expose_headers="*",
)

# Variables globales
history = []
historial = []

# Función para cargar historial desde archivo JSON
def load_historial(file_path):
    logger.info(f"Iniciando carga del historial desde {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            logger.info("Historial cargado exitosamente")
            return data.get("historial", [])
    except FileNotFoundError:
        logger.warning(f"Archivo {file_path} no encontrado. Se devolverá un historial vacío.")
        return []
    except Exception as e:
        logger.error(f"Error al cargar el historial: {e}")
        return []

# Función para guardar historial en archivo JSON
def save_historial(file_path, new_data):
    logger.info(f"Iniciando guardado del historial en {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        logger.warning(f"Archivo {file_path} no encontrado. Se creará un nuevo archivo.")
        data = {'historial': []}
    except Exception as e:
        logger.error(f"Error al leer el archivo {file_path}: {e}")
        data = {'historial': []}

    data['historial'].append(new_data)

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
            logger.info("Historial guardado exitosamente")
    except Exception as e:
        logger.error(f"Error al guardar el historial: {e}")

# Función para obtener el número total de tokens utilizados
def get_total_tokens(messages):
    logger.info("Obteniendo el número total de tokens utilizados")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
            max_tokens=1,  # Set to 1 to avoid the minimum value error
            logprobs=None  # Remove the logprobs parameter
        )
        total_tokens = response['usage']['total_tokens']
        logger.info(f"Número total de tokens utilizados: {total_tokens}")
        return total_tokens
    except Exception as e:
        logger.error(f"Error al obtener el número total de tokens: {e}")
        return None

def create_app():
    global history, historial, config

    logger.info("Iniciando servidor...")

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    logger.info(pformat(config))

    if config.seed != 0:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if config.device == "cuda":
            torch.cuda.manual_seed(config.seed)

    # Carga el historial desde el archivo
    historial = load_historial(config.historial_path)
    for entry in historial:
        history.append({"role": "user", "content": entry["usuario"]})
        history.append({"role": "assistant", "content": entry["respuesta"]})
    logger.info("Historial cargado")

# Esta función es el manejador para AWS Lambda
def handler(event, context):
    logger.info("Iniciando manejador de AWS Lambda")
    
    # Verificar que 'httpMethod' esté presente
    if 'httpMethod' not in event:
        logger.error("El evento no contiene el método HTTP")
        return {
            'statusCode': 400,
            'body': json.dumps({"error": "Método HTTP no proporcionado"}),
            'headers': {'Content-Type': 'application/json'}
        }

    try:
        # Decodificar el cuerpo si es necesario (Lambda envía el cuerpo como string)
        body = event.get('body', '{}')
        if isinstance(body, str):
            body = json.loads(body)

        with app.test_request_context(
            method=event['httpMethod'],
            path=event['path'],
            json=body,
            headers=event.get('headers', {})
        ):
            response = app.full_dispatch_request()
            logger.info("Solicitud procesada exitosamente")
            return {
                'statusCode': response.status_code,
                'body': response.get_data(as_text=True),
                'headers': {'Content-Type': 'application/json'}
            }
    
    except Exception as e:
        logger.error(f"Error en el manejador de AWS Lambda: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)}),
            'headers': {'Content-Type': 'application/json'}
        }

@app.route('/chat', methods=['POST'])
def chat():
    global history, config

    logger.info("Iniciando chat")
    data = request.get_json()  # Obtiene los datos de la solicitud POST en formato JSON
    raw_text = data['text']  # Extrae el texto ingresado por el usuario
    logger.info(f"Texto ingresado por el usuario: {raw_text}")
    
    # Procesa la solicitud de conversación
    logger.info("Procesando solicitud de conversación")
    history.append({"role": "user", "content": raw_text})

    # Recortar el historial a los últimos `max_history` mensajes
    if len(history) > config.max_history * 2:
        logger.info("Recortando historial a los últimos mensajes permitidos")
        history = history[-config.max_history * 2:]

    # Asegurarse de que el historial no exceda el límite de tokens
    max_context_length = 4096 - 150  # Longitud máxima menos el espacio necesario para la respuesta

    while True:
        total_tokens = get_total_tokens(history)
        if total_tokens is None or total_tokens <= max_context_length:
            break

        if history:  # Verifica si history tiene elementos antes de hacer pop
            logger.info("Eliminando mensajes más antiguos del historial para cumplir con el límite de tokens")
            history.pop(0)  # Eliminar los mensajes más antiguos

    # Genera una sola respuesta utilizando el endpoint de chat
    logger.info("Generando respuesta")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",  # Ejemplo de modelo de chat
            messages=history,
            max_tokens=config.max_token,
            temperature=config.temperature,
            top_p=config.top_p,
            n=config.num_suggestions,
            stop=None
        ).choices[0].message['content'].strip()
        logger.info(f"Respuesta generada: {response}")
    except openai.Error as e:
        logger.error(f"Error al generar respuesta: {e}")
        response = "Lo siento, no pude entender eso."
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Error al generar respuesta: {e}\n{trace}")
        response = "Lo siento, no pude entender eso."

    save_historial(config.historial_path, {
        "usuario": raw_text,
        "respuesta": response
    })
    return jsonify({"response": response})
        
if __name__ == '__main__':
    logger.info("Iniciando la aplicación...")
    create_app()  # Asegúrate de que create_app se llama antes de ejecutar la app Flask
    app.run(host='0.0.0.0', port=8025)
