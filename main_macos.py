import cv2
import numpy as np
import time
import os
import sys
import torch
import pandas as pd
from PIL import Image
import warnings
import websocket
import json
from datetime import datetime
from collections import deque
import threading
import queue

# =============================================
# VISUALIZADOR ESTABLE PARA macOS
# =============================================
class MacOSVisualizer:
    """Clase especializada para visualización estable en macOS"""
    def __init__(self, window_name="Detector de Amenazas", scale=1.0):
        self.window_name = window_name
        self.scale = scale
        self.window_created = False
        self.last_frame = None
        self.first_try = True
    
    def initialize(self):
        """Inicializa la visualización de manera segura"""
        try:
            # Crear ventana de manera segura sin mostrarla aún
            if not self.window_created:
                # En macOS, usar WINDOW_AUTOSIZE es más estable
                cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
                # Mover la ventana a una posición segura
                cv2.moveWindow(self.window_name, 100, 100)
                self.window_created = True
                print("✓ Visualizador inicializado correctamente")
                return True
        except Exception as e:
            print(f"⚠️ Advertencia: No se pudo inicializar visualizador: {e}")
            return False
    
    def imshow(self, frame):
        """Muestra un frame de manera segura"""
        if frame is None or frame.size == 0:
            return
        
        self.last_frame = frame.copy()
        
        # Si es el primer intento, inicializar con más cuidado
        if self.first_try:
            self.initialize()
            self.first_try = False
        
        try:
            # Escalar imagen si es necesario
            if self.scale != 1.0:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (int(w * self.scale), int(h * self.scale)))
            
            # Mostrar imagen con manejo de errores
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)  # Crucial para actualizar la ventana
            return True
        except Exception as e:
            if self.window_created:
                print(f"⚠️ Error al mostrar imagen: {e}")
                # Intentar recrear la ventana
                try:
                    cv2.destroyWindow(self.window_name)
                    self.window_created = False
                    self.initialize()
                except:
                    pass
            return False
    
    def destroy(self):
        """Cierra la ventana de visualización"""
        if self.window_created:
            try:
                cv2.destroyWindow(self.window_name)
                self.window_created = False
            except:
                pass

# =============================================
# CONFIGURACIÓN Y PARÁMETROS
# =============================================
CONFIG = {
    # Ajusta esta dirección IP para que coincida con tu servidor Node.js
    "WS_URL": "ws://192.168.0.224:8080",
    "CAMERA_INDEX": 0,
    "CAMERA_WIDTH": 640,
    "CAMERA_HEIGHT": 480,
    "CAMERA_FPS": 15,
    "DISPLAY_SCALE": 1.0,         # Escala para visualización (menor = más velocidad)
    "PROCESS_EVERY_N_FRAMES": 3,   # Procesar 1 de cada N frames (mayor = más velocidad)
    "CONFIDENCE_THRESHOLD": 0.35,  # Umbral de confianza para detecciones
    "ALERTS_THRESHOLD": 0.4,       # Umbral para generar alertas
    "ALERTS_CONFIRMATIONS": 2,     # Confirmaciones necesarias para alertar
    "ALERTS_WINDOW": 3,            # Ventana de tiempo (segundos) para confirmar alertas
    "MODEL_SIZE": "small"          # "nano", "small", "medium" (menor = más velocidad)
}

# Mapa de tamaños de modelo
MODEL_MAP = {
    "nano": "yolov5n",
    "small": "yolov5s",
    "medium": "yolov5m"
}

# =============================================
# CONFIGURACIÓN INICIAL PARA macOS
# =============================================
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
    warnings.filterwarnings("ignore", message="AVCaptureDeviceTypeExternal is deprecated")

# =============================================
# CONFIGURACIÓN DE ADVERTENCIAS
# =============================================
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore::Warning"
os.environ["TORCH_LOGS"] = "WARNING"
torch.set_warn_always(False)

# =============================================
# CLASE PARA GESTIÓN DE ALERTAS
# =============================================
class GestorAlertas:
    def __init__(self, umbral_confianza=0.4, confirmaciones_necesarias=2, tiempo_ventana=3):
        self.umbral_confianza = umbral_confianza
        self.confirmaciones_necesarias = confirmaciones_necesarias
        self.tiempo_ventana = tiempo_ventana
        
        # Cola para almacenar las últimas detecciones
        self.detecciones_recientes = deque(maxlen=20)
        
        # Diccionario para rastrear alertas confirmadas
        self.alertas_confirmadas = {}
        
        # Registro de alertas enviadas
        self.alertas_enviadas = {}
        
        # Lock para operaciones thread-safe
        self.lock = threading.Lock()
        
        # Estado de conexión
        self.ws = None
        self.ws_conectado = False
        self.ultimo_intento_conexion = 0
        
    def conectar_websocket(self, url=CONFIG["WS_URL"]):
        """Intenta conectar al servidor WebSocket"""
        tiempo_actual = time.time()
        
        # Evitar intentos frecuentes de reconexión
        if tiempo_actual - self.ultimo_intento_conexion < 5:
            return self.ws
            
        self.ultimo_intento_conexion = tiempo_actual
        
        try:
            if self.ws is not None:
                try:
                    self.ws.close()
                except:
                    pass
                    
            print(f"Intentando conectar a: {url}")
            self.ws = websocket.create_connection(url, timeout=5)
            self.ws_conectado = True
            print("✅ Conectado al servidor WebSocket")
            
            # Identificarse como detector Python
            self.identificarse()
            
            return self.ws
        except Exception as e:
            self.ws_conectado = False
            print(f"❌ Error al conectar con el WebSocket: {str(e)}")
            return None
    
    def identificarse(self):
        """Envía mensaje de identificación al servidor"""
        if not self.ws_conectado or not self.ws:
            return False
            
        try:
            datos = {
                "tipo": "identificacion",
                "cliente": "python",
                "sistema": sys.platform,
                "version": "1.0.0"
            }
            self.ws.send(json.dumps(datos))
            return True
        except Exception as e:
            print(f"Error al identificarse: {str(e)}")
            return False
    
    def registrar_deteccion(self, tipo, confianza, ubicacion=None):
        """Registra una nueva detección y determina si debe generar una alerta"""
        if confianza < self.umbral_confianza:
            return False
            
        tiempo_actual = time.time()
        
        with self.lock:
            # Registrar la nueva detección
            nueva_deteccion = {
                'tipo': tipo,
                'confianza': confianza,
                'ubicacion': ubicacion,
                'timestamp': tiempo_actual
            }
            self.detecciones_recientes.append(nueva_deteccion)
            
            # Generar una clave única para esta detección
            clave = f"{tipo}_{ubicacion if ubicacion else 'desconocido'}"
            
            # Contar detecciones similares recientes
            detecciones_similares = [
                d for d in self.detecciones_recientes
                if d['tipo'] == tipo and 
                d['ubicacion'] == ubicacion and
                tiempo_actual - d['timestamp'] <= self.tiempo_ventana
            ]
            
            # Actualizar el contador de confirmaciones
            if clave not in self.alertas_confirmadas:
                self.alertas_confirmadas[clave] = {
                    'contador': 1,
                    'ultima_deteccion': tiempo_actual,
                    'confianza_promedio': confianza
                }
            else:
                info = self.alertas_confirmadas[clave]
                info['contador'] += 1
                info['ultima_deteccion'] = tiempo_actual
                
                # Actualizar confianza promedio
                total = (info['contador'] - 1) * info['confianza_promedio'] + confianza
                info['confianza_promedio'] = total / info['contador']
            
            # Verificar si se alcanzó el umbral de confirmaciones
            if len(detecciones_similares) >= self.confirmaciones_necesarias:
                # Verificar si ya se envió una alerta similar recientemente
                if clave in self.alertas_enviadas:
                    ultima_alerta = self.alertas_enviadas[clave]
                    # Evitar spam de alertas - solo enviar si pasó suficiente tiempo
                    if tiempo_actual - ultima_alerta < self.tiempo_ventana * 1.5:
                        return False
                
                # Registrar que se envió una alerta
                self.alertas_enviadas[clave] = tiempo_actual
                
                # Calcular confianza promedio
                confianza_promedio = sum(d['confianza'] for d in detecciones_similares) / len(detecciones_similares)
                
                return {
                    'tipo': tipo,
                    'confianza': confianza_promedio,
                    'ubicacion': ubicacion,
                    'conteo': len(detecciones_similares)
                }
                
        return False
        
    def enviar_alerta(self, datos_alerta):
        """Envía una alerta al servidor WebSocket"""
        if not self.ws_conectado:
            self.ws = self.conectar_websocket()
            if not self.ws:
                return False
        
        try:
            datos = {
                "fecha": datetime.now().strftime("%Y-%m-%d"),
                "hora": datetime.now().strftime("%H:%M:%S"),
                "ubicacion": datos_alerta.get('ubicacion', "Ubicación sin determinar"),
                "confianza": float(datos_alerta['confianza']),
                "tipo": datos_alerta['tipo'],
                "conteo": datos_alerta.get('conteo', 1)
            }
            
            self.ws.send(json.dumps(datos))
            return True
        except Exception as e:
            print(f"⚠️ Error al enviar alerta: {str(e)}")
            self.ws_conectado = False
            return False
    
    def limpiar_datos_antiguos(self):
        """Elimina datos antiguos para evitar consumo excesivo de memoria"""
        tiempo_actual = time.time()
        with self.lock:
            # Limpiar alertas confirmadas antiguas
            claves_para_eliminar = []
            for clave, info in self.alertas_confirmadas.items():
                if tiempo_actual - info['ultima_deteccion'] > self.tiempo_ventana * 2:
                    claves_para_eliminar.append(clave)
            
            for clave in claves_para_eliminar:
                del self.alertas_confirmadas[clave]
                
            # Limpiar registro de alertas enviadas antiguas
            claves_para_eliminar = []
            for clave, timestamp in self.alertas_enviadas.items():
                if tiempo_actual - timestamp > self.tiempo_ventana * 3:
                    claves_para_eliminar.append(clave)
                    
            for clave in claves_para_eliminar:
                del self.alertas_enviadas[clave]

# =============================================
# CLASE PARA RASTREO DE MOVIMIENTO
# =============================================
class RastreadorMovimiento:
    def __init__(self, max_puntos=10):
        self.trayectorias = {}
        self.max_puntos = max_puntos
        self.colores = {}
        self.lock = threading.Lock()
        
    def actualizar(self, personas):
        with self.lock:
            if not self.trayectorias:
                for i, persona in enumerate(personas):
                    centro = ((persona[0] + persona[2]) // 2, (persona[1] + persona[3]) // 2)
                    self.trayectorias[i] = [centro]
                    self.colores[i] = tuple(np.random.randint(0, 255, 3).tolist())
                return
            
            if personas:
                for persona in personas:
                    centro_actual = ((persona[0] + persona[2]) // 2, (persona[1] + persona[3]) // 2)
                    
                    min_dist = float('inf')
                    min_id = None
                    
                    for id_trayectoria, puntos in self.trayectorias.items():
                        if puntos:
                            ultimo_punto = puntos[-1]
                            dist = np.sqrt((centro_actual[0] - ultimo_punto[0])**2 + 
                                        (centro_actual[1] - ultimo_punto[1])**2)
                            
                            if dist < min_dist:
                                min_dist = dist
                                min_id = id_trayectoria
                    
                    if min_id is not None and min_dist < 100:
                        self.trayectorias[min_id].append(centro_actual)
                        if len(self.trayectorias[min_id]) > self.max_puntos:
                            self.trayectorias[min_id].pop(0)
                    else:
                        nuevo_id = max(self.trayectorias.keys()) + 1 if self.trayectorias else 0
                        self.trayectorias[nuevo_id] = [centro_actual]
                        self.colores[nuevo_id] = tuple(np.random.randint(0, 255, 3).tolist())
    
    def dibujar(self, frame):
        with self.lock:
            for id_trayectoria, puntos in self.trayectorias.items():
                if len(puntos) < 2:
                    continue
                    
                color = self.colores[id_trayectoria]
                for i in range(1, len(puntos)):
                    grosor = int(np.sqrt(self.max_puntos / float(i + 1)) * 2.5)
                    cv2.line(frame, puntos[i-1], puntos[i], color, grosor)
                
                for i, punto in enumerate(puntos):
                    grosor = int(np.sqrt(self.max_puntos / float(i + 1)) * 2.5)
                    cv2.circle(frame, punto, grosor, color, -1)
                    
            if self.trayectorias:
                malla = np.zeros_like(frame)
                ids = list(self.trayectorias.keys())
                for i in range(len(ids)):
                    for j in range(i+1, len(ids)):
                        if (self.trayectorias[ids[i]] and self.trayectorias[ids[j]] and 
                            len(self.trayectorias[ids[i]]) > 0 and len(self.trayectorias[ids[j]]) > 0):
                            punto1 = self.trayectorias[ids[i]][-1]
                            punto2 = self.trayectorias[ids[j]][-1]
                            
                            dist = np.sqrt((punto1[0] - punto2[0])**2 + (punto1[1] - punto2[1])**2)
                            
                            if dist < 200:
                                intensidad = int(255 * (1 - dist/200))
                                cv2.line(malla, punto1, punto2, (0, intensidad, intensidad), 1)
                
                return cv2.addWeighted(frame, 1.0, malla, 0.4, 0)
            
            return frame

# =============================================
# FUNCIONES AUXILIARES
# =============================================
def obtener_emoji_confianza(confianza):
    """Devuelve un emoji de color según el nivel de confianza"""
    if confianza >= 0.7:
        return "🔴"  # Rojo - Alta confianza
    elif confianza >= 0.5:
        return "🟠"  # Naranja - Confianza media-alta
    elif confianza >= 0.4:
        return "🟡"  # Amarillo - Confianza media
    else:
        return "🟢"  # Verde - Confianza baja

# =============================================
# CLASE PARA PROCESAMIENTO EN SEGUNDO PLANO
# =============================================
class DetectorDeAmenazas:
    def __init__(self):
        self.modelo = None
        self.detector_thread = None
        self.visualizador_thread = None
        self.gestor_alertas = GestorAlertas(
            umbral_confianza=CONFIG["ALERTS_THRESHOLD"],
            confirmaciones_necesarias=CONFIG["ALERTS_CONFIRMATIONS"],
            tiempo_ventana=CONFIG["ALERTS_WINDOW"]
        )
        self.rastreador = RastreadorMovimiento(max_puntos=15)
        
        # Colas para comunicación entre hilos
        self.queue_frames = queue.Queue(maxsize=5)  # Limitar tamaño para evitar fugas de memoria
        self.queue_resultados = queue.Queue(maxsize=5)
        
        # Visualizador especializado para macOS
        self.visualizador = MacOSVisualizer(scale=CONFIG.get("DISPLAY_SCALE", 1.0))
        
        self.running = False
        self.camara = None
        self.frame_count = 0
        self.ultima_limpieza = time.time()
        self.ultima_deteccion = time.time()
        self.detecciones_armas = 0
        self.arma_detectada_antes = False
        self.fps_count = 0
        self.fps_time = time.time()
        self.fps_value = 0
        self.ultimo_frame_procesado = None
        
    def inicializar(self):
        """Inicializa el detector y la cámara"""
        print("\n" + "=" * 60)
        print("SISTEMA DE DETECCIÓN DE AMENAZAS CON ARMAS")
        print("=" * 60)
        
        # Inicializar gestor de alertas y WebSocket
        print("\n[1/4] Inicializando conexión WebSocket...")
        self.gestor_alertas.ws = self.gestor_alertas.conectar_websocket()
        
        # Inicializar cámara
        print("\n[2/4] Conectando con cámara...")
        self.inicializar_camara()
        
        # Inicializar modelo
        print("\n[3/4] Cargando modelo de detección...")
        self.cargar_modelo()
        
        # Iniciar hilos de procesamiento
        print("\n[4/4] Iniciando sistema de procesamiento...")
        self.iniciar_hilos()
        
        print("\nSISTEMA ACTIVO - Monitorizando amenazas")
        print("Presiona 'q' para salir del sistema")
        print("-" * 60 + "\n")
        
    def inicializar_camara(self):
        """Inicializa y configura la cámara"""
        try:
            # En macOS, utilizar explícitamente CAP_AVFOUNDATION
            if sys.platform == "darwin":
                self.camara = cv2.VideoCapture(CONFIG["CAMERA_INDEX"], cv2.CAP_AVFOUNDATION)
            else:
                self.camara = cv2.VideoCapture(CONFIG["CAMERA_INDEX"])
            
            # Configurar parámetros de la cámara
            self.camara.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["CAMERA_WIDTH"])
            self.camara.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["CAMERA_HEIGHT"])
            self.camara.set(cv2.CAP_PROP_FPS, CONFIG["CAMERA_FPS"])
            
            # Verificar que la cámara se haya abierto correctamente
            if not self.camara.isOpened():
                print("ERROR: No se pudo abrir la cámara. Intentando método alternativo...")
                # Intentar método alternativo para algunos sistemas
                self.camara = cv2.VideoCapture(CONFIG["CAMERA_INDEX"])
                if not self.camara.isOpened():
                    raise Exception("No se pudo acceder a la cámara después de múltiples intentos")
            
            # Leer dimensiones reales (pueden ser diferentes a las solicitadas)
            ancho = self.camara.get(cv2.CAP_PROP_FRAME_WIDTH)
            alto = self.camara.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.camara.get(cv2.CAP_PROP_FPS)
            
            print(f"Cámara conectada. Resolución: {int(ancho)}x{int(alto)}, FPS: {fps:.1f}")
        except Exception as e:
            print(f"ERROR al inicializar cámara: {str(e)}")
            sys.exit(1)
            
    def cargar_modelo(self):
        """Carga el modelo YOLOv5"""
        try:
            # Obtener tamaño de modelo según configuración
            modelo_nombre = MODEL_MAP.get(CONFIG["MODEL_SIZE"], "yolov5s")
            
            print(f"Cargando modelo {modelo_nombre} desde torch hub...")
            print("(Este proceso puede tardar hasta 1 minuto en la primera ejecución)")
            
            # Cargar modelo de torch hub
            modelo = torch.hub.load('ultralytics/yolov5', modelo_nombre, trust_repo=True)
            
            # Forzar uso de CPU para mayor compatibilidad
            modelo.to('cpu')
            
            # Configurar parámetros del modelo
            modelo.conf = CONFIG["CONFIDENCE_THRESHOLD"]
            modelo.classes = [0, 67]  # 0=Personas, 67=Pistolas
            
            print(f"✓ Modelo {modelo_nombre} cargado correctamente")
            print(f"Umbrales: Detección={modelo.conf}, Alertas={CONFIG['ALERTS_THRESHOLD']}")
            
            self.modelo = modelo
            return True
        except Exception as e:
            print(f"ERROR al cargar modelo: {str(e)}")
            sys.exit(1)
            
    def iniciar_hilos(self):
        """Inicia los hilos de procesamiento y visualización"""
        self.running = True
        
        # Hilo de procesamiento
        self.detector_thread = threading.Thread(target=self.procesar_frames, daemon=True)
        self.detector_thread.start()
        
        # Hilo de visualización
        self.visualizador_thread = threading.Thread(target=self.visualizar_resultados, daemon=True)
        self.visualizador_thread.start()
        
    def detener(self):
        """Detiene el detector y libera recursos"""
        self.running = False
        
        # Esperar a que terminen los hilos
        if self.detector_thread and self.detector_thread.is_alive():
            self.detector_thread.join(timeout=1.0)
        
        if self.visualizador_thread and self.visualizador_thread.is_alive():
            self.visualizador_thread.join(timeout=1.0)
            
        # Cerrar visualizador
        self.visualizador.destroy()
            
        # Cerrar conexiones y liberar recursos
        if self.gestor_alertas.ws:
            self.gestor_alertas.ws.close()
            
        if self.camara:
            self.camara.release()
            
        cv2.destroyAllWindows()
        print("\nSistema detenido correctamente.")
        
    def ejecutar(self):
        """Método principal que ejecuta el bucle de captura de frames"""
        try:
            self.inicializar()
            
            # Bucle principal - solo captura frames y los envía a la cola
            while self.running:
                # Capturar frame
                ret, frame = self.camara.read()
                
                if not ret or frame is None:
                    print("Error: No se pudo obtener frame de la cámara")
                    # Intentar reconectar
                    time.sleep(0.5)
                    continue
                
                # Incrementar contador de frames
                self.frame_count += 1
                self.fps_count += 1
                
                # Calcular FPS de captura
                tiempo_actual = time.time()
                if tiempo_actual - self.fps_time >= 1.0:
                    self.fps_value = self.fps_count / (tiempo_actual - self.fps_time)
                    self.fps_count = 0
                    self.fps_time = tiempo_actual
                
                # Enviar a la cola solo 1 de cada N frames para procesamiento
                if self.frame_count % CONFIG["PROCESS_EVERY_N_FRAMES"] == 0:
                    try:
                        # Si la cola está llena, descartar el frame más antiguo
                        if self.queue_frames.full():
                            try:
                                self.queue_frames.get_nowait()
                            except:
                                pass
                        
                        # Poner el frame actual en la cola
                        self.queue_frames.put(frame.copy(), block=False)
                    except:
                        pass  # Ignorar si no se puede añadir a la cola
                
                # Comprobar si el usuario quiere salir
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
            # Detener todo al final
            self.detener()
            
        except KeyboardInterrupt:
            print("\nPrograma interrumpido por el usuario")
            self.detener()
        except Exception as e:
            print(f"Error inesperado: {str(e)}")
            self.detener()
            
    def procesar_frames(self):
        """Hilo de procesamiento: analiza frames de la cola"""
        while self.running:
            try:
                # Obtener frame de la cola
                frame = self.queue_frames.get(timeout=1.0)
                
                if frame is None:
                    continue
                
                # Procesar frame
                try:
                    frame_procesado, detecciones = self.detectar_amenazas(frame)
                    
                    # Si la cola está llena, descartar el resultado más antiguo
                    if self.queue_resultados.full():
                        try:
                            self.queue_resultados.get_nowait()
                        except:
                            pass
                    
                    # Enviar resultado a la cola de visualización
                    self.queue_resultados.put((frame_procesado, detecciones), block=False)
                    
                except Exception as e:
                    print(f"Error al procesar frame: {str(e)}")
                    
            except queue.Empty:
                # No hay frames en la cola, esperar
                time.sleep(0.01)
            except Exception as e:
                print(f"Error en hilo de procesamiento: {str(e)}")
                time.sleep(0.1)
                
    def visualizar_resultados(self):
        """Hilo de visualización con manejo seguro para macOS"""
        ultimo_reporte = time.time()
        frames_procesados = 0
        
        # Inicializar visualizador
        self.visualizador.initialize()
        
        while self.running:
            try:
                # Obtener resultado de la cola
                try:
                    resultado = self.queue_resultados.get(timeout=0.5)
                except queue.Empty:
                    # Mostrar último frame si existe
                    if hasattr(self, 'ultimo_frame_procesado') and self.ultimo_frame_procesado is not None:
                        self.visualizador.imshow(self.ultimo_frame_procesado)
                    time.sleep(0.01)
                    continue
                    
                if resultado is None:
                    continue
                    
                frame_procesado, detecciones = resultado
                frames_procesados += 1
                
                # Guardar último frame para referencia
                self.ultimo_frame_procesado = frame_procesado.copy()
                
                # Preparar información para mostrar en pantalla
                tiempo_actual = time.time()
                
                # Añadir información al frame
                cv2.putText(frame_procesado, f"FPS: {self.fps_value:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Estado general del sistema
                armas_actuales = len([d for d in detecciones if d["clase"] == 67])
                estado = "NORMAL" if armas_actuales == 0 else "¡ALERTA!"
                color_estado = (0, 255, 0) if armas_actuales == 0 else (0, 0, 255)
                
                cv2.putText(frame_procesado, f"Estado: {estado}", 
                           (10, frame_procesado.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)
                
                # Estado de conexión WebSocket
                cv2.putText(frame_procesado, f"WebSocket: {'✓' if self.gestor_alertas.ws_conectado else '✗'}", 
                           (frame_procesado.shape[1] - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 255, 0) if self.gestor_alertas.ws_conectado else (0, 0, 255), 2)
                
                # Mostrar frame usando el visualizador estable
                self.visualizador.imshow(frame_procesado)
                
                # Verificar si es hora de generar un reporte
                if tiempo_actual - ultimo_reporte > 30:
                    tiempo_operacion = tiempo_actual - ultimo_reporte
                    print("\n" + "-" * 30)
                    print(f"Reporte de estado - Últimos {int(tiempo_operacion)} segundos:")
                    print(f"- Frames procesados: {frames_procesados}")
                    print(f"- Tasa media: {frames_procesados/tiempo_operacion:.1f} FPS")
                    print(f"- Nuevas detecciones de armas: {self.detecciones_armas}")
                    print(f"- Estado de WebSocket: {'Conectado' if self.gestor_alertas.ws_conectado else 'Desconectado'}")
                    print("-" * 30 + "\n")
                    
                    frames_procesados = 0
                    self.detecciones_armas = 0
                    ultimo_reporte = tiempo_actual
                    
                    # Intentar reconectar WebSocket si es necesario
                    if not self.gestor_alertas.ws_conectado:
                        self.gestor_alertas.ws = self.gestor_alertas.conectar_websocket()
                        
                    # Limpiar datos antiguos
                    self.gestor_alertas.limpiar_datos_antiguos()
                    
            except Exception as e:
                print(f"Error en hilo de visualización: {str(e)}")
                time.sleep(0.1)
        
        # Cerrar visualizador al terminar
        self.visualizador.destroy()
                
    def detectar_amenazas(self, frame):
        """Procesa un frame para detectar personas y posibles amenazas"""
        try:
            # Ejecutar inferencia
            resultados = self.modelo(frame)
            detecciones = resultados.pandas().xyxy[0]
            
            personas = []
            armas = []
            detecciones_lista = []
            
            # Procesar detecciones
            for idx, deteccion in detecciones.iterrows():
                x1, y1, x2, y2 = (
                    int(deteccion["xmin"]),
                    int(deteccion["ymin"]),
                    int(deteccion["xmax"]),
                    int(deteccion["ymax"]),
                )
                confianza = deteccion["confidence"]
                clase = int(deteccion["class"])
                
                # Guardar detección en lista
                detecciones_lista.append({
                    "clase": clase,
                    "confianza": confianza,
                    "bbox": [x1, y1, x2, y2]
                })
                
                if clase == 0:  # Persona
                    personas.append([x1, y1, x2, y2, confianza])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    emoji = obtener_emoji_confianza(confianza)
                    cv2.putText(
                        frame,
                        f"Persona {emoji}: {confianza:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    centro = ((x1 + x2) // 2, (y1 + y2) // 2)
                    cv2.circle(frame, centro, 5, (0, 255, 0), -1)
                
                elif clase == 67:  # Arma
                    armas.append([x1, y1, x2, y2, confianza])
                    emoji = obtener_emoji_confianza(confianza)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        frame,
                        f"¡ARMA! {emoji}: {confianza:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    
                    # Registrar alerta y enviar si cumple criterios
                    ubicacion_arma = f"x:{(x1+x2)//2},y:{(y1+y2)//2}"
                    resultado_alerta = self.gestor_alertas.registrar_deteccion("arma", confianza, ubicacion_arma)
                    
                    if resultado_alerta:
                        if self.gestor_alertas.enviar_alerta(resultado_alerta):
                            emoji = obtener_emoji_confianza(resultado_alerta['confianza'])
                            print(f"\n{emoji} ALERTA CONFIRMADA: {resultado_alerta['conteo']} detecciones de arma")
                            print(f"Confianza promedio: {resultado_alerta['confianza']:.2f}")
                            self.detecciones_armas += 1
                            self.ultima_deteccion = time.time()
                            self.arma_detectada_antes = True
                            
                            # Mostrar alerta visual
                            cv2.putText(
                                frame,
                                "¡ALERTA: ARMA DETECTADA!",
                                (frame.shape[1] // 2 - 150, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2,
                            )
            
            # Marcar fin de detección si antes había un arma
            if self.arma_detectada_antes and len(armas) == 0 and time.time() - self.ultima_deteccion > 5:
                print("\n🟢 INFO: Ya no se detectan armas en la escena")
                self.arma_detectada_antes = False
            
            # Actualizar trayectorias
            self.rastreador.actualizar(personas)
            frame = self.rastreador.dibujar(frame)
            
            # Analizar posibles amenazas entre personas y armas
            if len(armas) > 0 and len(personas) > 0:
                self.analizar_amenazas_persona_arma(frame, personas, armas)
            
            return frame, detecciones_lista
            
        except Exception as e:
            print(f"Error al procesar frame: {str(e)}")
            return frame, []
    
    def analizar_amenazas_persona_arma(self, frame, personas, armas):
        """Analiza posibles amenazas entre personas y armas"""
        for arma in armas:
            arma_centro_x = (arma[0] + arma[2]) // 2
            arma_centro_y = (arma[1] + arma[3]) // 2
            arma_centro = (arma_centro_x, arma_centro_y)
            
            persona_mas_cercana = None
            min_distancia = float('inf')
            
            # Encontrar persona más cercana al arma
            for i, persona in enumerate(personas):
                persona_centro_x = (persona[0] + persona[2]) // 2
                persona_centro_y = (persona[1] + persona[3]) // 2
                persona_centro = (persona_centro_x, persona_centro_y)
                
                distancia = np.sqrt((arma_centro_x - persona_centro_x)**2 + 
                                  (arma_centro_y - persona_centro_y)**2)
                
                if distancia < min_distancia:
                    min_distancia = distancia
                    persona_mas_cercana = (i, persona_centro)
            
            # Si hay una persona cercana al arma
            if persona_mas_cercana is not None and min_distancia < 100:
                idx_persona, centro_persona = persona_mas_cercana
                
                # Dibujar línea entre arma y persona
                cv2.line(frame, centro_persona, arma_centro, (255, 0, 0), 2)
                
                # Calcular dirección (vector)
                dx = arma_centro_x - centro_persona[0]
                dy = arma_centro_y - centro_persona[1]
                
                # Normalizar vector
                longitud = np.sqrt(dx*dx + dy*dy)
                if longitud > 0:
                    dx /= longitud
                    dy /= longitud
                
                # Extender línea de proyección
                longitud_proyeccion = 300
                punto_final = (int(arma_centro_x + dx * longitud_proyeccion), 
                             int(arma_centro_y + dy * longitud_proyeccion))
                
                cv2.line(frame, arma_centro, punto_final, (255, 255, 0), 1)
                
                # Analizar si el arma apunta a otra persona
                for j, otra_persona in enumerate(personas):
                    if j == idx_persona:
                        continue
                    
                    otra_centro_x = (otra_persona[0] + otra_persona[2]) // 2
                    otra_centro_y = (otra_persona[1] + otra_persona[3]) // 2
                    otra_centro = (otra_centro_x, otra_centro_y)
                    
                    # Vector desde arma a otra persona
                    dx_otro = otra_centro_x - arma_centro_x
                    dy_otro = otra_centro_y - arma_centro_y
                    
                    # Normalizar
                    longitud_otro = np.sqrt(dx_otro*dx_otro + dy_otro*dy_otro)
                    if longitud_otro > 0:
                        dx_otro /= longitud_otro
                        dy_otro /= longitud_otro
                    
                    # Producto escalar (coseno del ángulo)
                    producto_escalar = dx*dx_otro + dy*dy_otro
                    
                    # Si el arma está alineada con otra persona (coseno > 0.65)
                    if producto_escalar > 0.65:
                        mensaje = "¡ALERTA! Arma apuntando a una persona"
                        emoji = obtener_emoji_confianza(arma[4])
                        
                        print("\n" + "=" * 50)
                        print(f"{emoji} ¡AMENAZA DETECTADA! Arma apuntando a una persona")
                        print(f"Confianza de detección del arma: {arma[4]:.2f}")
                        print(f"Ángulo de alineación: {producto_escalar:.2f} (0-1, donde 1 es perfecto)")
                        print("=" * 50 + "\n")
                        
                        # Dibujar línea de amenaza
                        cv2.line(frame, arma_centro, otra_centro, (0, 0, 255), 4)
                        
                        # Mostrar mensaje de alerta
                        cv2.putText(frame, mensaje, (10, 90), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        
                        # Efecto de parpadeo del borde
                        if int(time.time() * 2) % 2 == 0:
                            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), 
                                        (0, 0, 255), 20)
                        
                        # Círculo alrededor del objetivo
                        cv2.circle(frame, otra_centro, 70, (0, 0, 255), 3)
                        
                        # Texto de distancia
                        distancia_texto = f"Dist: {int(np.sqrt((arma_centro_x - otra_centro_x)**2 + (arma_centro_y - otra_centro_y)**2))}px"
                        cv2.putText(frame, distancia_texto, 
                                  ((arma_centro_x + otra_centro_x) // 2, (arma_centro_y + otra_centro_y) // 2 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Enviar alerta especial de amenaza
                        alerta_amenaza = {
                            'tipo': 'amenaza',
                            'confianza': arma[4],
                            'ubicacion': f"x:{arma_centro_x},y:{arma_centro_y}",
                            'conteo': 1
                        }
                        self.gestor_alertas.enviar_alerta(alerta_amenaza)

# =============================================
# FUNCIÓN PRINCIPAL
# =============================================
def main():
    try:
        # Crear y ejecutar detector
        detector = DetectorDeAmenazas()
        detector.ejecutar()
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario")
    except Exception as e:
        print(f"Error inesperado: {str(e)}")

if __name__ == "__main__":
    main()