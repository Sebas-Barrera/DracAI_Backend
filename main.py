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
import threading

# =============================================
# CONFIGURACIÓN INICIAL PARA macOS
# =============================================
if sys.platform == 'darwin':
    # Configuraciones específicas para macOS
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    # Ignorar warnings específicos de macOS
    warnings.filterwarnings("ignore", message="AVCaptureDeviceTypeExternal is deprecated")

# =============================================
# CONFIGURACIÓN DE ADVERTENCIAS
# =============================================
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore::Warning'
os.environ["TORCH_LOGS"] = "WARNING"
torch.set_warn_always(False)

# =============================================
# VERIFICACIÓN DE DEPENDENCIAS
# =============================================
def verificar_dependencias():
    """Verifica todas las dependencias necesarias"""
    dependencias = {
        "cv2": "opencv-python-headless",
        "numpy": "numpy",
        "torch": "torch torchvision",
        "pandas": "pandas",
        "PIL": "pillow",
        "requests": "requests",
        "websocket": "websocket-client"
    }
    
    faltantes = []
    
    for modulo, paquete in dependencias.items():
        try:
            __import__(modulo)
        except ImportError:
            faltantes.append(paquete)
    
    if faltantes:
        print("ERROR: Faltan las siguientes dependencias. Instálalas con:")
        print("pip install " + " ".join(faltantes))
        sys.exit(1)

verificar_dependencias()

# =============================================
# FUNCIONES AUXILIARES
# =============================================
# Modifica la función conectar_websocket()
def conectar_websocket():
    try:
        ws = websocket.WebSocketApp(
            "ws://192.168.1.109:8080",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Ejecutar en un hilo separado
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()
        
        return ws
    except Exception as e:
        print(f"Error al inicializar WebSocket: {str(e)}")
        return None

def on_open(ws):
    print("Conexión WebSocket establecida")
    # Identificarse como cliente Python inmediatamente
    ws.send(json.dumps({
        "tipo": "identificacion",
        "cliente": "python"
    }))

def on_message(ws, message):
    try:
        data = json.loads(message)
        # Opcional: procesar mensajes del servidor si es necesario
    except Exception as e:
        print(f"Error al procesar mensaje: {str(e)}")

def on_error(ws, error):
    print(f"Error en WebSocket: {str(error)}")

def on_close(ws, close_status, close_msg):
    print(f"Conexión cerrada. Status: {close_status}, Mensaje: {close_msg}")

def on_ping(ws, data):
    print("Ping recibido del servidor")  # El cliente responde automáticamente con pong

def on_pong(ws, data):
    print("Pong enviado al servidor")  # Confirmación de que el servidor recibió nuestro ping

def enviar_alerta_websocket(ws, confianza, tipo_alerta="arma"):
    """Envía una alerta al WebSocket cuando se detecta una amenaza"""
    if ws is None:
        return False
    
    try:
        datos_alerta = {
            "fecha": datetime.now().strftime("%Y-%m-%d"),
            "hora": datetime.now().strftime("%H:%M:%S"),
            "ubicacion": "Edificio Principal, Calle Norte #123",
            "confianza": float(confianza),
            "tipo": tipo_alerta
        }
        
        ws.send(json.dumps(datos_alerta))
        return True
    except Exception as e:
        print(f"Error al enviar alerta: {str(e)}")
        return False

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
# CLASE PARA RASTREO DE MOVIMIENTO
# =============================================
class RastreadorMovimiento:
    def __init__(self, max_puntos=10):
        self.trayectorias = {}
        self.max_puntos = max_puntos
        self.colores = {}
        
    def actualizar(self, personas):
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
# FUNCIONES DE DETECCIÓN
# =============================================
def cargar_modelo():
    """Carga el modelo YOLOv5 pre-entrenado"""
    try:
        print("Cargando YOLOv5 desde torch hub...")
        
        # Usar CPU en macOS para mayor compatibilidad
        modelo = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
        modelo.to('cpu')
        
        modelo.conf = 0.35
        modelo.classes = [0, 67]  # Personas y pistolas
        
        print(f"Modelo configurado para detectar: Personas (0) y Armas/Pistolas (67)")
        
        return modelo
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        sys.exit(1)

def detectar_amenazas(frame, modelo, rastreador):
    """Procesa un frame para detectar personas y posibles amenazas"""
    try:
        resultados = modelo(frame)
        detecciones = resultados.pandas().xyxy[0]
        
        personas = []
        armas = []
        
        for idx, deteccion in detecciones.iterrows():
            x1, y1, x2, y2 = int(deteccion['xmin']), int(deteccion['ymin']), int(deteccion['xmax']), int(deteccion['ymax'])
            confianza = deteccion['confidence']
            clase = int(deteccion['class'])
            
            if clase == 0:  # Persona
                personas.append([x1, y1, x2, y2, confianza])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                emoji = obtener_emoji_confianza(confianza)
                cv2.putText(frame, f"Persona {emoji}: {confianza:.2f}", (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                centro = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.circle(frame, centro, 5, (0, 255, 0), -1)
            
            elif clase == 67:  # Arma
                armas.append([x1, y1, x2, y2, confianza])
                emoji = obtener_emoji_confianza(confianza)
                print(f"{emoji} ¡DETECCIÓN DE ARMA! Confianza: {confianza:.2f}")
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"¡ARMA! {emoji}: {confianza:.2f}", (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "¡ALERTA: ARMA DETECTADA!", (frame.shape[1]//2 - 150, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        rastreador.actualizar(personas)
        frame = rastreador.dibujar(frame)
        analizar_posibles_amenazas(frame, personas, armas)
        
        return frame
    except Exception as e:
        print(f"Error al procesar el frame: {str(e)}")
        return frame

def analizar_posibles_amenazas(frame, personas, armas):
    """Analiza si hay posibles amenazas basándose en la posición de personas y armas"""
    if len(personas) >= 2:
        for i, persona1 in enumerate(personas):
            for persona2 in personas[i+1:]:
                centro1 = ((persona1[0] + persona1[2]) // 2, (persona1[1] + persona1[3]) // 2)
                centro2 = ((persona2[0] + persona2[2]) // 2, (persona2[1] + persona2[3]) // 2)
                
                distancia = np.sqrt((centro1[0] - centro2[0])**2 + (centro1[1] - centro2[1])**2)
                
                if 50 < distancia < 200:
                    cv2.line(frame, centro1, centro2, (255, 165, 0), 1)
                    
    if len(armas) > 0 and len(personas) > 0:
        for arma in armas:
            arma_centro_x = (arma[0] + arma[2]) // 2
            arma_centro_y = (arma[1] + arma[3]) // 2
            arma_centro = (arma_centro_x, arma_centro_y)
            
            persona_mas_cercana = None
            min_distancia = float('inf')
            
            for i, persona in enumerate(personas):
                persona_centro_x = (persona[0] + persona[2]) // 2
                persona_centro_y = (persona[1] + persona[3]) // 2
                persona_centro = (persona_centro_x, persona_centro_y)
                
                distancia = np.sqrt((arma_centro_x - persona_centro_x)**2 + (arma_centro_y - persona_centro_y)**2)
                
                if distancia < min_distancia:
                    min_distancia = distancia
                    persona_mas_cercana = (i, persona_centro)
            
            if persona_mas_cercana is not None and min_distancia < 100:
                idx_persona, centro_persona = persona_mas_cercana
                cv2.line(frame, centro_persona, arma_centro, (255, 0, 0), 2)
                
                dx = arma_centro_x - centro_persona[0]
                dy = arma_centro_y - centro_persona[1]
                
                longitud = np.sqrt(dx*dx + dy*dy)
                if longitud > 0:
                    dx /= longitud
                    dy /= longitud
                
                longitud_proyeccion = 300
                punto_final = (int(arma_centro_x + dx * longitud_proyeccion), 
                             int(arma_centro_y + dy * longitud_proyeccion))
                
                cv2.line(frame, arma_centro, punto_final, (255, 255, 0), 1)
                
                for j, otra_persona in enumerate(personas):
                    if j == idx_persona:
                        continue
                    
                    otra_centro_x = (otra_persona[0] + otra_persona[2]) // 2
                    otra_centro_y = (otra_persona[1] + otra_persona[3]) // 2
                    otra_centro = (otra_centro_x, otra_centro_y)
                    
                    dx_otro = otra_centro_x - arma_centro_x
                    dy_otro = otra_centro_y - arma_centro_y
                    
                    longitud_otro = np.sqrt(dx_otro*dx_otro + dy_otro*dy_otro)
                    if longitud_otro > 0:
                        dx_otro /= longitud_otro
                        dy_otro /= longitud_otro
                    
                    producto_escalar = dx*dx_otro + dy*dy_otro
                    
                    if producto_escalar > 0.65:
                        mensaje = "¡ALERTA! Arma apuntando a una persona"
                        emoji = obtener_emoji_confianza(arma[4])
                        print("\n" + "="*50)
                        print(f"{emoji} ¡AMENAZA DETECTADA! Arma apuntando a una persona")
                        print(f"Confianza de detección del arma: {arma[4]:.2f}")
                        print(f"Distancia entre arma y objetivo: {int(np.sqrt((arma_centro_x - otra_centro_x)**2 + (arma_centro_y - otra_centro_y)**2))} píxeles")
                        print(f"Ángulo de alineación: {producto_escalar:.2f} (0-1, donde 1 es perfecto)")
                        print("="*50 + "\n")
                        
                        cv2.line(frame, arma_centro, otra_centro, (0, 0, 255), 4)
                        cv2.putText(frame, mensaje, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        
                        if int(time.time() * 2) % 2 == 0:
                            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 20)
                        
                        cv2.circle(frame, otra_centro, 70, (0, 0, 255), 3)
                        distancia_texto = f"Dist: {int(np.sqrt((arma_centro_x - otra_centro_x)**2 + (arma_centro_y - otra_centro_y)**2))}px"
                        cv2.putText(frame, distancia_texto, 
                                  ((arma_centro_x + otra_centro_x) // 2, (arma_centro_y + otra_centro_y) // 2 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# =============================================
# FUNCIÓN PRINCIPAL
# =============================================
def main():
    try:
        print("\n" + "="*60)
        print("SISTEMA DE DETECCIÓN DE AMENAZAS CON ARMAS")
        print("="*60)
        
        print("\n[1/3] Inicializando sistema de detección...")
        modelo = cargar_modelo()
        print("[1/3] Sistema de detección inicializado con éxito!")
        
        ws_connection = conectar_websocket()
        rastreador = RastreadorMovimiento(max_puntos=15)
        
        print("\n[2/3] Conectando con cámara...")
        camara = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camara.set(cv2.CAP_PROP_FPS, 15)
        
        if not camara.isOpened():
            print("ERROR: No se pudo abrir la cámara")
            return
        
        ancho = camara.get(cv2.CAP_PROP_FRAME_WIDTH)
        alto = camara.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[2/3] Cámara conectada. Resolución: {int(ancho)}x{int(alto)}")
        
        print("\n[3/3] Iniciando monitoreo de amenazas...")
        print("-"*60)
        print("SISTEMA ACTIVO - Monitorizando amenazas")
        print("Presiona 'q' para salir del sistema")
        print("-"*60 + "\n")
        
        # Variables de control inicializadas
        tiempo_previo = time.time()
        ultimo_tiempo_deteccion = time.time()
        ultimo_reporte = time.time()
        contador_frames = 0
        frames_procesados = 0
        detecciones_armas = 0
        arma_detectada_antes = False
        ultimo_conteo_armas = 0
        procesar_cada_n_frames = 2
        contador_salto_frames = 0
        frame_procesado_anterior = None
        
        
        
        while True:
            ret, frame = camara.read()
            if not ret:
                print("Error: No se pudo obtener un frame")
                break
                
            frame_procesado = frame.copy()
            contador_salto_frames += 1
            frames_procesados += 1
            
            if contador_salto_frames >= procesar_cada_n_frames:
                contador_salto_frames = 0
                frame_procesado = detectar_amenazas(frame, modelo, rastreador)
                frame_procesado_anterior = frame_procesado
                
                resultados = modelo(frame)
                detecciones_df = resultados.pandas().xyxy[0]
                armas_actuales = len(detecciones_df[detecciones_df['class'] == 67])
                
                if armas_actuales > 0:
                    max_confianza = detecciones_df[detecciones_df['class'] == 67]['confidence'].max()
                    
                    if max_confianza >= 0.4:
                        if enviar_alerta_websocket(ws_connection, max_confianza):
                            print(f"Alerta enviada al servidor con nivel de confianza: {max_confianza:.2f}")
                        
                        if time.time() - ultimo_tiempo_deteccion > 2:
                            emoji = obtener_emoji_confianza(max_confianza)
                            print(f"\n{emoji} ALERTA: Se ha detectado {armas_actuales} arma(s) en la escena")
                            detecciones_armas += armas_actuales
                            ultimo_tiempo_deteccion = time.time()
                            arma_detectada_antes = True
                elif arma_detectada_antes and time.time() - ultimo_tiempo_deteccion > 2:
                    print("\n🟢 INFO: Ya no se detectan armas en la escena")
                    arma_detectada_antes = False
                    ultimo_tiempo_deteccion = time.time()
                
                ultimo_conteo_armas = armas_actuales
            else:
                if frame_procesado_anterior is not None:
                    frame_procesado = frame_procesado_anterior
            
            contador_frames += 1
            if (time.time() - tiempo_previo) > 1.0:
                fps = contador_frames / (time.time() - tiempo_previo)
                cv2.putText(frame_procesado, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                contador_frames = 0
                tiempo_previo = time.time()
            
            estado = "NORMAL" if ultimo_conteo_armas == 0 else "¡ALERTA!"
            color_estado = (0, 255, 0) if ultimo_conteo_armas == 0 else (0, 0, 255)
            cv2.putText(frame_procesado, f"Estado: {estado}", 
                       (10, frame_procesado.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)
            
            cv2.imshow('Sistema de Detección de Amenazas', frame_procesado)
            
            if time.time() - ultimo_reporte > 30:
                tiempo_operacion = time.time() - ultimo_reporte
                print("\n" + "-"*30)
                print(f"Reporte de estado - Últimos {int(tiempo_operacion)} segundos:")
                print(f"- Frames procesados: {frames_procesados}")
                print(f"- Tasa media: {frames_procesados/tiempo_operacion:.1f} FPS")
                print(f"- Nuevas detecciones de armas: {detecciones_armas}")
                print("-"*30 + "\n")
                frames_procesados = 0
                detecciones_armas = 0
                ultimo_reporte = time.time()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nFinalizando sistema de detección...")
                break

        if ws_connection:
            ws_connection.close()
        camara.release()
        cv2.destroyAllWindows()
        
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario")
    except Exception as e:
        print(f"Error inesperado: {str(e)}")

if __name__ == "__main__":
    main()