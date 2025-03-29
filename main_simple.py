import cv2
import numpy as np
import time
import os
import sys
import warnings
import websocket
import json
from datetime import datetime

# =============================================
# CONFIGURACIÓN Y PARÁMETROS
# =============================================
# Cambiar esta dirección IP al servidor websocket
WS_URL = "ws://192.168.0.224:8080"

# =============================================
# CONFIGURACIÓN INICIAL PARA macOS
# =============================================
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
    warnings.filterwarnings("ignore", message="AVCaptureDeviceTypeExternal is deprecated")

# Ignorar warnings de PyTorch/CUDA
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================
# FUNCIONES AUXILIARES
# =============================================
def conectar_websocket():
    """Crea una conexión al servidor WebSocket"""
    try:
        ws = websocket.create_connection(WS_URL)
        print("Conectado al servidor WebSocket")
        return ws
    except Exception as e:
        print(f"Error al conectar con el WebSocket: {str(e)}")
        return None

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
            "tipo": tipo_alerta,
        }

        ws.send(json.dumps(datos_alerta))
        print(f"ALERTA ENVIADA: {tipo_alerta} con confianza {confianza:.2f}")
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
                centro = (
                    (persona[0] + persona[2]) // 2,
                    (persona[1] + persona[3]) // 2,
                )
                self.trayectorias[i] = [centro]
                self.colores[i] = tuple(np.random.randint(0, 255, 3).tolist())
            return

        if personas:
            for persona in personas:
                centro_actual = (
                    (persona[0] + persona[2]) // 2,
                    (persona[1] + persona[3]) // 2,
                )

                min_dist = float("inf")
                min_id = None

                for id_trayectoria, puntos in self.trayectorias.items():
                    if puntos:
                        ultimo_punto = puntos[-1]
                        dist = np.sqrt(
                            (centro_actual[0] - ultimo_punto[0]) ** 2
                            + (centro_actual[1] - ultimo_punto[1]) ** 2
                        )

                        if dist < min_dist:
                            min_dist = dist
                            min_id = id_trayectoria

                if min_id is not None and min_dist < 100:
                    self.trayectorias[min_id].append(centro_actual)
                    if len(self.trayectorias[min_id]) > self.max_puntos:
                        self.trayectorias[min_id].pop(0)
                else:
                    nuevo_id = (
                        max(self.trayectorias.keys()) + 1 if self.trayectorias else 0
                    )
                    self.trayectorias[nuevo_id] = [centro_actual]
                    self.colores[nuevo_id] = tuple(
                        np.random.randint(0, 255, 3).tolist()
                    )

    def dibujar(self, frame):
        for id_trayectoria, puntos in self.trayectorias.items():
            if len(puntos) < 2:
                continue

            color = self.colores[id_trayectoria]
            for i in range(1, len(puntos)):
                grosor = int(np.sqrt(self.max_puntos / float(i + 1)) * 2.5)
                cv2.line(frame, puntos[i - 1], puntos[i], color, grosor)

            for i, punto in enumerate(puntos):
                grosor = int(np.sqrt(self.max_puntos / float(i + 1)) * 2.5)
                cv2.circle(frame, punto, grosor, color, -1)

        if self.trayectorias:
            malla = np.zeros_like(frame)
            ids = list(self.trayectorias.keys())
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if (
                        self.trayectorias[ids[i]]
                        and self.trayectorias[ids[j]]
                        and len(self.trayectorias[ids[i]]) > 0
                        and len(self.trayectorias[ids[j]]) > 0
                    ):
                        punto1 = self.trayectorias[ids[i]][-1]
                        punto2 = self.trayectorias[ids[j]][-1]

                        dist = np.sqrt(
                            (punto1[0] - punto2[0]) ** 2 + (punto1[1] - punto2[1]) ** 2
                        )

                        if dist < 200:
                            intensidad = int(255 * (1 - dist / 200))
                            cv2.line(
                                malla, punto1, punto2, (0, intensidad, intensidad), 1
                            )

            return cv2.addWeighted(frame, 1.0, malla, 0.4, 0)

        return frame

# =============================================
# FUNCIÓN PRINCIPAL
# =============================================
def main():
    try:
        print("\n" + "="*60)
        print("SISTEMA DE DETECCIÓN DE AMENAZAS CON ARMAS")
        print("="*60)
        
        print("\n[1/3] Inicializando sistema...")
        
        # Crear conexión WebSocket
        ws_connection = conectar_websocket()
        rastreador = RastreadorMovimiento(max_puntos=15)
        
        # Inicializar cámara
        print("[2/3] Conectando con cámara...")
        
        # En macOS, intentamos abrir la cámara de forma más segura
        camara = None
        for intento in range(3):
            try:
                if sys.platform == "darwin":
                    camara = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
                else:
                    camara = cv2.VideoCapture(0)
                
                if camara.isOpened():
                    break
                else:
                    print(f"Intento {intento+1} de abrir la cámara falló.")
                    time.sleep(1)
            except Exception as e:
                print(f"Error en intento {intento+1}: {e}")
                time.sleep(1)
        
        if camara is None or not camara.isOpened():
            print("ERROR: No se pudo abrir la cámara después de varios intentos.")
            return
        
        # Configurar cámara
        camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camara.set(cv2.CAP_PROP_FPS, 15)
        
        ancho = camara.get(cv2.CAP_PROP_FRAME_WIDTH)
        alto = camara.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[2/3] Cámara conectada. Resolución: {int(ancho)}x{int(alto)}")
        
        # Cargar modelo YOLOv5
        print("\n[3/3] Cargando modelo de detección...")
        import torch
        
        # CLAVE: Usar CPU explícitamente para evitar problemas CUDA en macOS
        print("Forzando ejecución en CPU...")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        # Usar YOLOv5s (mejor detección)
        modelo = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', trust_repo=True)
        modelo.to('cpu')  # Forzar CPU
        
        # Clase 0 = persona, clase de arma varía según modelo (probar diferentes opciones)
        # COCO: clase 0 = persona
        # COCO: probemos clases que podrían ser armas (revisar output)
        CLASES_YOLO = [0]  # Persona siempre
        
        # IMPORTANTE: En vez de fijar una clase para arma, imprimiremos todas 
        # las detecciones y sus clases para identificar cuál corresponde a armas
        print("Modo de detección: se imprimirán todas las detecciones para identificar armas")
        modelo.conf = 0.35  # umbral de confianza
        
        # Verificar si se creó correctamente el modelo
        if modelo is None:
            print("ERROR: No se pudo cargar el modelo YOLOv5")
            return
            
        print("[3/3] Modelo cargado correctamente")
        
        print("\nSISTEMA ACTIVO - Monitorizando amenazas")
        print("Presiona 'q' para salir del sistema")
        print("-"*60 + "\n")
        
        # Crear ventana una sola vez
        cv2.namedWindow('Sistema de Detección de Amenazas', cv2.WINDOW_NORMAL)
        
        # Variables de control
        fps_time = time.time()
        frames_count = 0
        ultimo_tiempo_deteccion = time.time()
        arma_detectada_antes = False
        
        # El bucle principal ahora es simple y directo
        while True:
            # 1. Capturar frame
            ret, frame = camara.read()
            if not ret:
                print("Error al capturar frame")
                time.sleep(0.1)
                continue
            
            # 2. Procesar el frame y detectar objetos
            resultados = modelo(frame)
            detecciones = resultados.pandas().xyxy[0]
            
            # Imprimimos todas las detecciones para identificar armas
            if not detecciones.empty:
                print("\nDetecciones en este frame:")
                for idx, det in detecciones.iterrows():
                    print(f"Clase: {int(det['class'])} ('{det['name']}'), Confianza: {det['confidence']:.2f}")
                
            # 3. Analizar resultados y extraer información
            personas = []
            armas = []
            
            # Buscar específicamente la clase "pistol" o "gun" por nombre en vez de índice
            for idx, deteccion in detecciones.iterrows():
                x1, y1, x2, y2 = int(deteccion['xmin']), int(deteccion['ymin']), int(deteccion['xmax']), int(deteccion['ymax'])
                confianza = deteccion['confidence']
                clase = int(deteccion['class'])
                nombre = deteccion['name'] if 'name' in deteccion else ""
                
                if clase == 0:  # Persona
                    personas.append([x1, y1, x2, y2, confianza])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    emoji = obtener_emoji_confianza(confianza)
                    cv2.putText(frame, f"Persona {emoji}: {confianza:.2f}", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # CORREGIDO: Buscar armas por nombre en vez de por índice fijo
                elif nombre.lower() in ['pistol', 'gun', 'handgun', 'weapon', 'arma', 'knife'] or clase == 67:
                    # Encontramos un arma/objeto peligroso
                    armas.append([x1, y1, x2, y2, confianza])
                    emoji = obtener_emoji_confianza(confianza)
                    
                    # Imprimir información en terminal
                    print(f"{emoji} ¡DETECCIÓN DE ARMA ({nombre})! Confianza: {confianza:.2f}")
                    
                    # Dibujar en el frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"¡ARMA! {emoji}: {confianza:.2f}", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Texto de alerta central
                    cv2.putText(frame, "¡ALERTA: ARMA DETECTADA!", (frame.shape[1]//2 - 150, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # MODIFICADO: Enviar alerta siempre, sin filtros
                    enviar_alerta_websocket(ws_connection, confianza)
                    arma_detectada_antes = True
                    ultimo_tiempo_deteccion = time.time()
            
            # Verificar si ya no hay armas después de haber detectado alguna
            if arma_detectada_antes and len(armas) == 0 and time.time() - ultimo_tiempo_deteccion > 5:
                print("\n🟢 INFO: Ya no se detectan armas en la escena")
                arma_detectada_antes = False
            
            # 4. Actualizar rastreador de movimiento
            rastreador.actualizar(personas)
            frame = rastreador.dibujar(frame)
            
            # 5. Dibujar información adicional
            # Calcular FPS
            frames_count += 1
            tiempo_actual = time.time()
            if tiempo_actual - fps_time > 1.0:
                fps = frames_count / (tiempo_actual - fps_time)
                frames_count = 0
                fps_time = tiempo_actual
                
                # Dibujar FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Estado WebSocket
            cv2.putText(frame, f"WebSocket: {'✓' if ws_connection else '✗'}", 
                       (frame.shape[1] - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 255, 0) if ws_connection else (0, 0, 255), 2)
            
            # 6. Mostrar frame
            try:
                cv2.imshow('Sistema de Detección de Amenazas', frame)
            except cv2.error as e:
                print(f"Error al mostrar imagen: {e}")
                # Intentar recrear la ventana
                cv2.destroyAllWindows()
                time.sleep(0.1)
                cv2.namedWindow('Sistema de Detección de Amenazas', cv2.WINDOW_NORMAL)
            
            # 7. Verificar si el usuario quiere salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nFinalizando sistema de detección...")
                break
        
        # Limpiar recursos
        if ws_connection:
            ws_connection.close()
        camara.release()
        cv2.destroyAllWindows()
        
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario")
        if 'ws_connection' in locals() and ws_connection:
            ws_connection.close()
        if 'camara' in locals() and camara:
            camara.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error inesperado: {str(e)}")

if __name__ == "__main__":
    main()