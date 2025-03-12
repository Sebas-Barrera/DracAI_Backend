import cv2
import numpy as np
import time
import os
import sys
import torch
import pandas as pd
from PIL import Image
import warnings

# Supresión agresiva de advertencias FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'

# Desactivamos todos los mensajes de advertencia de Torch
os.environ["TORCH_LOGS"] = "WARNING"
torch.set_warn_always(False)

def verificar_dependencias():
    """
    Verifica que todas las dependencias estén instaladas y muestra un mensaje claro si falta alguna
    """
    dependencias = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "torch": "torch torchvision",
        "pandas": "pandas",
        "PIL": "pillow"
    }
    
    faltantes = []
    
    for modulo, paquete in dependencias.items():
        try:
            __import__(modulo)
        except ImportError:
            faltantes.append(f"  - {paquete}")
    
    if faltantes:
        print("ERROR: Faltan las siguientes dependencias. Instálalas con:")
        print("pip install " + " ".join([dep.split(" - ")[1] for dep in faltantes]))
        print("\nDependencias faltantes:")
        print("\n".join(faltantes))
        sys.exit(1)

# Verificar dependencias antes de continuar
verificar_dependencias()

# Función para obtener emoji por nivel de confianza
def obtener_emoji_confianza(confianza):
    """
    Devuelve un emoji de color según el nivel de confianza
    """
    if confianza >= 0.7:
        return "🔴"  # Rojo - Alta confianza
    elif confianza >= 0.5:
        return "🟠"  # Naranja - Confianza media-alta
    elif confianza >= 0.4:
        return "🟡"  # Amarillo - Confianza media
    else:
        return "🟢"  # Verde - Confianza baja

# Clase para el rastreo de movimiento
class RastreadorMovimiento:
    def __init__(self, max_puntos=10):
        self.trayectorias = {}  # Diccionario para almacenar trayectorias de personas
        self.max_puntos = max_puntos  # Número máximo de puntos por trayectoria
        self.colores = {}  # Colores para cada trayectoria
        
    def actualizar(self, personas):
        # Si no hay trayectorias previas, inicializar con las personas actuales
        if not self.trayectorias:
            for i, persona in enumerate(personas):
                centro = ((persona[0] + persona[2]) // 2, (persona[1] + persona[3]) // 2)
                self.trayectorias[i] = [centro]
                self.colores[i] = tuple(np.random.randint(0, 255, 3).tolist())
            return
        
        # Asignar personas actuales a trayectorias existentes
        if personas:
            # Para cada persona detectada actualmente
            for persona in personas:
                centro_actual = ((persona[0] + persona[2]) // 2, (persona[1] + persona[3]) // 2)
                
                # Encontrar la trayectoria más cercana
                min_dist = float('inf')
                min_id = None
                
                for id_trayectoria, puntos in self.trayectorias.items():
                    if puntos:  # Si hay puntos en esta trayectoria
                        # Calcular distancia al último punto de la trayectoria
                        ultimo_punto = puntos[-1]
                        dist = np.sqrt((centro_actual[0] - ultimo_punto[0])**2 + 
                                      (centro_actual[1] - ultimo_punto[1])**2)
                        
                        if dist < min_dist:
                            min_dist = dist
                            min_id = id_trayectoria
                
                # Si encontramos una trayectoria cercana y la distancia es razonable
                if min_id is not None and min_dist < 100:  # Umbral de 100 píxeles
                    # Añadir el punto a la trayectoria
                    self.trayectorias[min_id].append(centro_actual)
                    # Limitar el número de puntos
                    if len(self.trayectorias[min_id]) > self.max_puntos:
                        self.trayectorias[min_id].pop(0)
                else:
                    # Crear una nueva trayectoria
                    nuevo_id = max(self.trayectorias.keys()) + 1 if self.trayectorias else 0
                    self.trayectorias[nuevo_id] = [centro_actual]
                    self.colores[nuevo_id] = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Eliminar trayectorias sin actualizar (personas que ya no están)
        # Lo implementaremos en una versión futura para evitar complejidad
    
    def dibujar(self, frame):
        # Dibujar las trayectorias
        for id_trayectoria, puntos in self.trayectorias.items():
            if len(puntos) < 2:
                continue
                
            # Dibujar líneas entre puntos consecutivos
            color = self.colores[id_trayectoria]
            for i in range(1, len(puntos)):
                # Hacemos las líneas más gruesas para los puntos más recientes
                grosor = int(np.sqrt(self.max_puntos / float(i + 1)) * 2.5)
                cv2.line(frame, puntos[i-1], puntos[i], color, grosor)
            
            # Dibujar los puntos de la trayectoria
            for i, punto in enumerate(puntos):
                # Puntos más recientes más grandes
                grosor = int(np.sqrt(self.max_puntos / float(i + 1)) * 2.5)
                cv2.circle(frame, punto, grosor, color, -1)
                
        # Dibujar una malla de movimiento simple
        if self.trayectorias:
            # Crear una capa separada para la malla
            malla = np.zeros_like(frame)
            
            # Dibujar líneas entre los últimos puntos de cada trayectoria
            ids = list(self.trayectorias.keys())
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    if (self.trayectorias[ids[i]] and self.trayectorias[ids[j]] and 
                        len(self.trayectorias[ids[i]]) > 0 and len(self.trayectorias[ids[j]]) > 0):
                        punto1 = self.trayectorias[ids[i]][-1]
                        punto2 = self.trayectorias[ids[j]][-1]
                        
                        # Calcular distancia entre puntos
                        dist = np.sqrt((punto1[0] - punto2[0])**2 + (punto1[1] - punto2[1])**2)
                        
                        # Solo dibujar líneas si están a una distancia razonable
                        if dist < 200:
                            # Color de línea basado en la distancia
                            intensidad = int(255 * (1 - dist/200))
                            cv2.line(malla, punto1, punto2, (0, intensidad, intensidad), 1)
            
            # Mezclar la malla con el frame original
            return cv2.addWeighted(frame, 1.0, malla, 0.4, 0)
        
        return frame
def cargar_modelo():
    """
    Carga el modelo YOLOv5 pre-entrenado para detectar personas y armas
    """
    try:
        # Usamos el modelo YOLOv5 a través de torch hub con manejo de advertencias
        print("Cargando YOLOv5 desde torch hub...")
        
        # Usamos un modelo más ligero para menor carga de CPU
        modelo = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
        
        # Si estamos en un entorno sin CUDA (como macOS), no forzar la GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando dispositivo: {device}")
        modelo.to(device)
        
        # Configuramos el modelo para mayor sensibilidad pero con un balance de rendimiento
        modelo.conf = 0.35  # Umbral ligeramente más alto para reducir falsos positivos
        
        # El modelo estándar COCO detecta:
        # 0: persona
        # 67: pistola (arma de fuego)
        
        # Clases para detección de personas y posibles armas
        modelo.classes = [0, 67]  # Personas y pistolas
        
        print(f"Modelo configurado para detectar: Personas (0) y Armas/Pistolas (67)")
        
        return modelo
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        sys.exit(1)

# Función para procesar cada frame y detectar amenazas
def detectar_amenazas(frame, modelo, rastreador):
    """
    Procesa un frame para detectar personas y posibles amenazas
    """
    try:
        # Realizar la detección
        resultados = modelo(frame)
        
        # Extraer información relevante
        detecciones = resultados.pandas().xyxy[0]  # Resultados en formato pandas
        
        # Lista para almacenar personas detectadas (format: [x1, y1, x2, y2, confianza])
        personas = []
        # Lista para almacenar armas detectadas
        armas = []
        
        # Procesar cada detección
        for idx, deteccion in detecciones.iterrows():
            x1, y1, x2, y2 = int(deteccion['xmin']), int(deteccion['ymin']), int(deteccion['xmax']), int(deteccion['ymax'])
            confianza = deteccion['confidence']
            clase = int(deteccion['class'])
            
            # Si es una persona
            if clase == 0:
                # Guardar información de la persona
                personas.append([x1, y1, x2, y2, confianza])
                
                # Dibujar rectángulo alrededor de la persona
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Añadir emoji de confianza
                emoji = obtener_emoji_confianza(confianza)
                cv2.putText(frame, f"Persona {emoji}: {confianza:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Dibujar un punto en el centro de la persona (para referencia de movimiento)
                centro = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.circle(frame, centro, 5, (0, 255, 0), -1)
            
            # Si es un arma (pistola)
            elif clase == 67:  # Clase 67 en COCO = pistola
                # Guardar información del arma
                armas.append([x1, y1, x2, y2, confianza])
                
                # Notificar en la terminal sobre la detección de un arma
                emoji = obtener_emoji_confianza(confianza)
                print(f"{emoji} ¡DETECCIÓN DE ARMA! Confianza: {confianza:.2f}")
                
                # Dibujar rectángulo alrededor del arma con rojo brillante
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Línea más gruesa
                cv2.putText(frame, f"¡ARMA! {emoji}: {confianza:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Añadir un mensaje de alerta en la parte superior de la pantalla
                cv2.putText(frame, "¡ALERTA: ARMA DETECTADA!", (frame.shape[1]//2 - 150, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Actualizar el rastreador de movimiento con las personas detectadas
        rastreador.actualizar(personas)
        
        # Dibujar las trayectorias y malla de movimiento
        frame = rastreador.dibujar(frame)
        
        # Analizar posibles amenazas (ahora con armas)
        analizar_posibles_amenazas(frame, personas, armas)
        
        return frame
    except Exception as e:
        print(f"Error al procesar el frame: {str(e)}")
        return frame

# Función para analizar posibles amenazas entre personas y armas
def analizar_posibles_amenazas(frame, personas, armas):
    """
    Analiza si hay posibles amenazas basándose en la posición de personas y armas
    """
    # Analizar interacciones entre personas (código original)
    if len(personas) >= 2:
        for i, persona1 in enumerate(personas):
            for persona2 in personas[i+1:]:
                centro1 = ((persona1[0] + persona1[2]) // 2, (persona1[1] + persona1[3]) // 2)
                centro2 = ((persona2[0] + persona2[2]) // 2, (persona2[1] + persona2[3]) // 2)
                
                distancia = np.sqrt((centro1[0] - centro2[0])**2 + (centro1[1] - centro2[1])**2)
                
                if 50 < distancia < 200:
                    cv2.line(frame, centro1, centro2, (255, 165, 0), 1)  # Línea naranja para interacción normal
                    
    # NUEVO: Analizar si hay armas y si están cerca de personas (posible amenaza)
    if len(armas) > 0 and len(personas) > 0:
        for arma in armas:
            # Centro del arma
            arma_centro_x = (arma[0] + arma[2]) // 2
            arma_centro_y = (arma[1] + arma[3]) // 2
            arma_centro = (arma_centro_x, arma_centro_y)
            
            # Encontrar la persona más cercana al arma (posiblemente sosteniéndola)
            persona_mas_cercana = None
            min_distancia = float('inf')
            
            for i, persona in enumerate(personas):
                persona_centro_x = (persona[0] + persona[2]) // 2
                persona_centro_y = (persona[1] + persona[3]) // 2
                persona_centro = (persona_centro_x, persona_centro_y)
                
                distancia = np.sqrt((arma_centro_x - persona_centro_x)**2 + (arma_centro_y - persona_centro_y)**2)
                
                # Si el arma está cerca de una persona, podría estar sosteniéndola
                if distancia < min_distancia:
                    min_distancia = distancia
                    persona_mas_cercana = (i, persona_centro)
            
            # Si encontramos una persona cerca del arma (potencialmente sosteniéndola)
            if persona_mas_cercana is not None and min_distancia < 100:  # Umbral de 100 píxeles, ajustar según escala
                idx_persona, centro_persona = persona_mas_cercana
                
                # Conectar la persona con el arma
                cv2.line(frame, centro_persona, arma_centro, (255, 0, 0), 2)  # Línea azul
                
                # Determinar la dirección del arma (relativa a la persona)
                # El vector desde la persona hacia el arma
                dx = arma_centro_x - centro_persona[0]
                dy = arma_centro_y - centro_persona[1]
                
                # Normalizar el vector para obtener la dirección
                longitud = np.sqrt(dx*dx + dy*dy)
                if longitud > 0:
                    dx /= longitud
                    dy /= longitud
                
                # Extender esta dirección para crear una línea de proyección
                longitud_proyeccion = 300  # Longitud de la línea de proyección
                punto_final = (int(arma_centro_x + dx * longitud_proyeccion), 
                               int(arma_centro_y + dy * longitud_proyeccion))
                
                # Dibujar la línea de proyección (dirección del arma)
                cv2.line(frame, arma_centro, punto_final, (255, 255, 0), 1)  # Línea amarilla
                
                # Comprobar si esta línea intersecta con alguna otra persona
                # Excluimos la persona que sostiene el arma
                for j, otra_persona in enumerate(personas):
                    if j == idx_persona:  # Saltar la persona que sostiene el arma
                        continue
                    
                    otra_centro_x = (otra_persona[0] + otra_persona[2]) // 2
                    otra_centro_y = (otra_persona[1] + otra_persona[3]) // 2
                    otra_centro = (otra_centro_x, otra_centro_y)
                    
                    # Vector desde el arma hacia la otra persona
                    dx_otro = otra_centro_x - arma_centro_x
                    dy_otro = otra_centro_y - arma_centro_y
                    
                    # Comprobar si la dirección de este vector es similar a la dirección del arma
                    # Calculamos el producto escalar de los vectores normalizados
                    longitud_otro = np.sqrt(dx_otro*dx_otro + dy_otro*dy_otro)
                    if longitud_otro > 0:
                        dx_otro /= longitud_otro
                        dy_otro /= longitud_otro
                    
                    # Producto escalar (coseno del ángulo entre vectores)
                    producto_escalar = dx*dx_otro + dy*dy_otro
                    
                    # Si el coseno es cercano a 1, los vectores apuntan en direcciones similares
                    if producto_escalar > 0.65:  # Ángulo menor a unos 50 grados (más sensible)
                        # ¡ALERTA! El arma parece estar apuntando a otra persona
                        mensaje = "¡ALERTA! Arma apuntando a una persona"
                        
                        # Notificar en la terminal con información detallada
                        emoji = obtener_emoji_confianza(arma[4])
                        print("\n" + "="*50)
                        print(f"{emoji} ¡AMENAZA DETECTADA! Arma apuntando a una persona")
                        print(f"Confianza de detección del arma: {arma[4]:.2f}")
                        print(f"Distancia entre arma y objetivo: {int(np.sqrt((arma_centro_x - otra_centro_x)**2 + (arma_centro_y - otra_centro_y)**2))} píxeles")
                        print(f"Ángulo de alineación: {producto_escalar:.2f} (0-1, donde 1 es perfecto)")
                        print("="*50 + "\n")
                        
                        # Marcar visualmente la amenaza de forma más notoria
                        cv2.line(frame, arma_centro, otra_centro, (0, 0, 255), 4)  # Línea roja más gruesa
                        
                        # Texto de alerta más visible
                        cv2.putText(frame, mensaje, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        
                        # Añadir marco rojo parpadeante a toda la pantalla (efecto de alerta)
                        thickness = 20  # Grosor del borde
                        if int(time.time() * 2) % 2 == 0:  # Parpadeo cada 0.5 segundos
                            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), thickness)
                        
                        # Dibujar círculo rojo alrededor de la persona amenazada
                        cv2.circle(frame, otra_centro, 70, (0, 0, 255), 3)
                        
                        # Mostrar distancia entre el arma y la persona objetivo
                        distancia_texto = f"Dist: {int(np.sqrt((arma_centro_x - otra_centro_x)**2 + (arma_centro_y - otra_centro_y)**2))}px"
                        cv2.putText(frame, distancia_texto, 
                                  ((arma_centro_x + otra_centro_x) // 2, (arma_centro_y + otra_centro_y) // 2 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Función principal
def main():
    """
    Función principal que ejecuta la detección en tiempo real
    """
    try:
        print("\n" + "="*60)
        print("SISTEMA DE DETECCIÓN DE AMENAZAS CON ARMAS")
        print("="*60)
        
        # Cargar el modelo
        print("\n[1/3] Inicializando sistema de detección...")
        modelo = cargar_modelo()
        print("[1/3] Sistema de detección inicializado con éxito!")
        
        # Inicializar el rastreador de movimiento
        rastreador = RastreadorMovimiento(max_puntos=15)
        
        # Iniciar la cámara - MODIFICADO PARA MAC
        print("\n[2/3] Conectando con cámara...")
        # Utilizamos CAP_AVFOUNDATION para macOS
        camara = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        # Intentar configurar la resolución para mejor rendimiento
        # Usamos resolución más baja para mejor rendimiento
        camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not camara.isOpened():
            print("ERROR: No se pudo abrir la cámara")
            print("Asegúrate de que la cámara esté conectada y no esté siendo usada por otra aplicación.")
            return
        
        # Obtener la resolución real
        ancho = camara.get(cv2.CAP_PROP_FRAME_WIDTH)
        alto = camara.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[2/3] Cámara conectada. Resolución: {int(ancho)}x{int(alto)}")
        
        print("\n[3/3] Iniciando monitoreo de amenazas...")
        print("-"*60)
        print("SISTEMA ACTIVO - Monitorizando amenazas")
        print("Instrucciones:")
        print("- Se alertará en esta terminal cuando se detecte un arma")
        print("- Se mostrará un mensaje detallado si el arma apunta a una persona")
        print("- Presiona 'q' para salir del sistema")
        print("-"*60 + "\n")
        
        # Variables para control de rendimiento
        tiempo_previo = time.time()
        contador_frames = 0
        
        # Para estadísticas
        ultimo_reporte = time.time()
        frames_procesados = 0
        detecciones_armas = 0
        
        # Variables para contar armas detectadas
        arma_detectada_antes = False
        ultimo_conteo_armas = 0
        
        # Variable para procesar solo cada N frames (optimización)
        procesar_cada_n_frames = 2  # Procesar cada 2 frames
        contador_salto_frames = 0
        
        # Frame anterior para usar cuando saltamos frames
        frame_procesado_anterior = None
        
        # Escenario de prueba descrito
        print("\n" + "-"*60)
        print("ESCENARIO DE PRUEBA RECOMENDADO:")
        print("1. Primero, aparezca solo en el video para verificar la detección de personas")
        print("2. Muestre un objeto que simule un arma (control remoto, puntero láser, etc.)")
        print("3. Acerque este objeto 'arma' a su mano para que el sistema detecte que lo sostiene")
        print("4. Si hay otra persona, apunte el objeto hacia ella para activar la alerta")
        print("5. Si está solo, puede usar un muñeco o imagen como segundo objetivo")
        print("-"*60 + "\n")
        
        # Memoria para optimización
        ultimo_tiempo_deteccion = time.time()
        
        while True:
            # Leer un frame de la cámara
            ret, frame = camara.read()
            
            # Si no hay frame, salir
            if not ret:
                print("Error: No se pudo obtener un frame")
                break
            
            # Incrementar contador de frames
            contador_salto_frames += 1
            frames_procesados += 1
            
            # Procesar solo cada N frames para ahorrar CPU
            if contador_salto_frames >= procesar_cada_n_frames:
                contador_salto_frames = 0
                
                # Procesar el frame para detectar amenazas
                frame_procesado = detectar_amenazas(frame, modelo, rastreador)
                frame_procesado_anterior = frame_procesado
                
                # Contar armas en la escena actual
                resultados = modelo(frame)
                detecciones_df = resultados.pandas().xyxy[0]
                armas_actuales = len(detecciones_df[detecciones_df['class'] == 67])
                
                # Notificar si aparece un arma nueva en la escena (solo si han pasado al menos 2 segundos)
                if time.time() - ultimo_tiempo_deteccion > 2:
                    if armas_actuales > 0 and not arma_detectada_antes:
                        emoji = obtener_emoji_confianza(max(detecciones_df[detecciones_df['class'] == 67]['confidence']) if not detecciones_df[detecciones_df['class'] == 67].empty else 0.3)
                        print(f"\n{emoji} ALERTA: Se ha detectado {armas_actuales} arma(s) en la escena")
                        arma_detectada_antes = True
                        detecciones_armas += 1
                        ultimo_tiempo_deteccion = time.time()
                    elif armas_actuales == 0 and arma_detectada_antes:
                        print(f"\n🟢 INFO: Ya no se detectan armas en la escena")
                        arma_detectada_antes = False
                        ultimo_tiempo_deteccion = time.time()
                    elif armas_actuales > ultimo_conteo_armas:
                        emoji = obtener_emoji_confianza(max(detecciones_df[detecciones_df['class'] == 67]['confidence']) if not detecciones_df[detecciones_df['class'] == 67].empty else 0.3)
                        print(f"\n{emoji} ALERTA: Se ha detectado {armas_actuales - ultimo_conteo_armas} arma(s) adicional(es)")
                        detecciones_armas += armas_actuales - ultimo_conteo_armas
                        ultimo_tiempo_deteccion = time.time()
                
                ultimo_conteo_armas = armas_actuales
            else:
                # Si no procesamos este frame, usamos el anterior
                frame_procesado = frame_procesado_anterior if frame_procesado_anterior is not None else frame
            
            # Calcular y mostrar FPS
            contador_frames += 1
            if (time.time() - tiempo_previo) > 1.0:
                fps = contador_frames / (time.time() - tiempo_previo)
                cv2.putText(frame_procesado, f"FPS: {fps:.1f}", (
                    10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                contador_frames = 0
                tiempo_previo = time.time()
            
            # Mostrar estado del sistema
            estado = "NORMAL" if ultimo_conteo_armas == 0 else "¡ALERTA!"
            color_estado = (0, 255, 0) if ultimo_conteo_armas == 0 else (0, 0, 255)
            cv2.putText(frame_procesado, f"Estado: {estado}", (10, frame_procesado.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)
            
            # Mostrar el frame
            cv2.imshow('Sistema de Detección de Amenazas', frame_procesado)
            
            # Mostrar estadísticas cada 30 segundos
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
            
            # Si se presiona 'q', salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nFinalizando sistema de detección por solicitud del usuario...")
                break
        
        # Liberar recursos
        camara.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario")
    except Exception as e:
        print(f"Error inesperado: {str(e)}")

# Ejecutar el programa si se llama directamente
if __name__ == "__main__":
    main()