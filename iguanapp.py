import os 
import sys
import tkinter as tk 
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import folium
from folium import plugins
import webbrowser
import tempfile
import numpy as np
from datetime import datetime
import json
import sqlite3
import shutil
from ultralytics import YOLO
import cv2
import uuid
import re

class IguanaSightingsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Registro de Avistamientos de Iguanas Verdes")
        self.root.geometry("800x600")
        self.root.configure(bg="#100F0F")
        
        # Rutas configurables
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.icon_path = os.path.join(self.base_dir, "icons", "iguanapp.ico")
        self.model_path = os.path.join(self.base_dir, "yolo_model", "best.pt")
        self.default_image_path = os.path.join(self.base_dir, "images", "iguanapp.png")
        self.saved_images_dir = os.path.join(self.base_dir, "saved_sightings")
        
        # Crea dictorio para imágenes guardadas
        os.makedirs(self.saved_images_dir, exist_ok=True)
        
        # Configuracion del icono de la aplicacion
        if os.path.exists(self.icon_path):
            try:
                self.root.iconbitmap(self.icon_path)
            except Exception as e:
                print(f"No se pudo cargar el icono: {e}")
        
        # Variables de la aplicación, estado inicial para las rutas de imagenes, resultados, coordenadas, etc.
        self.current_image_path = None
        self.location_coords = None
        self.detection_result = None
        self.db_path = "iguana_sightings.db"
        self.saved_image_path = None
        
        # Inicializacion base de datos
        self.init_database()
        
        # Carga del modelo YOLOv8
        self.load_yolo_model()
        
        # Creación del interfaz
        self.create_widgets()
    
    #Carga del modelo YOLOv8
    def load_yolo_model(self):
        """Carga el modelo YOLO con manejo de errores."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado en: {self.model_path}")
            
            self.model = YOLO(self.model_path)
            print("Modelo YOLO cargado correctamente.")
        except Exception as e:
            error_msg = f"No se pudo cargar el modelo YOLO: {str(e)}\n\nVerifica que el archivo 'best.pt' esté en la carpeta 'yolo_model'"
            messagebox.showerror("Error Crítico", error_msg)
            sys.exit(1)
    
    def create_widgets(self):
        """Crea los widgets de la interfaz de usuario."""
        # Panel superior para imágenes y controles
        top_frame = tk.Frame(self.root, width=400, height=300, padx=10, pady=10, bg="#6CE45E")
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Panel inferior para opciones de mapa y coordenadas
        bottom_frame = tk.Frame(self.root, width=400, height=300, padx=10, pady=10, bg="#6CE45E")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Área de imágenes
        self.image_label = tk.Label(top_frame, bg="#08BE38")
        self.image_label.pack(pady=10)
        
        # Muestra de la imagen por defecto
        if os.path.exists(self.default_image_path):
            self.display_image(self.default_image_path)
        else:
            self.image_label.config(text="No hay imagen por defecto", fg="white")
        
        # Botones de control de imágenes
        buttons_frame = tk.Frame(top_frame, bg="#6CE45E")
        buttons_frame.pack()
        
        # Botón para seleccionar imágenes
        self.btn_select = tk.Button(buttons_frame, text="Seleccionar Imagen",
                                    font=("Arial", 10, "bold"), fg="white",
                                    bg="#292929",
                                    command=self.select_image)
        self.btn_select.grid(row=0, column=0, padx=5)
        
        # Botón para detectar iguanas
        self.btn_detect = tk.Button(buttons_frame, text="Detectar Iguana", 
                                    font=("Arial", 10, "bold"), fg="white",
                                    bg="#292929", state=tk.DISABLED,
                                    command=self.detect_iguana)
        self.btn_detect.grid(row=0, column=1, padx=5)
        
        # Área de información de detección
        self.info_frame = tk.LabelFrame(top_frame, text="Información de Detección",
                                       padx=10, pady=10,
                                       bg="#292929", fg="white",
                                       font=("Arial", 10, "bold"))
        self.info_frame.pack(fill=tk.X, pady=10)
        
        # Etiqueta de resultado de detección
        self.result_label = tk.Label(self.info_frame, text="Resultado de Detección: No analizado", 
                                     font=("Arial", 10, "bold"), fg="white",
                                     bg="#292929")
        self.result_label.pack(anchor=tk.W)
        
        # Entrada de coordenadas
        coords_frame = tk.Frame(bottom_frame, bg="#6CE45E")
        coords_frame.pack(fill=tk.X, pady=10)
        
        # Configuracion de columnas para centrar
        coords_frame.grid_columnconfigure(0, weight=1)
        coords_frame.grid_columnconfigure(5, weight=1)
        
        # Título de coordenadas
        tk.Label(coords_frame, text="Coordenadas de avistamiento",
                font=("Arial", 12, "bold"), fg="white", bg="#6CE45E").grid(
                    row=0, column=1, columnspan=4, pady=(0, 10))
        
        # Entrada de latitud
        tk.Label(coords_frame, text="Latitud:",
                 font=("Arial", 10, "bold"), fg="white",
                 bg="#292929").grid(row=1, column=1, sticky=tk.W)
        self.lat_entry = tk.Entry(coords_frame, width=15)
        self.lat_entry.grid(row=1, column=2, padx=5)
        
        # Entrada de longitud 
        tk.Label(coords_frame, text="Longitud:",
                 font=("Arial", 10, "bold"), fg="white",
                 bg="#292929").grid(row=1, column=3, sticky=tk.W)
        self.lon_entry = tk.Entry(coords_frame, width=15)
        self.lon_entry.grid(row=1, column=4, padx=5)
        
        # Botones para el mapa
        map_buttons = tk.Frame(bottom_frame, bg="#6CE45E")
        map_buttons.pack(pady=5)
        
        # Botón para actualizar y abrir el mapa
        self.btn_update_map = tk.Button(map_buttons, text="Actualizar mapa",
                                        font=("Arial", 10, "bold"), fg="white",
                                        bg="#292929", 
                                        command=self.update_map)
        self.btn_update_map.grid(row=0, column=0, padx=5)
        
        # Botón para guardar avistamiento
        self.btn_save_sighting = tk.Button(map_buttons, text="Guardar avistamiento",
                                           font=("Arial", 10, "bold"), fg="white",
                                           bg="#292929", 
                                           command=self.save_sighting,
                                           state=tk.DISABLED)
        self.btn_save_sighting.grid(row=0, column=1, padx=5)
        
        # Botón para mostrar todos los avistamientos
        self.btn_show_all = tk.Button(map_buttons, text="Ver todos los avistamientos",
                                      font=("Arial", 10, "bold"), fg="white",
                                      bg="#292929", 
                                      command=self.show_all_sightings)
        self.btn_show_all.grid(row=0, column=2, padx=5)
        
        self.btn_explore_map = tk.Button(map_buttons, text="Explorar mapa",
                                   font=("Arial", 10, "bold"), fg="white",
                                   bg="#292929", 
                                   command=self.explore_interactive_map)
        self.btn_explore_map.grid(row=0, column=3, padx=5)
        
    def explore_interactive_map(self):
        """Abre un mapa interactivo para explorar y obtener coordenadas sin necesidad de avistamientos."""
        try:
            # Creacion mapa centrado en Panamá
            center_location = [8.9943, -79.5188]  # Centro de Panamá
            m = self.create_enhanced_exploration_map(center_location, zoom_start=8)
            
            # Guardado del mapa temporal
            temp_map = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            m.save(temp_map.name)
            
            # Con esto se abre el mapa en el navegador
            webbrowser.open('file://' + temp_map.name, new=2)
            
            messagebox.showinfo("Bienvenido", f"Mapa de Exploración")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir el mapa de exploración: {str(e)}")
    
    def create_enhanced_exploration_map(self, center_location, zoom_start=10):
        """Crea un mapa interactivo mejorado para exploración con popup de coordenadas."""
        # Creacion el mapa base
        m = folium.Map(location=center_location, zoom_start=zoom_start)
        
        # Codigo JavaScript para capturar clicks
        click_script = """
        <script>
        function onMapClick(e) {
            var lat = e.latlng.lat.toFixed(6);
            var lng = e.latlng.lng.toFixed(6);
            
            // Crear popup con las coordenadas
            var popup = L.popup()
                .setLatLng(e.latlng)
                .setContent('<b>Coordenadas:</b><br>Latitud: ' + lat + '<br>Longitud: ' + lng)
                .openOn(this);
            
            console.log('Latitud: ' + lat + ', Longitud: ' + lng);
        }
        
        // Esperar a que el mapa se cargue
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                var mapId = Object.keys(window).find(key => key.startsWith('map_'));
                if (mapId && window[mapId]) {
                    window[mapId].on('click', onMapClick);
                }
            }, 100);
        });
        </script>
        """
        
        # Agreganndo el script al mapa de Folium
        m.get_root().html.add_child(folium.Element(click_script))
        
        return m
        
    def init_database(self):
        """Inicializa la base de datos y crea la tabla si no existe."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Se crea una tabla para los avistamientos en la base de datos (sino existe)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL NOT NULL, 
            longitude REAL NOT NULL,
            original_image_path TEXT NOT NULL,
            saved_image_path TEXT NOT NULL,
            detection_confidence REAL,
            detections_count INTEGER,
            timestamp TEXT
        )
        ''')    
        
        conn.commit()
        conn.close()
        
    def validate_coordinates(self, lat_str, lon_str):
        """Validacion de las coordenadas sean válidas para Panamá."""
        try:
            if not lat_str.strip() or not lon_str.strip():
                return False, "Las coordenadas no pueden estar vacías."
            
            # Removiendo espacios y validar formato
            lat_str = lat_str.strip()
            lon_str = lon_str.strip()
            
            # Validacion para que solo contengan números, puntos y signos negativos
            if not re.match(r'^-?\d+\.?\d*$', lat_str) or not re.match(r'^-?\d+\.?\d*$', lon_str):
                return False, "Las coordenadas deben ser números válidos."
            
            lat = float(lat_str)
            lon = float(lon_str)
            
            # Validación de rangos para Panamá
            # Panamá está entre 7°-10° N y 77°-83° W
            if not (7.0 <= lat <= 10.0):
                return False, f"Latitud fuera del rango de Panamá (7° - 10° N). Valor ingresado: {lat}°"
            
            if not (-83.0 <= lon <= -77.0):
                return False, f"Longitud fuera del rango de Panamá (77° - 83° W). Valor ingresado: {lon}°"
            
            return True, None
            
        except ValueError:
            return False, "Las coordenadas deben ser números válidos."
    
    def select_image(self):
        """Permite seleccionar una imagen."""    
        file_path = filedialog.askopenfilename(
            title="Seleccione una imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.current_image_path = file_path
            # Mostrar la imagen en el interfaz
            self.display_image(file_path)
            # Habilitar el botón de detección
            self.btn_detect.config(state=tk.NORMAL)
            # Deshabilitar actualizar mapa hasta que haya detección
            self.btn_update_map.config(state=tk.DISABLED)
            # Restablecer el resultado de detección
            self.result_label.config(text="Resultado de Detección: No analizado")
            self.detection_result = None
            self.btn_save_sighting.config(state=tk.DISABLED)
            self.saved_image_path = None
       
    def display_image(self, image_path):
        """Muestra la imagen seleccionada en la interfaz."""
        try:
            img = Image.open(image_path)
            img = img.resize((200, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen: {str(e)}")
    
    def detect_iguana(self):
        """Detecta si hay una iguana en la imagen seleccionada."""
        if not self.current_image_path:
            return messagebox.showerror("Error", "Por favor, seleccione una imagen primero.")
        
        try:
            # Carga la imagen
            image = cv2.imread(self.current_image_path)
            if image is None:
                messagebox.showerror("Error", "No se pudo cargar la imagen.")
                return
            
            # Predicción de YOLOv8
            results = self.model(image)
            
            # Procesar resultados
            detections = []
            max_confidence = 0.0
            total_detections = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        detections.append({
                            'confidence': confidence,
                            'class_id': class_id,
                            'bbox': box.xyxy[0].tolist()
                        })
                        
                        if confidence > max_confidence:
                            max_confidence = confidence
                        total_detections += 1
            
            # Mostrar resultados
            if detections:
                best_detection = max(detections, key=lambda x: x['confidence'])
                confidence_percentage = best_detection['confidence'] * 100
                
                result_text = f"Resultado: Iguana detectada con {confidence_percentage:.1f}% de confianza."
                if total_detections > 1:
                    result_text += f" Total de detecciones: {total_detections}."
                    
                self.result_label.config(text=result_text, fg="green")
                self.detection_result = {
                    'is_iguana': True,
                    'confidence': best_detection['confidence'],
                    'detections_count': total_detections,
                    'all_detections': detections
                }
                
                # Habilitar botones después de detección exitosa
                self.btn_update_map.config(state=tk.NORMAL)
                
                # El botón de guardar se habilitará después de ingresar coordenadas válidas
                
                # Mostrar imagen con bounding boxes
                self.display_image_with_detections(image, detections)
            else:
                result_text = "Resultado: No se detectaron iguanas."
                self.result_label.config(text=result_text, fg="red")
                self.detection_result = {
                    'is_iguana': False,
                    'confidence': 0.0,
                    'detections_count': 0,
                    'all_detections': []
                }
                # No se habilitan botones si no hay detección
                self.btn_update_map.config(state=tk.DISABLED)
                self.btn_save_sighting.config(state=tk.DISABLED)
                
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo detectar iguanas: {str(e)}")
            print(f"Error al detectar iguanas: {str(e)}")
    
    def display_image_with_detections(self, cv_image, detections):
        """Muestra la imagen con las detecciones de iguanas marcadas."""
        try:
            # Conversion de BGR a RGB
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Crear objeto para dibujar
            draw = ImageDraw.Draw(pil_image)
            
            # Dibujar bounding boxes
            for detection in detections:
                bbox = detection['bbox']
                confidence = detection['confidence']
                
                # Coordenadas del bounding box
                x1, y1, x2, y2 = map(int, bbox)
                
                # Dibujo del rectángulo
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                # Dibujo del texto con confianza
                text = f"Iguana {confidence * 100:.1f}%"
                draw.text((x1, y1 - 20), text, fill="red")
                
            # Redimensionar y mostrar
            pil_image = pil_image.resize((200, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo mostrar la imagen con detecciones: {str(e)}")
            print(f"Error detallado: {e}")
            # Si hay error, mostrar imagen original
            self.display_image(self.current_image_path)
    
    def save_image_for_sighting(self):
        """Guarda una copia de la imagen para el avistamiento."""
        try:
            if not self.current_image_path:
                return None
            
            # Generar nombre único para la imagen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            file_extension = os.path.splitext(self.current_image_path)[1]
            new_filename = f"iguana_{timestamp}_{unique_id}{file_extension}"
            
            # Ruta completa para guardar
            saved_path = os.path.join(self.saved_images_dir, new_filename)
            
            # Copiar la imagen
            shutil.copy2(self.current_image_path, saved_path)
            
            return saved_path
            
        except Exception as e:
            print(f"Error al guardar imagen: {e}")
            return None
    
    def update_map(self):
        """Actualiza el mapa con las coordenadas ingresadas y lo abre en el navegador."""
        # Validar que hay una detección exitosa
        if not self.detection_result or not self.detection_result['is_iguana']:
            messagebox.showwarning("Advertencia", "Primero debe detectar una iguana en la imagen.")
            return
        
        try:    
            lat_str = self.lat_entry.get()
            lon_str = self.lon_entry.get()
            
            # Validar coordenadas
            is_valid, error_msg = self.validate_coordinates(lat_str, lon_str)
            if not is_valid:
                return messagebox.showerror("Error", error_msg)
            
            lat = float(lat_str.strip())
            lon = float(lon_str.strip())
            self.location_coords = (lat, lon)
            
            # Habilitar el botón de guardar avistamiento
            self.btn_save_sighting.config(state=tk.NORMAL)
            
            # Crear mapa interactivo centrado en el punto dado
            m = self.create_interactive_map([lat, lon], zoom_start=15)
            
            # Guardar imagen para mostrar en el popup
            temp_image_path = self.save_image_for_sighting()
            
            # Crear contenido del popup
            popup_html = f"""
            <div style='width: 250px;'>
                <b>🦎 Ubicación de avistamiento</b><br>
                <b>Coordenadas:</b><br>
                Lat: {lat}<br>
                Lng: {lon}<br>
                <b>Detección:</b><br>
                Confianza: {self.detection_result['confidence']*100:.1f}%<br>
                Cantidad: {self.detection_result['detections_count']}<br>
            </div>
            """
            
            if temp_image_path and os.path.exists(temp_image_path):
                # Convertir ruta a URL file://
                file_url = temp_image_path.replace("\\", "/")
                popup_html += f'<br><img src="file:///{file_url}" width="200" height="150" style="border-radius: 5px;">'

            popup = folium.Popup(popup_html, max_width=270)
            
            # Añadir marcador en la ubicación dada
            folium.Marker(
                location=[lat, lon],
                popup=popup,
                icon=folium.Icon(color="green", icon="leaf", prefix='fa')
            ).add_to(m)
            
            # Guardar el mapa temporalmente
            temp_map = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            m.save(temp_map.name)
            
            # Abrir el mapa en el navegador
            webbrowser.open('file://' + temp_map.name, new=2)
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo actualizar el mapa: {str(e)}")
    
    def create_interactive_map(self, center_location, zoom_start=10):
        """Crea un mapa interactivo con funcionalidades adicionales incluyendo popup de coordenadas."""
        # Crear el mapa base
        m = folium.Map(location=center_location, zoom_start=zoom_start)
        
        # JavaScript para capturar clicks
        click_script = """
        <script>
        function onMapClick(e) {
            var lat = e.latlng.lat.toFixed(6);
            var lng = e.latlng.lng.toFixed(6);
            
            // Crear popup con las coordenadas
            var popup = L.popup()
                .setLatLng(e.latlng)
                .setContent('<b>Coordenadas:</b><br>Latitud: ' + lat + '<br>Longitud: ' + lng)
                .openOn(this);
            
            console.log('Latitud: ' + lat + ', Longitud: ' + lng);
        }
        
        // Esperar a que el mapa se cargue
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                var mapId = Object.keys(window).find(key => key.startsWith('map_'));
                if (mapId && window[mapId]) {
                    window[mapId].on('click', onMapClick);
                }
            }, 100);
        });
        </script>
        """
    
        # Agregar el script al mapa
        m.get_root().html.add_child(folium.Element(click_script))
        
        return m
    
    def save_sighting(self):
        """Versión modificada que incluye limpieza de imágenes."""
        if not self.current_image_path or not self.detection_result or not self.location_coords:
            messagebox.showwarning("Advertencia", "Información incompleta para guardar el avistamiento.")
            return
        
        if not self.detection_result['is_iguana']:
            messagebox.showwarning("Advertencia", "No se detectó una iguana en la imagen.")
            return

        try:
            lat, lon = self.location_coords
            
            # Validacion coordenadas
            is_valid, error_msg = self.validate_coordinates(str(lat), str(lon))
            if not is_valid:
                messagebox.showerror("Error", f"Coordenadas inválidas: {error_msg}")
                return
            
            # Guardado de copia de la imagen
            saved_image_path = self.save_image_for_sighting()
            if not saved_image_path:
                messagebox.showerror("Error", "No se pudo guardar la imagen del avistamiento.")
                return
            
            # Conectar a la base de datos y guardar
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO sightings (latitude, longitude, original_image_path, saved_image_path, 
                                detection_confidence, detections_count, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                lat, lon, self.current_image_path, saved_image_path,
                self.detection_result['confidence'],
                self.detection_result['detections_count'],
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            # Pregunta para eliminar imagen original
            if self.ask_delete_original_image():
                self.cleanup_original_image()
            
            messagebox.showinfo("✅ Éxito", 
                            f"Avistamiento guardado correctamente!\n\n"
                            f"Ubicación: {lat:.6f}, {lon:.6f}\n"
                            f"Confianza: {self.detection_result['confidence']*100:.1f}%\n"
                            f"Detecciones: {self.detection_result['detections_count']}")
            
            # Limpiaeza de formulario
            self.reset_form()
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar el avistamiento: {str(e)}")
            print(f"Error al guardar el avistamiento: {str(e)}")

    def reset_form(self):
        """Limpia el formulario después de guardar un avistamiento."""
        self.current_image_path = None
        self.location_coords = None
        self.detection_result = None
        self.saved_image_path = None
        
        # Limpiar entradas
        self.lat_entry.delete(0, tk.END)
        self.lon_entry.delete(0, tk.END)
        
        # Restablecer botones
        self.btn_detect.config(state=tk.DISABLED)
        self.btn_update_map.config(state=tk.DISABLED)
        self.btn_save_sighting.config(state=tk.DISABLED)
        
        # Restablecer etiquetas
        self.result_label.config(text="Resultado de Detección: No analizado", fg="white")
        
        # Mostrar imagen por defecto
        if os.path.exists(self.default_image_path):
            self.display_image(self.default_image_path)
        else:
            self.image_label.config(image="", text="Seleccione una imagen", fg="white")
            
    def show_all_sightings(self):
        """Muestra todos los avistamientos guardados en el mapa."""
        try:
            # Conectar a la base de datos
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Obtener todos los avistamientos
            cursor.execute("""
                SELECT latitude, longitude, timestamp, detection_confidence, 
                       detections_count, saved_image_path 
                FROM sightings 
                ORDER BY timestamp DESC
            """)
            sightings = cursor.fetchall()
            
            conn.close()
            
            if not sightings:
                messagebox.showinfo("Información", "No hay avistamientos guardados todavía.")
                return
            
            # Crear un mapa interactivo centrado en Panamá
            m = self.create_interactive_map([8.9943, -79.5188], zoom_start=8)
            
            # Añadir marcadores para todos los avistamientos
            for i, (lat, lon, timestamp, confidence, count, image_path) in enumerate(sightings):
                # Formatear fecha
                try:
                    dt = datetime.fromisoformat(timestamp.replace('T', ' '))
                    formatted_date = dt.strftime("%d/%m/%Y %H:%M")
                except:
                    formatted_date = timestamp.split('T')[0]
                
                # Crear contenido del popup
                popup_html = f"""
                <div style='width: 280px; text-align: center;'>
                    <h4 style='margin: 5px 0; color: #2E7D32;'>🦎 Avistamiento #{i+1}</h4>
                    <hr style='margin: 5px 0;'>
                    <table style='width: 100%; font-size: 12px;'>
                        <tr><td><b>📅Fecha:</b></td><td>{formatted_date}</td></tr>
                        <tr><td><b>📍 Coordenadas:</b></td><td>{lat:.6f}, {lon:.6f}</td></tr>
                        <tr><td><b>🎯 Confianza:</b></td><td>{confidence*100:.1f}%</td></tr>
                        <tr><td><b>🔢 Cantidad:</b></td><td>{count}</td></tr>
                    </table>
                """
                
                # Agregar imagen si existe
                if image_path and os.path.exists(image_path):
                    # Convertir ruta a URL file://
                    file_url = image_path.replace("\\", "/")
                    popup_html += f"""
                    <hr style='margin: 10px 0;'>
                    <img src="file:///{file_url}" 
                         width="240" height="180" 
                         style="border-radius: 8px; border: 2px solid #4CAF50;">
                    """
                else:
                    popup_html += "<br><i>🚫 Imagen no disponible</i>"
                
                popup_html += "</div>"
                
                # Determinar color del marcador basado en confianza
                if confidence >= 0.8:
                    marker_color = "green"
                    icon_name = "leaf"
                elif confidence >= 0.6:
                    marker_color = "orange"
                    icon_name = "exclamation-triangle"
                else:
                    marker_color = "red"
                    icon_name = "question"
                
                # Crear popup
                popup = folium.Popup(popup_html, max_width=300)
                
                # Añadir marcador
                folium.Marker(
                    location=[lat, lon],
                    popup=popup,
                    tooltip=f"Avistamiento {formatted_date} - {confidence*100:.1f}%",
                    icon=folium.Icon(color=marker_color, icon=icon_name, prefix='fa')
                ).add_to(m)
            
            # Agregar información estadística
            total_sightings = len(sightings)
            avg_confidence = sum(s[3] for s in sightings) / total_sightings
            total_iguanas = sum(s[4] for s in sightings)
            
            stats_html = f"""
            <div style='position: fixed; 
                        top: 10px; left: 10px; 
                        background: rgba(255,255,255,0.9); 
                        padding: 10px; 
                        border-radius: 8px; 
                        border: 2px solid #4CAF50;
                        font-family: Arial;
                        z-index: 1000;'>
                <h4 style='margin: 0 0 10px 0; color: #2E7D32;'>📊 Estadísticas</h4>
                <div style='font-size: 14px;'>
                    <div>🏷️ <b>Total avistamientos:</b> {total_sightings}</div>
                    <div>🦎 <b>Total iguanas:</b> {total_iguanas}</div>
                    <div>📈 <b>Confianza promedio:</b> {avg_confidence*100:.1f}%</div>
                </div>
            </div>
            """
            
            m.get_root().html.add_child(folium.Element(stats_html))
            
            # Guardar mapa como HTML temporal
            temp_map = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
            m.save(temp_map.name)
            
            # Abre el mapa en el navegador predeterminado
            webbrowser.open('file://' + temp_map.name, new=2)
            
            print(f"Mapa generado con {total_sightings} avistamientos")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar avistamientos: {str(e)}")
            print(f"Error detallado: {e}")

    def ask_delete_original_image(self):
        """Pregunta al usuario si desea eliminar la imagen original después de guardar."""
        try:
            result = messagebox.askyesno(
                "Limpieza de Archivos",
                "¿Desea eliminar la imagen original del dispositivo?\n\n"
                "La imagen ya ha sido guardada de forma segura\n"
                "Una copia permanece en la carpeta de avistamientos\n"
                "Esto ayuda a mantener limpio su dispositivo\n\n"
                "¿Continuar con la eliminación?",
                icon='question'
            )
            return result
        except Exception as e:
            print(f"Error al preguntar sobre eliminación: {e}")
            return False

    def cleanup_original_image(self):
        """Elimina la imagen original de forma segura."""
        try:
            if self.current_image_path and os.path.exists(self.current_image_path):
                # Verificacion que no sea la imagen por defecto
                if self.current_image_path == self.default_image_path:
                    print("No se eliminará la imagen por defecto")
                    return
                
                # Verificacion que la imagen no esté en la carpeta de avistamientos guardados
                if self.saved_images_dir in self.current_image_path:
                    print("No se eliminará imagen ya guardada en avistamientos")
                    return
                
                # Eliminacion de imagen original
                os.remove(self.current_image_path)
                print(f"Imagen original eliminada: {self.current_image_path}")
                
                # Muestra confirmación discreta
                self.show_cleanup_notification()
                
        except Exception as e:
            print(f"Error al eliminar imagen original: {e}")
            # No mostrar error al usuario para no interrumpir el flujo

    def show_cleanup_notification(self):
        """Muestra una notificación discreta de limpieza exitosa."""
        try:
            # Ventana de notificación temporal
            notification = tk.Toplevel(self.root)
            notification.title("Limpieza Completada")
            notification.geometry("300x100")
            notification.configure(bg="#4CAF50")
            notification.resizable(False, False)
            
            # Centrar la ventana
            notification.geometry("+{}+{}".format(
                self.root.winfo_rootx() + 250,
                self.root.winfo_rooty() + 200
            ))
            
            # Mensaje de confirmación
            message_label = tk.Label(
                notification,
                text="🧹 Imagen original eliminada\nEspacio liberado exitosamente",
                font=("Arial", 10, "bold"),
                fg="white",
                bg="#4CAF50",
                pady=20
            )
            message_label.pack(expand=True, fill=tk.BOTH)
            
            # Auto-cerrar después de 2 segundos
            notification.after(2000, notification.destroy)
            
        except Exception as e:
            print(f"Error al mostrar notificación: {e}")

    # Gestión de archivos
    def manage_saved_images(self):
        """Función para gestionar imágenes guardadas (opcional)."""
        try:
            # Obtencion información de archivos guardados
            saved_files = []
            if os.path.exists(self.saved_images_dir):
                for filename in os.listdir(self.saved_images_dir):
                    filepath = os.path.join(self.saved_images_dir, filename)
                    if os.path.isfile(filepath):
                        file_size = os.path.getsize(filepath)
                        file_time = os.path.getmtime(filepath)
                        saved_files.append({
                            'name': filename,
                            'path': filepath,
                            'size': file_size,
                            'modified': datetime.fromtimestamp(file_time)
                        })
            
            total_size = sum(f['size'] for f in saved_files)
            total_count = len(saved_files)
            
            info_message = f"Gestión de Imágenes Guardadas\n\n"
            info_message += f"Total de archivos: {total_count}\n"
            info_message += f"Espacio ocupado: {total_size / (1024*1024):.2f} MB\n"
            info_message += f"Ubicación: {self.saved_images_dir}\n\n"
            info_message += "Las imágenes se eliminan automáticamente\n"
            info_message += "después de guardar cada avistamiento (opcional)"
            
            messagebox.showinfo("Gestión de Archivos", info_message)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al gestionar imágenes: {str(e)}")

def main():
    """Función principal para ejecutar la aplicación."""
    try:
        root = tk.Tk()
        app = IguanaSightingsApp(root)
        
        messagebox.showinfo("Sugerencia", f"Se recomiendan usar 4 decimales para latitud y longitud")
        
        root.mainloop()
        
    except Exception as e:
        print(f"Error crítico al iniciar la aplicación: {e}")
        messagebox.showerror("Error Crítico", f"No se pudo iniciar la aplicación: {str(e)}")
        
if __name__ == "__main__":
    main()