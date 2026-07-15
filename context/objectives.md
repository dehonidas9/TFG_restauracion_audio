# Objetivos del TFG

**Título:** Análisis y aplicación de modelos de aprendizaje profundo para la restauración interactiva de señales de audio degradadas.

El presente Trabajo de Fin de Grado se plantea los siguientes objetivos:

1. Realizar una revisión del estado del arte en restauración de señales de audio mediante técnicas de aprendizaje profundo, cubriendo las cinco categorías de degradación consideradas: eliminación de ruido (*denoising*), eliminación de reverberación (*dereverberation*), super-resolución o ampliación de ancho de banda (*BWE*), corrección de recorte (*de-clipping*) y separación de fuentes.

2. Analizar los fundamentos teóricos y matemáticos de los modelos seleccionados para cada una de las categorías anteriores, así como de las técnicas clásicas (no basadas en IA) empleadas como referencia comparativa.

3. Seleccionar e integrar modelos preentrenados representativos de cada categoría, implementando un flujo de inferencia reproducible sobre Google Colab sin necesidad de reentrenamiento.

4. Definir y calcular un conjunto de métricas objetivas y perceptuales para la evaluación de los modelos, distinguiendo entre un bloque bibliográfico (métricas ya publicadas, como PESQ, STOI, SI-SDR y LSD) y un bloque empírico (métricas no intrusivas, como DNSMOS y NISQA, aplicadas sobre audio real sin referencia limpia).

5. Realizar un análisis comparativo entre modelos de inteligencia artificial y técnicas clásicas (baselines no-IA) para cada categoría de degradación.

6. Realizar un análisis comparativo entre modelos especializados (un modelo por tarea) y un modelo combinado multi-tarea (ClearVoice/MossFormer2), evaluando ventajas e inconvenientes de cada enfoque.

7. Extraer conclusiones sobre la viabilidad, limitaciones y aplicabilidad práctica de los modelos estudiados, proponiendo líneas de trabajo futuro.

8. Desarrollar una aplicación interactiva mediante Gradio, con una interfaz de dos niveles (categoría de degradación → modelo o baseline), que permita cargar audio, aplicar el modelo seleccionado y visualizar los resultados (audio antes/después, espectrogramas comparativos y métricas), desplegada finalmente como Hugging Face Space.
