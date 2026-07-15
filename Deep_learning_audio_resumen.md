# Notas de estudio — Deep Learning para Audio
*Fuente base: Purwins, Li, Virtanen, Schlüter, Chang & Sainath (2019), "Deep Learning for Audio Signal Processing", IEEE JSTSP. arXiv:1905.00078*
*Complementado con dudas resueltas en conversación aparte (Gemini)*

> Nota: esto es una síntesis de estudio personal, NO texto para la memoria. Sirve para tener claros los conceptos antes de redactar.

---

## 1. Cómo se categorizan las tareas de audio

Dos ejes independientes:
- **Cuántas etiquetas hay que predecir**: una global (clasificación de secuencia, ej. idioma/hablante), una por paso temporal (sequence labeling, ej. detección de actividad vocal), o una secuencia libre no ligada a la longitud de entrada (sequence transduction, ej. speech-to-text).
- **Qué tipo de etiqueta es**: clase única, conjunto de clases (multi-label), o valor numérico continuo (regresión).

Para mis tareas (denoising, dereverb, SR, de-clipping, separación): son básicamente **regresión por paso temporal** — la salida es una secuencia de igual "resolución" que la entrada (un espectrograma limpio del mismo tamaño que el sucio), no una clasificación.

---

## 2. Representaciones de entrada (features)

- **MFCC** (clásico, pre-deep learning): banco de filtros Mel → log → **DCT** → te quedas con los primeros 12-13 coeficientes.
- **Log-mel spectrogram**: es la representación dominante en deep learning actual. Es básicamente el paso anterior a la DCT del pipeline MFCC — se descubrió que aplicar la DCT en redes profundas es innecesario/contraproducente porque **destruye la relación espacial** entre bandas de frecuencia vecinas, algo que a una CNN le interesa conservar.
- **Raw waveform**: usado directamente sin transformar (ej. WaveNet), evita features "diseñadas a mano" pero exige más datos/cómputo.
- **Constant-Q spectrum**: alternativa al log-mel con escala logarítmica de frecuencia, útil cuando interesa que una transposición de tono se vea como un simple desplazamiento (más música que voz).

**Dato clave para mis notebooks/app**: los espectrogramas NO se pueden normalizar globalmente como una imagen (media/std de toda la "imagen"). Las frecuencias bajas concentran muchísima más energía que las altas de forma естructural (voz/música), así que una normalización global aplasta las altas frecuencias a casi cero (vanishing gradients) y el modelo deja de poder distinguir consonantes sibilantes, platillos, etc. **Solución: estandarizar banda a banda** (media y std propias por cada fila de frecuencia).

### Por qué la DCT tiene sentido en MFCC (contexto, aunque en DL ya no se use tanto)
- **Compactación de energía**: casi toda la información útil cae en los primeros coeficientes.
- **Solo números reales** (a diferencia de la DFT, que da complejos) → más barato computacionalmente.
- **Descorrelaciona**: los filtros Mel se solapan mucho entre sí (correlación alta entre bandas vecinas); la DCT actúa como aproximación a la transformada óptima de descorrelación (KLT), dejando coeficientes ~independientes.
- Truco de simetría (reflejo especular de la ventana) evita las discontinuidades artificiales de borde que sí introduce la DFT/FFT.

---

## 3. Modelos

### CNNs
- Convolucionan la entrada con kernels aprendibles (no diseñados a mano, a diferencia del procesado clásico tipo Sobel).
- **Campo receptivo**: cuánto "ve" de contexto una neurona profunda. Fijo por arquitectura; crece con kernels más grandes o más capas apiladas.
- Estructura típica: capas convolucionales + pooling intercaladas, luego capas densas al final (para clasificación).
- **Fully-Convolutional Network (FCN)**: se eliminan las capas densas finales. Imprescindible para mis tareas, porque necesito que la salida conserve la estructura temporal/espectral exacta de la entrada (espectrograma limpio del mismo tamaño que el sucio) — las capas densas destruirían esa estructura y además fijarían un tamaño de entrada constante.

**Kernels y mapas de características — cómo se conectan (aclarado con Gemini):**
- Un kernel es una matriz pequeña de pesos que se desliza sobre la entrada; cada kernel es como una "perspectiva" distinta desde la que buscar un patrón (bordes, armónicos, transitorios...).
- El resultado de deslizar un kernel es un **mapa de características (feature map)**: una "imagen" 2D que se ilumina donde se ha encontrado ese patrón.
- Si usas N kernels en una capa, obtienes N mapas 2D que se apilan → de ahí sale la **tercera dimensión** de los mapas de características intermedios: no es una coordenada espacial, es el número de canal/filtro (qué característica se detectó, no dónde).
- Jerarquía típica: capas iniciales detectan patrones simples (bordes, transitorios, tonos constantes) → capas intermedias combinan esos patrones en conceptos más complejos (armónicos, subidas de tono) → capas profundas detectan conceptos abstractos (un fonema, el patrón de cola de una reverberación).

**Pooling vs. capas convolucionales siguientes — quién combina qué (aclarado con Gemini, este era mi punto de confusión):**
- El **pooling** actúa canal por canal, de forma independiente: solo reduce el tamaño espacial/temporal de cada mapa (ej. max-pooling 2×2), pero NUNCA mezcla información entre canales distintos.
- Quien realmente **combina/mezcla las distintas perspectivas (canales)** es el kernel de la SIGUIENTE capa convolucional, que ya no es 2D sino 3D (ancho × alto × nº de canales de la capa anterior) — "mira" todos los canales a la vez y los funde en un solo valor de salida.
- El número de dimensiones del mapa de características (3D: frecuencia, tiempo, canales) se mantiene constante capa a capa; lo que cambia es el tamaño de cada dimensión (frecuencia/tiempo se reducen con pooling, canales suelen aumentar).

**Convoluciones dilatadas (atrous):**
- Problema que resuelven: con audio raw waveform de alta tasa de muestreo, alcanzar un campo receptivo suficiente con convoluciones normales exigiría demasiadas capas o kernels enormes (explosión de parámetros).
- Solución: insertar "huecos" (ceros) entre los coeficientes del kernel, controlados por un factor de dilatación. Apilando capas con dilatación creciente exponencialmente (1, 2, 4, 8...) se consigue un campo receptivo enorme con pocas capas, sin perder resolución (a diferencia del pooling, que si reduce resolución) y sin disparar el nº de parámetros. Es la base de WaveNet y arquitecturas similares (relevante para HTDemucs/Conv-TasNet-like).

### RNNs / LSTM
- Procesan la secuencia paso a paso, manteniendo un estado oculto — permiten (en teoría) contexto indefinido hacia atrás, y bidireccionales también hacia delante (offline).
- LSTM soluciona el problema de gradientes que desaparecen/explotan de las RNN simples.

**F-LSTM / TF-LSTM (variantes específicas de audio, aclaradas con Gemini):**
- **F-LSTM**: en vez de recorrer solo el eje temporal, recorre el eje de **frecuencia**. Logra invarianza traslacional (equivalente a lo que logra un kernel de CNN al deslizarse) pero de forma secuencial/recurrente. No necesita pooling.
- **TF-LSTM**: se despliega en AMBOS ejes (tiempo y frecuencia) simultáneamente — cada celda recibe memoria tanto de la banda de frecuencia anterior como del paso temporal anterior. Modela mejor tanto la evolución temporal como la distribución espectral.
- **Trade-off clave**: TF-LSTM puede superar a las CNN en ciertas tareas, pero es mucho menos paralelizable (cómputo estrictamente secuencial) → más lento en GPU. Las CNN (sobre todo con dilatadas) ganan en la práctica cuando importa la velocidad/tiempo real.

### Otros modelos mencionados
- **Seq2seq / atención / CTC**: relevantes sobre todo para ASR (reconocimiento), no tanto para mis tareas de restauración.
- **GANs**: uso limitado en audio comparado con imagen, pero sí se han aplicado a separación de fuentes, transformación de timbre y **speech enhancement** (SEGAN es el ejemplo citado en el paper — interesante para mi estado del arte de denoising).
- **Funciones de pérdida**: el MSE simple en dominio temporal no es robusto (dos señales con misma frecuencia pero fase distinta ya dan error alto aunque suenen igual). Se usan MSE sobre log-mel, dynamic time warping, o funciones de pérdida diseñadas específicamente en base a métricas de inteligibilidad perceptual (conexión directa con el bloque de métricas: STOI se ha usado literalmente como objetivo de entrenamiento, no solo como métrica de evaluación).
- **Modelado de fase**: el espectro de magnitud pierde la fase; para síntesis hace falta reconstruirla (Griffin-Lim) o trabajar directamente con espectro complejo / redes que ingieren magnitud+fase o targets complejos.

---

## 4. Separación de fuentes — formulación formal (relevante para HTDemucs)

- Mezcla: $x_m(n) = \sum_i s_{m,i}(n)$ — la señal en el micrófono m es la suma de las fuentes i.
- Métodos estado del arte: enmascaramiento en el dominio tiempo-frecuencia. La máscara $M_{m,i}(f,t)$ se multiplica por el espectro de la mezcla para estimar el espectro de la fuente separada.
- Por qué tiempo-frecuencia y no dominio temporal directo: 1) la estructura de las fuentes naturales es más prominente ahí, 2) el mezclado convolutivo se aproxima a mezclado instantáneo en frecuencia (simplifica), 3) las fuentes son dispersas (sparse) en tiempo-frecuencia, facilitando separarlas.
- Métricas de evaluación mencionadas explícitamente en el paper: **SDR, SIR (signal-to-interference), SAR (signal-to-artifacts)** — coincide con el bloque de métricas que ya tengo trabajado.
- Con múltiples micrófonos se puede incorporar información espacial (relevante si en algún momento comparo mono vs. estéreo, dado que el audio de mi tutor es estéreo).

---

## 5. Audio enhancement (denoising/dereverb) — contexto histórico

- Métodos clásicos previos a deep learning: **Wiener filtering, non-negative matrix factorization (NMF)** — el paper señala explícitamente que el deep learning "ha resuelto tareas antes abordadas por NMF y métodos de Wiener", frase que me sirve literalmente para justificar por qué elegí baselines clásicos de este tipo en mi TFG.
- Arquitecturas usadas: autoencoders de denoising, redes convolucionales, redes recurrentes, y GANs (SEGAN).
- Los métodos convencionales (Wiener) suelen asumir ruido estacionario; el deep learning puede modelar ruido variable en el tiempo — buen argumento para justificar por qué mis modelos de IA superan al baseline clásico en escenarios reales no estacionarios (como el audio de mi tutor).

---

## 6. Capas bajas vs. capas altas — qué aprende cada una (aclarado con Gemini)

Frase original del paper: las activaciones de las capas bajas de una DNN de ASR se pueden interpretar como *features adaptadas al hablante*, mientras que las capas altas hacen *discriminación basada en clases*.

- **Capas bajas**: capturan el "cómo" suena — timbre, tono, acento, velocidad, acústica de la sala. Actúan normalizando/neutralizando las particularidades físicas de quién habla.
- **Capas altas**: una vez neutralizado el "quién", se concentran en el "qué" se dice (fonemas) — son invariantes a si la voz es aguda o grave, lo que importa es identificar el contenido lingüístico.
- Patrón general en deep learning: capas iniciales aprenden características físicas/genéricas, capas finales se vuelven abstractas y task-specific.

---

## 7. Preguntas abiertas que señala el propio paper (útiles para "trabajo futuro" de mi TFG)

- No hay consenso sobre qué representación de entrada es óptima (log-mel vs. constant-Q vs. raw waveform) ni qué arquitectura es superior (CNN vs. RNN vs. CRNN) — depende mucho de la tarea concreta.
- No existe un equivalente a ImageNet para audio (dataset masivo para transfer learning generalizable entre dominios de audio) — relevante para justificar por qué mi TFG usa modelos preentrenados específicos por tarea en vez de un único modelo fundacional.
- Interpretabilidad de las redes sigue siendo un problema abierto.

---

## Resumen de conexión directa con mi TFG

| Concepto del paper | Dónde lo uso en mi TFG |
|---|---|
| FCN sin capas densas | Justifica por qué los modelos de restauración (DeepFilterNet, MP-SENet, etc.) son arquitecturas totalmente convolucionales/recurrentes, no clasificadores |
| Convoluciones dilatadas | Contexto arquitectónico para HTDemucs y modelos basados en waveform |
| F-LSTM / TF-LSTM y su lentitud | Argumento de trade-off velocidad/calidad al comparar arquitecturas en el estado del arte |
| Estandarización por banda | Detalle técnico de preprocesamiento relevante si explico el pipeline interno de algún modelo |
| SDR/SIR/SAR como métricas de separación | Conecta directamente con mi bloque de métricas ya trabajado (Bloque 1 de la conversación anterior) |
| DL "reemplazó" a Wiener/NMF | Frase-argumento lista para justificar la elección de baselines clásicos por categoría |
| Sin "ImageNet del audio" | Argumento para el porqué de usar 1-2 modelos preentrenados específicos por tarea en vez de un único modelo fundacional |
