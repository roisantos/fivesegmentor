# Análisis Técnico de las Arquitecturas de Segmentación en FiveSegmentor

## Estructura del Proyecto
```
fivesegmentor/code/
├── models/         # Arquitecturas de redes neuronales
├── utils/          # Utilidades y funciones auxiliares
├── training/       # Scripts de entrenamiento
├── evaluation/     # Métricas y evaluación
├── ds/            # Manejo de datasets
├── config/        # Configuraciones
├── inference/      # Scripts de inferencia
└── scripts/        # Scripts de ejecución
```

## 1. Descripción Técnica de FRNet (Full Resolution Network)

### 1.1 Fundamentos y Diseño Conceptual

FRNet (Full Resolution Network) es una arquitectura neuronal diseñada específicamente para la segmentación de imágenes médicas, con un enfoque en el procesamiento de características manteniendo la resolución espacial completa durante todo el flujo de datos. Esta arquitectura se propone para abordar:

1. **El problema de preservación de estructuras finas**: Los vasos sanguíneos son estructuras delgadas que podrían perderse con reducciones de resolución.
2. **La necesidad de procesamiento eficiente**: Capacidad para trabajar con imágenes de alta resolución (2048×2048 píxeles) manteniendo un uso razonable de recursos.

FRNet se origina como una adaptación inspirada en la arquitectura ResNet, modificada específicamente para tareas de segmentación vascular. La idea central es mantener la resolución espacial completa, evitando las operaciones de downsampling y upsampling típicas de las arquitecturas encoder-decoder.

### 1.2 Implementación Técnica

Nuestra implementación de FRNet presenta la siguiente estructura:

```python
class FRNet(nn.Module):
    def __init__(self, ch_in, ch_out, ls_mid_ch=([32]*6), out_k_size=11, k_size=3,
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
```

La arquitectura se caracteriza por:

- **Estructura secuencial de bloques residuales**: Cadena de bloques convolucionales que mantienen la resolución constante.
- **Kernel final amplio**: Utiliza un kernel de salida de 11×11 para capturar mayor contexto en la decisión final.
- **Sistema modular**: Permite intercambiar diferentes tipos de bloques convolucionales según las necesidades.

En el método `forward`, la red procesa secuencialmente la entrada:

```python
def forward(self, x):
    for i in range(len(self.ls_mid_ch)):
        conv = self.dict_module[f'conv{i}']
        x = conv(x)

    x = self.dict_module['final'](x)
    x = torch.max(x, dim=1, keepdim=True)[0]
    return x
```

### 1.3 Componentes Arquitectónicos Principales

#### 1.3.1 ResidualBlock como Unidad Básica

```python
class ResidualBlock(ConvBlock):
    def init(self, in_channels, out_channels, stride, k_size, dilation, layer_num=None):
        p = k_size//2 * dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=k_size,
                               stride=stride, padding=p, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=k_size,
                               stride=1, padding=p, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
```

Este bloque implementa:
- Doble capa convolucional con normalización por lotes
- Conexión residual para facilitar el flujo de gradientes
- Soporte para dilatación para aumentar el campo receptivo

#### 1.3.2 Capa Final Especializada

```python
self.dict_module.add_module(f"final", nn.Sequential(
    nn.Conv2d(ch1, ch_out*4, out_k_size, padding=out_k_size//2, bias=False),
    nn.Sigmoid()
))
```

Esta capa incorpora:
- Kernel grande para contexto local amplio
- Generación de múltiples mapas para mayor robustez
- Activación sigmoide para normalización
- Selección del máximo para resultado final

### 1.4 Efectos Esperados

Con esta arquitectura, se podrían anticipar los siguientes efectos:

1. **Preservación de detalles finos**:
   - Se espera una mejor conservación de vasos capilares delgados
   - Potencial mejora en la continuidad de estructuras vasculares

2. **Comportamiento en entrenamiento**:
   - Posible convergencia más rápida por la simplicidad del flujo de datos
   - Aprovechamiento de batch sizes mayores al requerir menos memoria por imagen

3. **Rendimiento en resolución alta**:
   - Potencial ventaja al procesar imágenes de 2048×2048 píxeles
   - Posible mejor equilibrio entre precisión y eficiencia computacional

4. **Limitaciones posibles**:
   - Campo receptivo limitado que podría afectar la captura de contexto global
   - Posible dificultad para capturar relaciones de largo alcance

## 2. Descripción Técnica de RoiNet: Adaptación hacia Arquitectura Encoder-Decoder

### 2.1 Concepto e Implementación

RoiNet surge como una evolución arquitectónica de FRNet, incorporando elementos de la arquitectura U-Net para abordar algunas limitaciones identificadas en el diseño full-resolution. Esta adaptación busca:

1. **Mejorar la captura de contexto global**: Mediante procesamiento multiescala
2. **Mantener la capacidad de preservar detalles**: A través de skip connections
3. **Balancear resolución y contexto**: Combinando características de diferentes niveles

```python
class RoiNet(nn.Module):
    def __init__(self, ch_in, ch_out, ls_mid_ch=[32, 64, 128, 128, 64, 32],
                 k_size=9, cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
```

### 2.2 Modificaciones Arquitectónicas Clave

#### 2.2.1 Implementación de Encoder-Decoder

A diferencia de la estructura lineal de FRNet, RoiNet implementa un diseño de codificador-decodificador:

```python
# ---- Encoder ----
self.dict_module.add_module("conv0", cls_init_block(ch, ls_mid_ch[0], k_size=k_size))
self.dict_module.add_module("pool1", nn.Sequential(
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(ch, ch * 2, kernel_size=1)
))

# ---- Bottleneck ----
self.dict_module.add_module("bottle1", cls_conv_block(ch, ch, k_size=k_size))

# ---- Decoder ----
self.dict_module.add_module("up3", nn.Sequential(
    nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
))
```

#### 2.2.2 Skip Connections

Se implementan conexiones que vinculan el encoder con el decoder:

```python
# Forward pass con skip connections
skip1 = out1  # Guardamos salida del encoder
# ...
out3 = torch.cat([out3, skip1], dim=1)  # Fusionamos en el decoder
```

### 2.3 Flujo de Procesamiento

En la propagación forward, la información fluye:

```python
def forward(self, x):
    # Encoder path
    out0 = self.dict_module["conv0"](x)           # Resolución completa
    out1 = self.dict_module["pool1"](out1)        # Reducción de resolución
    # ...
    
    # Bottleneck
    bottle1 = self.dict_module["bottle1"](out2)
    # ...
    
    # Decoder path
    out3 = self.dict_module["up3"](out3)          # Aumento de resolución
    out3 = torch.cat([out3, skip1], dim=1)        # Fusión con skip connection
    # ...
    
    # Salida final
    final = self.dict_module["final"](out5)
    return final
```

### 2.4 Comparativa Técnica FRNet vs RoiNet

| Aspecto Técnico | FRNet | RoiNet |
|------------|-------|--------|
| **Patrón arquitectónico** | Lineal, sin cambios de resolución | Encoder-decoder con múltiples niveles |
| **Gestión de resolución** | Constante durante todo el procesamiento | Reducción y recuperación progresiva |
| **Estrategia de contexto** | Acumulativo a través de capas secuenciales | Multiresolución con campos receptivos amplios |
| **Transferencia de información** | Directa a través de la cadena de bloques | A través de skip connections entre niveles |
| **Estrategia de salida** | Múltiples candidatos con selección de máximo | Proyección directa a canales de salida |
| **Requisitos de memoria** | Proporcional a resolución de entrada | Variable según niveles de características |

### 2.5 Efectos Esperados en RoiNet

La arquitectura RoiNet podría mostrar los siguientes comportamientos:

1. **Captura de contexto**:
   - Posible mejora en la comprensión de estructuras vasculares completas
   - Potencial para detectar patrones de mayor escala

2. **Representación de características**:
   - Capacidad para combinar información a diferentes escalas
   - Potencial equilibrio entre detalle local y contexto global

3. **Comportamiento en el entrenamiento**:
   - Posible mayor necesidad de memoria por la estructura jerárquica
   - Potencial robustez a la inicialización de pesos

4. **Consideraciones de implementación**:
   - Mayor complejidad técnica en la implementación
   - Necesidad de ajustar hiperparámetros adicionales

## 3. Proceso de Entrenamiento y Evaluación

### 3.1 Configuración del Entrenamiento

El sistema de entrenamiento está diseñado para trabajar con el dataset FIVES en varios formatos:

- **Alta resolución (2048×2048)**: Para evaluar el rendimiento en condiciones óptimas
- **Resoluciones reducidas**: Para comparativas con métodos tradicionales

### 3.2 Preprocesamiento de Datos

```python
def preprocess_image_label(self, image, label):
    # Normalización y preparación
    image = image.astype("float32") / 255.0
    
    # Padding para dimensiones múltiplos de 32
    pad_x = (image.shape[1] // 32 + 1) * 32 - image.shape[1]
    pad_y = (image.shape[0] // 32 + 1) * 32 - image.shape[0]
    # ...
```

### 3.3 Estrategias de Augmentación

Para aumentar la robustez del entrenamiento, se implementan:

1. **Transformaciones geométricas**: Rotaciones, flips, escalado
2. **Deformaciones elásticas**: Para simular variabilidad vascular
3. **Modificaciones de intensidad**: Variaciones de brillo y contraste

### 3.4 Funciones de Pérdida Implementadas

El sistema soporta múltiples funciones de pérdida:

1. **DiceLoss**: Medida de superposición entre predicción y ground truth
2. **CompositeLoss**: Combinación ponderada de múltiples funciones
3. **Funciones especializadas**: Adaptadas para estructuras vasculares

### 3.5 Métricas de Evaluación

Para la evaluación del rendimiento, se medirán:

1. **Métricas de segmentación**: Dice, precisión, recall, F1-score
2. **Eficiencia computacional**: Tiempo de inferencia, uso de memoria
3. **Robustez**: Comportamiento ante variabilidad en los datos

## 4. Próximos Pasos y Experimentación

Para la evaluación empírica de las arquitecturas propuestas, se plantean los siguientes experimentos:

1. **Comparativa de rendimiento**: Evaluación del Dice score en diferentes resoluciones
2. **Análisis de eficiencia**: Medición de tiempo de inferencia y uso de memoria
3. **Ablación de componentes**: Evaluación del impacto de elementos arquitectónicos específicos
4. **Transferencia a otros datasets**: Verificación de generalización a otros conjuntos de datos

Estos experimentos proporcionarán evidencia empírica sobre las hipótesis planteadas respecto a las ventajas y limitaciones de cada arquitectura, permitiendo seleccionar el enfoque más adecuado según los requisitos específicos de aplicación.