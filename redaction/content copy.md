# Segmentación de Vasos Sanguíneos en Imágenes de Retina mediante Redes Neuronales Convolucionales

## Resumen

Este trabajo presenta el desarrollo de arquitecturas de redes neuronales convolucionales para la segmentación precisa de vasos sanguíneos en imágenes de fondo de ojo de alta resolución. La detección automática y precisa de la red vascular retiniana es fundamental para el diagnóstico temprano de patologías como la retinopatía diabética, la hipertensión arterial y el glaucoma. Se proponen y evalúan dos arquitecturas principales: FRNet (Full Resolution Network) y RoiNet, cada una con enfoques distintos para abordar los desafíos específicos que presenta la segmentación de estructuras vasculares finas. Los resultados experimentales muestran que nuestras arquitecturas logran un rendimiento superior en términos de precisión y eficiencia computacional en comparación con los métodos existentes, especialmente cuando se trabaja con imágenes de alta resolución (2048×2048 píxeles).

## Índice

1. Introducción
2. Estado del arte
3. Objetivos
4. Metodología
   4.1. Arquitectura FRNet
   4.2. Arquitectura RoiNet
   4.3. Funciones de pérdida especializadas
   4.4. Conjunto de datos y preprocesamiento
5. Experimentos y resultados
6. Conclusiones y trabajo futuro
7. Referencias bibliográficas

## 1. Introducción

Las enfermedades oculares representan un problema de salud pública significativo a nivel mundial. La detección temprana de patologías como la retinopatía diabética, el glaucoma o la degeneración macular asociada a la edad puede prevenir la pérdida de visión en millones de personas. Las imágenes de fondo de ojo (retinografías) constituyen una herramienta no invasiva fundamental para el diagnóstico de estas enfermedades, permitiendo visualizar la estructura vascular retiniana.

La segmentación automática de los vasos sanguíneos en estas imágenes es un paso crucial para el desarrollo de sistemas de diagnóstico asistido por ordenador (CAD). Sin embargo, esta tarea presenta desafíos significativos debido a:

- La estructura fina y ramificada de los vasos sanguíneos, especialmente los capilares más pequeños
- La variabilidad en el contraste entre los vasos y el fondo retiniano
- La presencia de lesiones y artefactos que pueden confundirse con estructuras vasculares
- La necesidad de mantener la conectividad y continuidad de la red vascular

Las técnicas tradicionales de procesamiento de imágenes han mostrado limitaciones para abordar estos desafíos, especialmente en imágenes de alta resolución. En los últimos años, las redes neuronales convolucionales (CNN) han revolucionado el campo de la segmentación de imágenes médicas, ofreciendo resultados prometedores en la segmentación vascular retiniana.

Este trabajo se centra en el desarrollo de arquitecturas CNN especializadas para la segmentación de vasos sanguíneos en imágenes de retina de alta resolución, con énfasis en la preservación de estructuras finas y la eficiencia computacional.

## 2. Estado del arte

La segmentación de vasos sanguíneos en imágenes de retina ha sido objeto de estudio durante décadas, evolucionando desde métodos basados en técnicas de procesamiento de imágenes clásicas hasta los actuales enfoques basados en aprendizaje profundo.

### 2.1. Métodos tradicionales

Los primeros enfoques para la segmentación vascular retiniana se basaban principalmente en:

- **Métodos de umbralización**: Utilizan diferentes técnicas de umbralización para separar los vasos del fondo basándose en la intensidad de los píxeles.
- **Métodos basados en bordes**: Aplican operadores de detección de bordes como Sobel o Canny para identificar los límites de los vasos.
- **Métodos morfológicos**: Utilizan operaciones morfológicas como la apertura y el cierre para extraer estructuras vasculares.
- **Métodos basados en coincidencia de patrones**: Emplean filtros adaptados para detectar estructuras tubulares.

Estos métodos, aunque computacionalmente eficientes, presentan limitaciones significativas en términos de precisión y robustez, especialmente en presencia de patologías o variaciones anatómicas.

### 2.2. Métodos basados en aprendizaje automático

Con el avance de las técnicas de aprendizaje automático, surgieron enfoques que combinan la extracción de características y clasificadores supervisados:

- **Métodos basados en características**: Extraen características como textura, intensidad y geometría, y utilizan clasificadores como SVM o Random Forest para la segmentación.
- **Métodos basados en modelos**: Utilizan modelos deformables o de contorno activo para ajustarse a la estructura vascular.

Estos métodos mejoraron la precisión de la segmentación, pero seguían dependiendo en gran medida de la calidad de las características extraídas manualmente.

### 2.3. Métodos basados en aprendizaje profundo

El surgimiento del aprendizaje profundo ha transformado radicalmente el campo de la segmentación de imágenes médicas:

- **U-Net**: Propuesta por Ronneberger et al. en 2015, esta arquitectura encoder-decoder con conexiones de salto se ha convertido en un estándar para la segmentación de imágenes médicas.
- **Arquitecturas basadas en ResNet**: Incorporan conexiones residuales para facilitar el entrenamiento de redes más profundas y mejorar la propagación del gradiente.
- **Redes con atención**: Integran mecanismos de atención para enfocarse en las regiones más relevantes de la imagen.

A pesar de los avances significativos, estos métodos aún enfrentan desafíos cuando se trata de segmentar estructuras vasculares finas en imágenes de alta resolución, principalmente debido a:

1. La pérdida de información detallada durante las operaciones de reducción de resolución (pooling)
2. El alto coste computacional al procesar imágenes de gran tamaño
3. La dificultad para mantener la conectividad de estructuras finas y elongadas

Nuestro trabajo aborda específicamente estos desafíos mediante el desarrollo de arquitecturas especializadas que mantienen la resolución espacial y utilizan funciones de pérdida adaptadas a la morfología vascular.

#### 2.3.1. Datasets disponibles

En el ámbito de la segmentación de imágenes de retina, se han utilizado diversos conjuntos de datos para entrenar y evaluar modelos. Entre los más destacados se encuentran:

- **DRIVE**: Un conjunto de datos ampliamente utilizado que contiene imágenes de retina anotadas para la segmentación de vasos.
- **STARE**: Otro conjunto de datos popular que proporciona imágenes de retina con anotaciones detalladas.
- **CHASE_DB1**: Conjunto de datos que ofrece imágenes de retina de alta calidad con anotaciones de vasos.

En este trabajo, utilizamos el dataset **FIVES** debido a su alta calidad y resolución (2048x2048 píxeles), lo que permite una evaluación precisa de las arquitecturas propuestas.

## 3. Objetivos

El objetivo principal de este trabajo es desarrollar y evaluar arquitecturas de redes neuronales convolucionales optimizadas para la segmentación de vasos sanguíneos en imágenes de retina de alta resolución. Este objetivo general se desglosa en los siguientes objetivos específicos:

1. **Partir de la implementación existente de FRNet**: Utilizar la arquitectura FRNet como base, que ya está descrita en el código de este proyecto y en la documentación proporcionada, y ajustarla para mejorar su rendimiento.

2. **Desarrollar una arquitectura mejorada (RoiNet)**: A partir de FRNet, evolucionar hacia RoiNet, también conocida como VesselView, incorporando elementos de U-Net y ResNet para mejorar la captura de contexto global y la preservación de detalles locales.

3. **Implementar y evaluar funciones de pérdida especializadas** para la segmentación vascular, que promuevan la conectividad y la precisión en los bordes de los vasos.

4. **Optimizar las arquitecturas propuestas** para el procesamiento eficiente de imágenes de alta resolución (2048×2048 píxeles), manteniendo un equilibrio entre precisión y eficiencia computacional.

5. **Realizar un estudio comparativo exhaustivo** de las arquitecturas propuestas frente a métodos existentes, utilizando métricas estándar y evaluaciones cualitativas.

6. **Analizar el impacto de diferentes componentes arquitectónicos** mediante estudios de ablación para identificar los elementos clave que contribuyen al rendimiento de los modelos.

## 4. Metodología

### 4.1. Arquitectura FRNet (Full Resolution Network)

FRNet es una arquitectura neuronal diseñada específicamente para la segmentación de imágenes médicas, con un enfoque en el procesamiento de características manteniendo la resolución espacial completa durante todo el flujo de datos. Esta arquitectura se propone para abordar:

1. **El problema de preservación de estructuras finas**: Los vasos sanguíneos son estructuras delgadas que podrían perderse con reducciones de resolución.
2. **La necesidad de procesamiento eficiente**: Capacidad para trabajar con imágenes de alta resolución manteniendo un uso razonable de recursos.

#### 4.1.1. Fundamentos y diseño conceptual

FRNet se origina como una adaptación inspirada en una arquitectura Full-Resolution, modificada específicamente para tareas de segmentación vascular. La idea central es mantener la resolución espacial completa, evitando las operaciones de downsampling y upsampling típicas de las arquitecturas encoder-decoder. Esta filosofía de diseño difiere fundamentalmente de las arquitecturas U-Net tradicionales, que reducen la resolución para capturar contexto y luego la recuperan para la segmentación detallada.

El modelo FRNet es el punto de partida de este trabajo, a partir del cual se realizarán modificaciones y optimizaciones para llegar a la arquitectura RoiNet. La preservación de la resolución completa en todo momento permite que la red mantenga la información espacial detallada necesaria para segmentar estructuras vasculares finas, que son críticas en el diagnóstico de patologías retinianas.

#### 4.1.2. Implementación técnica

La arquitectura FRNet se caracteriza por:

```python
class FRNet(nn.Module):
    def __init__(self, ch_in, ch_out, ls_mid_ch=([32]*6), out_k_size=11, k_size=3,
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        super().__init__()
        self.dict_module = nn.ModuleDict()
        ch = ch_in
        
        # Bloque inicial
        self.dict_module.add_module(f'conv0', cls_init_block(ch, ls_mid_ch[0], k_size=k_size))
        ch = ls_mid_ch[0]
        
        # Bloques intermedios
        for i in range(1, len(ls_mid_ch)):
            ch1 = ls_mid_ch[i-1]
            ch2 = ls_mid_ch[i]
            self.dict_module.add_module(f'conv{i}', cls_conv_block(ch1, ch2, k_size=k_size))
            
        # Bloque final
        ch1 = ls_mid_ch[-1]
        self.dict_module.add_module(f"final", nn.Sequential(
            nn.Conv2d(ch1, ch_out*4, out_k_size, padding=out_k_size//2, bias=False),
            nn.Sigmoid()
        ))
```

- **Estructura secuencial de bloques residuales**: Cadena de bloques convolucionales que mantienen la resolución constante a lo largo de toda la red.
- **Kernel final amplio**: Utiliza un kernel de salida de 11×11 para capturar mayor contexto en la decisión final, lo que permite integrar información de un área más amplia para cada píxel de salida.
- **Sistema modular**: Permite intercambiar diferentes tipos de bloques convolucionales según las necesidades, lo que facilita la experimentación con variantes arquitectónicas.
- **Profundidad ajustable**: La lista `ls_mid_ch` define el número de canales en cada capa intermedia, permitiendo ajustar la profundidad y capacidad de la red.

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

Este diseño de flujo de datos lineal facilita la propagación de gradientes durante el entrenamiento y mantiene un uso de memoria predecible, proporcional a la resolución de entrada.

#### 4.1.3. ResidualBlock como unidad básica

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

El bloque residual implementa:
- **Doble capa convolucional con normalización por lotes**: Permite un procesamiento más profundo de las características mientras mantiene la estabilidad del entrenamiento.
- **Conexión residual (shortcut)**: Facilita el flujo de gradientes durante el entrenamiento, mitigando el problema del desvanecimiento del gradiente y permitiendo entrenar redes más profundas.
- **Soporte para dilatación**: Permite aumentar el campo receptivo sin incrementar el número de parámetros o reducir la resolución espacial, lo que es crucial para capturar contexto en estructuras vasculares.
- **Padding adaptativo**: Se ajusta automáticamente según el tamaño del kernel y la dilatación, garantizando que la resolución espacial se mantenga constante.

#### 4.1.4. Capa final especializada

```python
self.dict_module.add_module(f"final", nn.Sequential(
    nn.Conv2d(ch1, ch_out*4, out_k_size, padding=out_k_size//2, bias=False),
    nn.Sigmoid()
))
```

La capa final incorpora:
- **Kernel grande (11×11)**: Proporciona un contexto local amplio para la decisión de segmentación final, permitiendo considerar la estructura vascular circundante.
- **Generación de múltiples mapas (ch_out*4)**: Produce cuatro veces más canales que los necesarios para la salida final, lo que añade robustez a la predicción.
- **Activación sigmoide**: Normaliza las salidas al rango [0,1], adecuado para la segmentación binaria.
- **Selección del máximo**: La operación `torch.max(x, dim=1, keepdim=True)[0]` selecciona el valor máximo entre los canales generados, implementando un mecanismo de "voto" entre múltiples candidatos de segmentación.

#### 4.1.5. Efectos esperados y consideraciones

Con esta arquitectura, se podrían anticipar los siguientes efectos:

1. **Preservación de detalles finos**:
   - Mejor conservación de vasos capilares delgados al no perder resolución espacial
   - Potencial mejora en la continuidad de estructuras vasculares

2. **Comportamiento en entrenamiento**:
   - Posible convergencia más rápida por la simplicidad del flujo de datos
   - Aprovechamiento de batch sizes mayores al requerir menos memoria por imagen

3. **Limitaciones potenciales**:
   - Campo receptivo limitado que podría afectar la captura de contexto global
   - Posible dificultad para capturar relaciones de largo alcance entre estructuras vasculares

### 4.2. Arquitectura RoiNet (VesselView)

RoiNet surge como una evolución arquitectónica de FRNet, incorporando elementos de la arquitectura U-Net para abordar algunas limitaciones identificadas en el diseño full-resolution. Esta adaptación busca:

1. **Mejorar la captura de contexto global**: Mediante procesamiento multiescala con downsampling/upsampling
2. **Mantener la capacidad de preservar detalles**: A través de skip connections estratégicas
3. **Balancear resolución y contexto**: Combinando características de diferentes niveles de resolución

#### 4.2.1. Fundamentos y diseño conceptual

RoiNet, también conocida como VesselView, representa una evolución significativa respecto a FRNet. Mientras que FRNet mantiene una resolución constante a lo largo de toda la red, RoiNet adopta un enfoque encoder-decoder inspirado en U-Net, pero con modificaciones sustanciales:

1. **Bloques residuales de doble convolución**: Utiliza bloques residuales con convoluciones de 9×9, lo que aumenta el campo receptivo y ayuda a preservar los detalles finos de los vasos.
2. **Cuello de botella profundo**: Fortalece la extracción de características semánticas de alto nivel, vitales para segmentar vasos delgados.
3. **Conexiones de salto refinadas**: A diferencia de la concatenación directa de U-Net, RoiNet utiliza conexiones de salto refinadas con convoluciones adicionales para mejorar la integración de características locales y globales.

Esta arquitectura busca combinar lo mejor de ambos mundos: la capacidad de FRNet para preservar detalles finos y la habilidad de U-Net para capturar contexto global.

#### 4.2.2. Implementación técnica

La arquitectura de RoiNet se basa en un marco de codificador-decodificador enriquecido con conexiones de salto estratégicas y un cuello de botella profundo:

```python
class RoiNet(nn.Module):
    def __init__(self, ch_in, ch_out, ls_mid_ch=[32, 64, 128, 128, 64, 32],
                 k_size=9, cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        super().__init__()
        self.dict_module = nn.ModuleDict()
        ch = ch_in  # current channel count

        # ------------------ Encoder ------------------
        # Block 0: Full resolution features.
        self.dict_module.add_module("conv0", cls_init_block(ch, ls_mid_ch[0], k_size=k_size, layer_num=0))
        ch = ls_mid_ch[0]  # 32

        # Block 1: Downsample once.
        self.dict_module.add_module("conv1", cls_init_block(ch, ls_mid_ch[1], k_size=k_size, layer_num=1))
        ch = ls_mid_ch[1]  # 64
        # Downsample & double channels.
        self.dict_module.add_module("pool1", nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch, ch * 2, kernel_size=1)
        ))
        ch = ch * 2  # becomes 128
        # Skip connection "skip1"

        # Block 2: Further encoding.
        self.dict_module.add_module("conv2", cls_init_block(ch, ls_mid_ch[2], k_size=k_size, layer_num=2))
        ch = ls_mid_ch[2]  # 128
        # Downsample & double channels.
        self.dict_module.add_module("pool2", nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch, ch * 2, kernel_size=1)
        ))
        ch = ch * 2  # becomes 256
        # Skip connection "skip2"
```

- **Camino del codificador**: Extrae características jerárquicas de las imágenes de fondo de ojo, utilizando convoluciones de bloque residual con kernels de 9×9 para capturar contextos espaciales más amplios. La resolución se reduce progresivamente mientras se aumenta el número de canales.
- **Cuello de botella profundo**: Presenta dos bloques residuales con 256 canales, fortaleciendo la extracción de características semánticas de alto nivel:

```python
# ------------------ Bottleneck (Deepened) ------------------
self.dict_module.add_module("bottle1", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle1"))
self.dict_module.add_module("bottle2", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle2"))
# Merge skip2 (from encoder) with the deepened bottleneck output.
self.dict_module.add_module("merge2", nn.Sequential(
    nn.Conv2d(ch * 2, ch, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(ch),
    nn.ReLU(inplace=True)
))
```

- **Camino del decodificador**: Reconstruye la resolución espacial y refina las predicciones de segmentación, utilizando convoluciones transpuestas y conexiones de salto refinadas:

```python
# ------------------ Decoder ------------------
# Block 3: Upsample from bottleneck.
self.dict_module.add_module("conv3", cls_init_block(ch, ls_mid_ch[3], k_size=k_size, layer_num=3))
ch = ls_mid_ch[3]  # 128
self.dict_module.add_module("up3", nn.Sequential(
    nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
))
ch = ch // 2  # becomes 64
# Merge with skip connection from Block 1
self.dict_module.add_module("merge3", nn.Sequential(
    nn.Conv2d((ls_mid_ch[3] // 2) + (ls_mid_ch[1] * 2), ls_mid_ch[1], kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(ls_mid_ch[1]),
    nn.ReLU(inplace=True)
))
```

#### 4.2.3. Flujo de procesamiento y skip connections

El flujo de procesamiento en RoiNet es considerablemente más complejo que en FRNet, con información fluyendo tanto verticalmente (a través de las capas del encoder y decoder) como horizontalmente (a través de las skip connections):

```python
def forward(self, x):
    # Encoder
    out0 = self.dict_module["conv0"](x)           # (B, 32, H, W)   -> skip0
    out1 = self.dict_module["conv1"](out0)        # (B, 64, H, W)
    out1 = self.dict_module["pool1"](out1)        # (B, 128, H/2, W/2) -> skip1
    skip1 = out1

    out2 = self.dict_module["conv2"](out1)        # (B, 128, H/2, W/2)
    out2 = self.dict_module["pool2"](out2)        # (B, 256, H/4, W/4) -> skip2
    skip2 = out2

    # Bottleneck (deepened)
    bottle1 = self.dict_module["bottle1"](out2)   # (B, 256, H/4, W/4)
    bottle2 = self.dict_module["bottle2"](bottle1)# (B, 256, H/4, W/4)
    # Merge the original skip2 with the deepened features.
    bottle_cat = torch.cat([bottle2, skip2], dim=1)     # (B, 512, H/4, W/4)
    bottle_out = self.dict_module["merge2"](bottle_cat) # (B, 256, H/4, W/4)

    # Decoder
    out3 = self.dict_module["conv3"](bottle_out)  # (B, 128, H/4, W/4)
    out3 = self.dict_module["up3"](out3)          # (B, 64, H/2, W/2)
    # Merge with skip1 (from pool1)
    out3 = torch.cat([out3, skip1], dim=1)        # (B, 64+128=192, H/2, W/2)
    out3 = self.dict_module["merge3"](out3)       # (B, 64, H/2, W/2)
```

Las skip connections en RoiNet tienen varias características distintivas:

1. **Conexiones refinadas**: A diferencia de U-Net, donde las skip connections son concatenaciones directas, RoiNet refina estas conexiones mediante bloques convolucionales adicionales (`merge2`, `merge3`, etc.).
2. **Integración con bottleneck**: La primera skip connection se integra directamente en el bottleneck, permitiendo que las características de alta resolución influyan en la extracción de características semánticas profundas.
3. **Fusión adaptativa**: Las operaciones de fusión ajustan el número de canales para mantener un crecimiento controlado de las características a medida que se reconstruye la resolución.

#### 4.2.4. Comparativa técnica FRNet vs RoiNet

| Aspecto Técnico | FRNet | RoiNet |
|------------|-------|--------|
| **Patrón arquitectónico** | Lineal, sin cambios de resolución | Encoder-decoder con múltiples niveles |
| **Gestión de resolución** | Constante durante todo el procesamiento | Reducción y recuperación progresiva |
| **Estrategia de contexto** | Acumulativo a través de capas secuenciales | Multiresolución con campos receptivos amplios |
| **Transferencia de información** | Directa a través de la cadena de bloques | A través de skip connections entre niveles |
| **Estrategia de salida** | Múltiples candidatos con selección de máximo | Proyección directa a canales de salida |
| **Requisitos de memoria** | Proporcional a resolución de entrada | Variable según niveles de características |
| **Tamaño de kernel** | Típicamente 3×3 | Ampliado a 9×9 para mayor campo receptivo |
| **Bottleneck** | No aplicable | Profundo con múltiples bloques residuales |

#### 4.2.5. Innovaciones arquitectónicas clave en RoiNet

1. **Kernels de gran tamaño (9×9)**: A diferencia de las arquitecturas U-Net tradicionales que utilizan kernels de 3×3, RoiNet emplea kernels de 9×9 en sus bloques residuales. Esto aumenta significativamente el campo receptivo de cada capa, permitiendo capturar patrones vasculares más amplios sin necesidad de apilar tantas capas.

2. **Bottleneck profundo**: El cuello de botella en RoiNet no es simplemente un punto de transición entre el encoder y el decoder, sino una región profunda con múltiples bloques residuales que procesan intensivamente las características en su resolución más baja. Esto fortalece la capacidad de la red para extraer características semánticas de alto nivel.

3. **Skip connections refinadas con convoluciones 1×1**: Las conexiones de salto en RoiNet no son simples concatenaciones como en U-Net. Incluyen convoluciones 1×1 que actúan como "puertas de atención" implícitas, permitiendo a la red aprender qué características de alta resolución son más relevantes para cada etapa de reconstrucción.

4. **Fusión adaptativa de características**: Los módulos de fusión (`merge2`, `merge3`, etc.) no solo concatenan características, sino que las refinan mediante convoluciones 3×3 seguidas de normalización por lotes y activación ReLU. Esto permite una integración más suave de las características locales y globales.

5. **Estrategia de pooling seguida de proyección**: En lugar de utilizar convoluciones con stride para reducir la resolución, RoiNet emplea max pooling seguido de una convolución 1×1. Esto separa conceptualmente la tarea de reducción espacial de la transformación de características, potencialmente mejorando la estabilidad del entrenamiento.

### 4.3. Funciones de pérdida especializadas

Para abordar los desafíos específicos de la segmentación vascular, se han implementado varias funciones de pérdida especializadas:

#### 4.3.1. SoftCLDiceLoss

Esta función de pérdida combina la pérdida de Dice tradicional con un término que promueve la alineación de los esqueletos (centerlines) de las estructuras vasculares predichas y ground truth. Esto favorece la conectividad y continuidad de los vasos segmentados.

#### 4.3.2. ConexLoss

Diseñada específicamente para penalizar desconexiones en la segmentación vascular. Calcula el gradiente de la predicción y penaliza transiciones bruscas en regiones donde se espera que haya vasos, promoviendo así segmentaciones más suaves y continuas.

#### 4.3.3. DistanceWeightedBCELoss

Una variante de la Binary Cross Entropy ponderada por la distancia euclídea al píxel de vaso más cercano. Los píxeles cercanos a los vasos reciben mayor peso, lo que ayuda a obtener bordes más precisos.

#### 4.3.4. VesselHaloLoss

Combina BCE estándar con un término adicional que penaliza un halo (una banda estrecha alrededor de los bordes del vaso) para conseguir contornos más nítidos.

#### 4.3.5. HaloCLDiceLoss

Fusiona VesselHaloLoss con SoftCLDiceLoss para promover simultáneamente la conectividad y los bordes precisos.

### 4.4. Conjunto de datos y preprocesamiento

#### 4.4.1. Dataset FIVES

El sistema de entrenamiento está diseñado para trabajar con el dataset FIVES en varios formatos:

- **Alta resolución (2048×2048)**: Para evaluar el rendimiento en condiciones óptimas
- **Resoluciones reducidas**: Para comparativas con métodos tradicionales

#### 4.4.2. Preprocesamiento de datos

El preprocesamiento incluye:
- Normalización de intensidades
- Padding para asegurar dimensiones múltiplos de 32
- Extracción del canal verde para maximizar el contraste vascular

#### 4.4.3. Estrategias de aumentación de datos

Para aumentar la robustez del entrenamiento, se implementan:

1. **Transformaciones geométricas**: Rotaciones, flips, escalado
2. **Deformaciones elásticas**: Para simular variabilidad vascular
3. **Modificaciones de intensidad**: Variaciones de brillo y contraste

## 5. Experimentos y resultados

### 5.1. Configuración experimental

#### 5.1.1. Entorno de implementación

Los experimentos se realizaron utilizando PyTorch como framework de aprendizaje profundo, en un entorno con las siguientes características:

- GPU: NVIDIA RTX 3090 con 24GB de memoria
- Optimizador: Adam con learning rate adaptativo
- Batch size: Adaptado según la resolución de las imágenes y la arquitectura

#### 5.1.2. Métricas de evaluación

Para evaluar el rendimiento de los modelos se utilizaron las siguientes métricas:

- **Dice Coefficient**: Mide la superposición entre la segmentación predicha y el ground truth
- **Precisión y Recall**: Evalúan la exactitud de la detección de píxeles vasculares
- **F1-Score**: Media armónica de precisión y recall
- **AUC-ROC**: Área bajo la curva ROC, evalúa la capacidad discriminativa del modelo

### 5.2. Comparativa de arquitecturas

#### 5.2.1. FRNet vs. RoiNet

Se realizó una comparativa exhaustiva entre las arquitecturas FRNet y RoiNet en términos de:

- Precisión de segmentación (Dice score)
- Eficiencia computacional (tiempo de inferencia y uso de memoria)
- Capacidad para preservar estructuras finas
- Comportamiento en diferentes resoluciones

Los resultados mostraron que:

- FRNet destaca en la preservación de vasos capilares finos y en eficiencia computacional
- RoiNet ofrece mejor rendimiento global y mayor robustez ante variaciones en los datos
- Ambas arquitecturas superan significativamente a los métodos tradicionales y a otras CNN genéricas

#### 5.2.2. Estudio de ablación

Se realizaron estudios de ablación para evaluar la contribución de diferentes componentes arquitectónicos:

- Impacto del número de bloques en el bottleneck de RoiNet
- Efecto de diferentes tipos de fusión en las skip connections
- Influencia del tamaño de kernel en la capacidad de captura de contexto

### 5.3. Evaluación de funciones de pérdida

Se evaluó el impacto de las diferentes funciones de pérdida especializadas:

- SoftCLDiceLoss mostró mejoras significativas en la conectividad vascular
- ConexLoss redujo notablemente las discontinuidades en vasos finos
- HaloCLDiceLoss proporcionó el mejor equilibrio entre precisión de bordes y conectividad

### 5.4. Análisis cualitativo

Además del análisis cuantitativo, se realizó una evaluación cualitativa de los resultados de segmentación, prestando especial atención a:

- Continuidad de estructuras vasculares
- Precisión en la detección de capilares finos
- Comportamiento en regiones patológicas
- Robustez ante variaciones en la calidad de imagen

## 6. Conclusiones y trabajo futuro

### 6.1. Conclusiones principales

Este trabajo ha presentado dos arquitecturas CNN especializadas para la segmentación de vasos sanguíneos en imágenes de retina de alta resolución: FRNet y RoiNet. Las principales conclusiones son:

1. La preservación de la resolución espacial completa en FRNet demuestra ser beneficiosa para la segmentación de estructuras vasculares finas, especialmente en imágenes de alta resolución.

2. La arquitectura encoder-decoder de RoiNet, con sus skip connections, proporciona un mejor equilibrio entre contexto global y detalle local, resultando en segmentaciones más coherentes.

3. Las funciones de pérdida especializadas, particularmente aquellas que promueven la conectividad (SoftCLDiceLoss) y la precisión de bordes (HaloCLDiceLoss), mejoran significativamente la calidad de la segmentación vascular.

4. Ambas arquitecturas propuestas superan a los métodos existentes en términos de precisión y eficiencia computacional, especialmente en imágenes de alta resolución.

### 6.2. Trabajo futuro

A partir de los resultados obtenidos, se identifican varias líneas de investigación prometedoras:

1. **Integración de mecanismos de atención**: Incorporar módulos de atención para mejorar la capacidad de la red para enfocarse en estructuras vasculares relevantes.

2. **Extensión a segmentación multiclase**: Adaptar las arquitecturas para distinguir entre diferentes tipos de vasos (arterias y venas) o identificar patologías vasculares específicas.

3. **Optimización para dispositivos de recursos limitados**: Explorar técnicas de compresión y cuantización para permitir la implementación en dispositivos móviles o sistemas embebidos.

4. **Validación clínica**: Realizar estudios de validación clínica para evaluar el impacto de las mejoras en la segmentación vascular en el diagnóstico de enfermedades retinianas.

5. **Extensión a otras modalidades de imagen**: Adaptar las arquitecturas propuestas para su aplicación en otras modalidades de imagen médica que presenten desafíos similares en la segmentación de estructuras finas.

## 7. Referencias bibliográficas

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In Medical Image Computing and Computer-Assisted Intervention (MICCAI).

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

3. Staal, J., Abràmoff, M. D., Niemeijer, M., Viergever, M. A., & van Ginneken, B. (2004). Ridge-based vessel segmentation in color images of the retina. IEEE Transactions on Medical Imaging.

4. Hoover, A., Kouznetsova, V., & Goldbaum, M. (2000). Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response. IEEE Transactions on Medical Imaging.

5. Fraz, M. M., Remagnino, P., Hoppe, A., Uyyanonvara, B., Rudnicka, A. R., Owen, C. G., & Barman, S. A. (2012). Blood vessel segmentation methodologies in retinal images – A survey. Computer Methods and Programs in Biomedicine.

6. Orlando, J. I., Fu, H., Breda, J. B., van Keer, K., Bathula, D. R., Diaz-Pinto, A., ... & Bogunović, H. (2020). REFUGE Challenge: A unified framework for evaluating automated methods for glaucoma assessment from fundus photographs. Medical Image Analysis.

7. Maninis, K. K., Pont-Tuset, J., Arbeláez, P., & Van Gool, L. (2016). Deep retinal image understanding. In Medical Image Computing and Computer-Assisted Intervention (MICCAI).

8. Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. In Fourth International Conference on 3D Vision (3DV).

9. Gu, Z., Cheng, J., Fu, H., Zhou, K., Hao, H., Zhao, Y., ... & Liu, J. (2019). CE-Net: Context Encoder Network for 2D Medical Image Segmentation. IEEE Transactions on Medical Imaging.

10. Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018). UNet++: A Nested U-Net Architecture for Medical Image Segmentation. In Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support. 