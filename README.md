# Footbalytics

**Pedro Cuevas**

## Descripción

Footbalytics es un proyecto innovador que integra una plataforma web desarrollada con Dash, enfocada en presentar estadísticas detalladas de equipos de fútbol. Además, ofrece una herramienta única para generar modelos de Machine Learning (ML) personalizados para jugadores, basados en datos de la temporada 2022-2023.

## Consideraciones de Diseño

Secciones Principales:

- Dashboard de Estadísticas: permite visualizar una amplia gama de estadísticas de cualquier equipo para la temporada 2022-2023.
- Generador de Modelos de ML:
  - Descripción: permite al usuario elegir qué variables usar (de un dataset de estadísticas de jugadores de la FIFA), mientras visualiza un heatmap de correlaciones en tiempo real. Cuando ha elegido las variables, puede generar un modelo de regresión lineal y ver información sobre su desempeño y características.
  - Enfoque: Se ha optado por un modelo de Regresión Lineal debido a:
    - Su simplicidad y facilidad de comprensión, ideal para el usuario promedio, que es el público objetivo de la aplicación.
    - Eficiencia computacional.
    - Capacidad para revelar relaciones interesantes entre variables, lo cual es crucial en este contexto.
Otros Comentarios
Debido a las limitaciones de mi API, que permite un máximo de 100 llamadas diarias, he decidido emplear un conjunto de datos diferente para la generación del modelo de ML. Esta decisión fue necesaria para garantizar la viabilidad y el funcionamiento óptimo del proyecto.

## Agradecimientos y Menciones

- **Dataset:** El preprocesamiento y la selección del conjunto de datos se basaron en el proyecto disponible en Kaggle: [Análisis de FIFA - Regresión en el Valor del Jugador](https://www.kaggle.com/code/phdoot2/fifa-analysis-regression-on-player-value)
  
- **API de Equipos:** La información sobre los equipos se obtiene gracias a la API proporcionada por [API-Football](https://www.api-football.com/)
