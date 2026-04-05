# Original Theoretical Notes (Spanish)

> Extracted from the preserved original notebook so the delivery narrative is readable without opening Jupyter.

## Importacion de la libreria de Pandas y numpy

# Importacion de Datasets

# Encuesta Anual de Hogares (EAH)

Fuente: https://data.buenosaires.gob.ar/dataset/encuesta-anual-hogares

## ABSTRACT

La Dirección General de Estadísticas y Censos (DGEyC) de la Ciudad de Buenos Aires lleva a cabo anualmente la Encuesta Anual de Hogares (EAH), cuyo objetivo principal es obtener información valiosa sobre las características demográficas, educativas, laborales y de salud de la población de la ciudad. Esta encuesta se realiza en todo el territorio de la ciudad y se diseña una muestra representativa que permita obtener resultados confiables para el total de la ciudad y cada una de sus comunas.

La EAH es una herramienta fundamental para la planificación de políticas y programas sociales, educativos y laborales en la ciudad. La información recopilada en esta encuesta permite conocer la situación de la población de la Ciudad de Buenos Aires en diversas áreas, como la salud, la educación y el empleo, lo que a su vez permite diseñar estrategias y programas específicos para abordar las necesidades y problemáticas de cada comunidad y grupo de población.

Además, la EAH permite el análisis de la evolución de diversas variables sociodemográficas a lo largo del tiempo, lo que permite identificar tendencias y patrones que pueden ser útiles para la toma de decisiones informadas en la planificación de políticas públicas a largo plazo. En este sentido, el análisis de los resultados de la EAH puede contribuir significativamente a la mejora del bienestar y la calidad de vida de la población de la Ciudad de Buenos Aires.

## Preguntas e Hipotesis

1. **¿La ubicación geográfica influye en el ingreso familiar?**

  - Hipótesis nula: No existe una diferencia significativa en el ingreso familiar promedio entre distintas zonas geográficas en la Ciudad de Buenos Aires.
  - Hipótesis alternativa: Existe una diferencia significativa en el ingreso familiar promedio entre distintas zonas geográficas en la Ciudad de Buenos Aires.

2. **¿Cuál es la relación entre el número de integrantes de la familia y el ingreso total?**

  - Hipótesis nula: No existe una correlación significativa entre el número de integrantes de la familia y el ingreso total familiar en la Ciudad de Buenos Aires.
  - Hipótesis alternativa: Existe una correlación significativa entre el número de integrantes de la familia y el ingreso total familiar en la Ciudad de Buenos Aires.

3. **¿Existe una relación entre el ingreso per cápita familiar y la situación conyugal de los miembros del hogar?**
  - Hipótesis nula: No existe una diferencia significativa en el ingreso per cápita familiar entre los hogares con miembros casados y los hogares con miembros solteros o en otra situación conyugal en la Ciudad de Buenos Aires.
  - Hipótesis alternativa: Existe una diferencia significativa en el ingreso per cápita familiar entre los hogares con miembros casados y los hogares con miembros solteros o en otra situación conyugal en la Ciudad de Buenos Aires.

4. **¿Existe una relación entre la educación y el ingreso total de los hogares por familia en la Ciudad de Buenos Aires?**
  - Hipótesis nula: No existe una correlación significativa entre el nivel educativo de los miembros de la familia y el ingreso total familiar en la Ciudad de Buenos Aires.
  - Hipótesis alternativa: Existe una correlación significativa entre el nivel educativo de los miembros de la familia y el ingreso total familiar en la Ciudad de Buenos Aires.

###OBJETIVO

El objetivo de esta actividad es realizar un análisis de datos utilizando la Encuesta Anual de Hogares (EAH) de la Ciudad de Buenos Aires para identificar las variables que afectan el ingreso total de una familia y construir un modelo de regresión lineal que permita predecir el ingreso total en función de estas variables.

###CONTEXTO COMERCIAL

La Dirección General de Estadísticas y Censos (DGEyC) de la Ciudad de Buenos Aires lleva a cabo la Encuesta Anual de Hogares (EAH) para obtener información valiosa sobre las características demográficas, educativas, laborales y de salud de la población de la ciudad. La información recopilada en esta encuesta es fundamental para la planificación de políticas y programas sociales, educativos y laborales en la ciudad.

###Problema Comercial

La DGEyC quiere entender qué factores influyen en el ingreso total de las familias en la ciudad y cómo se pueden utilizar estos factores para planificar políticas y programas que mejoren el bienestar y la calidad de vida de la población.

###Contexto Analítico

Se utilizará la Encuesta Anual de Hogares (EAH) de la Ciudad de Buenos Aires para realizar un análisis exploratorio de datos y construir un modelo de regresión lineal. Se identificarán las variables que tienen una mayor correlación con el ingreso total de las familias y se utilizará un enfoque de modelado de regresión lineal para predecir el ingreso total en función de estas variables.

# EDA

##Verificacion de datos faltantes
Una vez cargados los datos, lo primero que debemos hacer es ver el tamaño del conjunto de datos y si hay datos faltantes.

Podemos ver que faltan datos principalmente en 3 columnas, pero estas no van a ser usadas en el analiis, igualmente vamos a reemplazar los faltantes con nulos. **Esto no se entiende muy bien, en especial lo que hiciste en la parte de abajo del codigo. Por lo general, las matrices de nulos se usan para ver la correlacion visual de nulos entre columnas, es decir, si dos columnas con nulos coinciden en las ocurrencias de nulos. Ademas te recomiendo que no uses msgno, no porque sea mala, sino porque podes realizar lo mismo utilizando las otras librerias que estan mas consolidadas. Msgno puede desaparecer el dia de mañana**

##Exploramos el dataset en busqueda de insights


**Hubiera sido interesante que crearas secciones de ingresos (bajos, medios, altos, muy altos) basadas en los quartiles y luego ploteado como grafico de barras.
Te permite ya tener una nueva variable que puede tener peso predictivo y para graficar hubiese quedado, probablemente, algo mas significativo.**

Podemos también explorar la relación entre algunas variables y el ingreso total de las familias. Parece haber una correlación positiva débil entre el ingreso total de la familia y la edad de los miembros, lo que sugiere que las familias con miembros más viejos tienden a tener ingresos más altos en promedio. **Aca idealmente deberias haber usado una iteracion para plotear todas las variables cuantitativas en relación al ingreso**

**Lo que graficas a continuación no es un scatter plot ni muestra dispersión, sino distribución**

##AGRUPACION POR FAMILIA

En este ejemplo, estamos cargando los datos desde un archivo CSV llamado datos_familiares.csv (asumiendo que este es el nombre del archivo que contiene los datos que mencionaste anteriormente). Luego, usamos el método groupby() para agrupar los datos por ID familiar y seleccionamos las variables relevantes mediante una lista de nombres de columnas ('INGRESOS_FAMILIARES', 'NUM_MIEMBRO_PADRE', etc.). Finalmente, usamos el método mean() para obtener la media de cada variable para cada grupo de ID familiar.

# Pregunta 1: ¿La ubicacion Geografica influye en el ingreso familiar?
---

En estas visualizaciones haremos un recorrido por los datos mas importantes de la ciudad de buenos aires e intentaremos llegar a la conclusion de si es homogenea o no en sus caracteristicas

###Distribucion de empleo en la ciudad
podemos ver que en proporcion la comuna con mas desocupados e inactivos es la 8. Mientras que el resto de las comunas (menos la 9) tienen mas ocupados que inactivos. **Ojo con esto, que las barras sean mas grandes solo indica mayor cantidad de ocurrencias dentro del dataset, pero comparativamente lo mas importante aca es la proporcionalidad del hue para cada comuna. Y en general las proporcionalidades son relativamente similares. Otro tema a tener en cuenta es si inactivo no deberia ser parte desocupado, porque a los efectos practicos ambas categorias, a priori, van a tener el mismo valor predictivo.**

###DISTRIBUCION GEOGRAFICA DE FAMILIAS

En este primer grafico de barras podemos observar que las comunas 1 (Retiro, San Nicolás, Puerto Madero, San Telmo, Montserrat y Constitución), 4 (La Boca, Barracas, Parque Patricios y Nueva Pompeya) y 8 (Villa Soldati, Villa Riachuelo y Villa Lugano) tienen una mayor concentracion de familias respecto a las demas comunas. Y las comunas 10 (Villa Real, Monte Castro, Versalles, Floresta, Vélez Sarsfield y Villa Luro) y 11 (Villa General Mitre, Villa Devoto, Villa del Parque y Villa Santa Rita) son las que menos concentracion de familias tienen. Esto es importante ya que es importante tenerlo en cuenta al interpretar los resultados de la regresión y comprender mejor la relación entre las variables.

###DISTRIBUCION DEL INGRESO FAMILIAR POR COMUNA

Para ver si la ubicación geográfica influye en el ingreso familiar, podemos utilizar un diagrama de dispersión donde el eje x sea la comuna y el eje y sea el ingreso total familiar. La visualización muestra la distribución de ingresos familiares para cada una de las comunas de la Ciudad de Buenos Aires. Podemos ver que las comunas 14, 2 y 13 tienen una mediana de ingresos familiares más alta en comparación con las otras comunas, lo que indica que las familias en esas comunas tienen un ingreso más alto en promedio. Por otro lado, las comunas 8, 9, 3 y 4 tienen una mediana de ingresos familiares más baja, lo que indica que las familias en esas comunas tienen un ingreso más bajo en promedio.

Para finalizar podemos concluir que en este caso se cumple la hipótesis alternativa. Esta sugiere que sí hay una diferencia significativa en el ingreso familiar promedio entre distintas zonas geográficas en la Ciudad de Buenos Aires. En este caso, se espera que el ingreso promedio varíe en función de la ubicación geográfica de los hogares. Si se acepta esta hipótesis, se concluye que la ubicación geográfica sí tiene un impacto significativo en el ingreso familiar promedio.

# 2. ¿Cual es la relacion entre el numero de integrantes de la familia y el ingreso total?

###REGRESION LINEAL

**Por lo general statsmodels no es la primera opcion, se suele usar mas scikitlearn**

podemos definir nuestras variables dependiente e independientes. En este caso, la variable dependiente es el ingreso total de la familia y las variables independientes son las características familiares que queremos incluir en nuestra regresión. Podemos seleccionar estas columnas del DataFrame df_familias que creamos anteriormente:

**Aca falto separar el dataframe en uno de entrenamiento y testeo, para probar el modelo con datos no utilizados.**

**Me parece que habia muchas mas variables que se podian emplear dentro del moedelo, ademas de separar los datos para testear su capacidad real de generalización**

*   El modelo de regresión que hemos
aplicado nos muestra que hay una relación positiva entre el número de integrantes de la familia y los ingresos familiares totales. Es decir, a medida que aumenta el número de integrantes en la familia, es posible que haya más fuentes de ingresos en la familia, lo que puede llevar a un mayor ingreso familiar.

*   Por otro lado, hemos observado que existe una relación negativa entre el ingreso per cápita promedio de la familia y los ingresos familiares totales. Es decir, a medida que aumenta el ingreso per cápita promedio de la familia, disminuyen los ingresos familiares totales.
Esta situación puede ser vista como negativa en el sentido de que una familia con ingresos más altos individualmente podría no estar necesariamente mejor en términos de su ingreso total, y esto puede indicar una desigualdad económica dentro de la familia.

*   hemos encontrado una relación positiva entre los ingresos laborales totales y los ingresos familiares totales. Es decir, si los miembros de la familia tienen más ingresos por trabajo, es más probable que la familia en su conjunto tenga un ingreso alto.

*   l valor de R-cuadrado ajustado de la regresión es 0.764, lo que significa que aproximadamente el 76.4% de la variabilidad en el ingreso familiar total puede explicarse por las tres variables incluidas en el modelo (NUM_INTEGRANTES, INGRESO_PER_CAPITA_PROMEDIO e INGRESOS_LABORALES_TOTALES). Esto indica que el modelo de regresión tiene un buen ajuste y que las variables seleccionadas son relevantes para explicar la variación en el ingreso familiar total.** Esto esta mal. El modelo solo corrio con esas variables seleccionadas, asi que solo explica en relación a esas variables. Para ver el comportamiento real con todas las variables originales deberias correr el modelo con regularizacion L1 o Lasso**

*   El valor de AIC y BIC en la regresión es el mismo, lo que indica que el modelo no está sobrecargado y no hay variables redundantes que puedan estar afectando negativamente la capacidad del modelo para explicar la variación en el ingreso familiar total.

En resumen, podemos decir que el tamaño de la familia y los ingresos laborales totales tienen un impacto positivo en los ingresos familiares totales, mientras que el ingreso per cápita promedio tiene un impacto negativo.

Para finalizar podemos concluir que en este caso se cumple la hipótesis alternativa. Esta sugiere que sí existe una correlación significativa entre el número de integrantes de la familia y el ingreso total familiar en la Ciudad de Buenos Aires. En este caso, se espera que el tamaño de la familia tenga un efecto en el ingreso total, pudiendo ser positivo o negativo. Si se acepta esta hipótesis, se concluye que el tamaño de la familia tiene un impacto significativo en el ingreso total familiar.

# 3. ¿Existe una relación entre el ingreso per cápita familiar y la situación conyugal y edad de los miembros del hogar?

Podemos ver que los que estan divorciados, viudos y unidos ganan mas en promedio que los solteros, ademas tienen valores maximos mas altos.

el ultimo grafico relación entre todas las variables al mismo tiempo, lo que nos permite identificar patrones y relaciones complejas entre las variables. Com que a medida que crecemos vamos teniendo mas porcentaje del ingreso familiar.

Concluimos que H0 es falso y decimos que rechazamos H0. En este caso concluiremos que existe evidencia estadística para la hipótesis alternativa Ha ya que Existe una relación significativa entre el ingreso per cápita familiar y la situación conyugal y la edad de los miembros del hogar.

# 4.  ¿Existe una relación entre la educación y el ingreso total de los hogares por familia en la Ciudad de Buenos Aires?