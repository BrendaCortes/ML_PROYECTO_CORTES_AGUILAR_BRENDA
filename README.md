# ML_PROYECTO_CORTES_AGUILAR_BRENDA

Objetivos:
•	Emplear técnicas de ML para realizar predicciones.
•	Reforzar el conocimientos en el análisis y exploración de los datos.

Descripción de la actividad:
Este proyecto consiste en cuatro etapas principales:
1.	EDA.
2.	Manipulación y tratamiento de los datos.
3.	Machine Learning
4.	Presentación del proyecto.


El dataset a utilizar se llama “cirrhosis.csv"


## 1.	ETAPA 1 EDA.

Paso 1. Carga y muestra.
- Cargue el dataset llamado "cirrhosis.csv" en un dataframe, posteriormente muestre su información.

Se observar que el data set, tienen un total de 20 columnas, contiene tipos de dato, "Object", "Int" y "float" y un total de 418 registros, también se observa que más del 50% del data set, está representado por columnas numericas.

Paso 2. Análisis estadístico.
 - De las columnas numéricas, muestre la información estadística relevante (promedio, cuartíles, desviación estándar y coeficiente de variación).
 - A través del coeficiente de variación determine qué columnas presentan mucha dispersión en sus datos.

Siguiendo el criterio de,  0 - 0.5, existe poca dispersión, 0.5-1 existen dispersión media y anomalías no graves y coeficiente > 1, demasiada dispersión y anomalías graves. 

La mayoria de las columnas no presentan una dispersion alarmante, pero existen dos columna que presentan una dispersión bastante alta, como lo es Bilirubin y Alk_Phos. 

El coefiente mayor a 1 son un indicador de dispersión bastante grande, por lo que podría no ser viable utilizar el método del rango intercuartílico para tratar los datos atípicos.

Paso 3. Búsqueda de nulos y datos atípicos.
 - Muestre cuántos datos nulos tienen las columnas, puede apoyarse de un gráfico para mostrar la cantidad de nulos que hay.

Podemos observar presencia de datos nulos en mas del  50% de las columnas, la gran mayoria de las columnas contienen una cantidad similar de datos nulos y un trio de columnas presentan niveles bajos, abra que determinar la mejor manera para tratarlos.

- Muestre las distribuciones de las columnas numéricas y mencione si presentan datos atípicos/anomalías/outliers.

Como podemos observar en los graficos de boxplot, las unicas columnas numericas que no presentan anomalias, son: N_Days, Stage y Age.

Paso 4. Análisis de categorización.
- Revise las columnas que son de tipo objeto, analice la cantidad de posibilidades que tienen.

- Determine si una (o algunas) puede ser categorizable, aún no la(s) categorice.

Una vez analizado los valores únicos detectados en las columnas objeto, puedo determinar que es viable la categorización de las 7 columnas.


La mayoría de estas columnas presentan valores simples que nos indica una pertenencia a un grupo u otro. Este dato puede ser clave para determinar que tipo de encoder podriamos utilizar en dado caso que se requiera convertir a numerico.  

Paso 5. Correlación y análisis del problema.

- Muestre la correlación de los datos numéricos con respecto a la columna "Status", para ello tendrá que volver numérica dicha columna (para este paso puede usar un encoder).

Únicamente contamos con dos relaciones que se consideran mediante fuertes con un coeficiente de relación de 4, una de ellas con la variable Bilirubin y una relación negativa con la variable N_Days

- El objetivo es predecir con estos datos la variable "Status".

Siguiendo el criterio de que, 0.1 - 0.3 se considera una relación débil, de 0.3- 0.5 una relación media y de 0.5 a 1 una relación muy fuerte.

En general presentamos correlaciones débiles, si observamos el listado de coeficientes únicamente contamos con dos relaciones que se consideran mediante fuertes con un coeficiente de relación de 4, una de ellas con la variable Bilirubin y una relación negativa con la variable N_Days.


- ¿Con los datos que se tienen se puede predecir correctamente esa variable?

Es probable, pero considero que al no presentar relaciones fuertes, tendríamos que recurrir a alguna técnica avanzada para mejorar el modelo. 

- Si tuviera que seleccionar un modelo para hacer las predicciones, ¿cuál sería?

Considerando que estamos ante un problema de clasificación y que no contamos con correlaciones lo suficientemente fuertes, optaría por trabajar un modelo de regresión logística o un arbol de decisión. 

# PARTE II: MANIPULACIÓN Y TRATAMIENTO.

Paso 1. Tratamiento de datos nulos.
 - Trate los datos nulos acorde a lo que se ha visto previamente en el curso.

 Después de listar los datos nulos, se puede observar que existen columnas cuyo número de nulos es bajo,  por lo tanto, verificaremos las columnas que no exceden el 5% recomendado para realizar la eliminación. 
Posterior a esto, si aún existen datos nulos, recordaremos los gráficos de distribución y boxplot, para determinar cuál de la medida de tendencia central es la más óptima para imputar, todas las columnas objeto será imputada con la moda.   

Trataremos las columnas 'Drug','Ascites','Hepatomegaly', y 'Spiders', realizando la imputación con la moda. 

Recordando los graficos de distribucion y los boxplot mostrado anteriormente, estas columnas presentan sesgo positivo, los cual nos indica que lo mas viable es imputar con la medida de tendencia central "mediana"


 - Muestre la cantidad de datos nulos antes y después del tratamiento.

 Paso 2. Tratamiento de anomalías.
- Enfrente los datos extremos y las anomalías.
- Es libre de utilizar los métodos que prefiera, trate de no perder muchos datos (se tiene un dataset pequeño).



Tomando en consideración los coeficientes de dispersión analizados anteriormente, excluiremos las columnas que superen "Bilirubin", "Alk_Phos" y "Copper" debido a que su coeficiente de variación es alto y podrían representar una perdida significativa de datos, considerando de tratar los nulos nuestra data cuenta con 399 datos, considero que es un número pequeño.

De igual manera excluiremos las columnas   Age, stage y N_Days, debido a que dichas columnas no presentan datos atipicos.

Para tratar estos datos atípicos utilizaremos el método del rango intercuartílico.  



- Muestre las distribuciones tratadas antes del tratamiento.

- Muestre las distribuciones tratadas despues del tratamiento.

Paso 3. Categorización.

 - Seleccione las columnas de tipo objeto candidatas a la categorización.

Como mencione anteriormente, debido a que las columnas objeto parecen contener datos que reflejan el resultado de respuestas de (Y/N), (F/M), que indican categorías binarias, y una columna que presenta 3 valores para el estado, todas son candidatas a la categorización. 

 - Realice la categorización.

  - Muestre la información del dataframe para demostrar que se realizó con éxito.

Paso 4. Tratamiento de incosistencias.
 - En la categorización anterior, ¿hay inconsistencias?


En el vistazo previo, podíamos detectar en muchas de ellas una existía una categoría NAN que hace referencia a aquellas celdas nulas, al contar con valores que reflejan una pertenencia clara a dos grupos diferentes, el set de datos no presentaba anomalías a simple vista, ahora que tratamos los datos nulos será más fácil identificar si alguna columna presenta alguna anomalía.  


- Revise las posibles opciones de cada columna categorizada, si encuentra alguna inconsistencia hay que tratarla.

- Puede emplear gráficas de conteos para validar si una categoría se podría considerar como inconsistente.

Ahora que analizamos las columnas, como mencione anteriormente, fue muy fácil descartar las anomalías, ya que las columnas tienen valores binarios fáciles de detectar; sin embargo, existe una columna llamada "Edema" en la cual podemos detectar valores de (Y/N) que claramente dan a entender que son valores de Si y No en inglés, dentro de esta columna identificamos una tercera clasificación (S) que podría ser que sea un error al ingresar las respuestas (Y) como (S) como si se estuvieran capturando en español. 


Así que bajo esta hipótesis mapearemos los datos que contienen el valor (S) y cambiaremos por la respectiva (Y), para tener consistencia en la informacion.  

Paso 5. Conversión a numérico.

 - Ya que tiene un dataframe sin datos nulos, sin inconsistencias y sin anomalías, hay que convertirlo a numérico.
 - Emplee un tipo de encoding adecuado a cada columna.

En este caso considero que es más apropiado utilizar LabelEncoder, ya que las categorías no presentan orden, la mayoría son categorías binarias, por esa razón utilizaremos este encoder para realizar la conversión a numérico. 

 - Construya un nuevo dataframe completamente numérico (incluyendo "Status", este debe ser forzosamente mediante LabelEncoding).

  - Muestre la información del nuevo dataframe.

 # PARTE III: MACHINE LEARNING.

   Paso 1. División de los datos.
  - Divide las columnas en la variable "X" y la variable "y".

   - Muestree los datos en dos: entrenamiento y pruebas. La proporción de cada muestra queda a decisión suya.
   - Utilice una semilla para que los resultados puedan ser reproducibles.

   Paso 2. Abordaje mediante modelo simple.
  - Importe un modelo simple de ML, puede ser KNN, Regresión Logística o un árbol de decisión.

  - Entrene el modelo con los datos y realice las predicciones con la muestra de pruebas.
  - Muestre los resultados con f1_score, accuracy_score y classification_report.

 - Analizando el reporte de clasificación, ¿qué tal se desempeñó su modelo?


El modelo de clasificación muestra un rendimiento general aceptable, con un F1 Score y Accuracy Score de aproximadamente 0.7115. Aunque logra predecir efectivamente, al observar el reporte podemos ver que se predice con exito la clase mayoritaria (Clase 0), pero enfrenta desafíos significativos en las clases minoritarias (Clase 1 y Clase 2).

El accuracy global es del 71%, indicando un buen rendimiento en general, pero el reporte de clasificación, nos indica que podria tener problemas por las clases desbalanceadas.

Paso 3. Mejorando el modelo.

- Emplee GridSearchCV para encontrar los mejores hiperparámetros para su modelo.
- Valide con varias opciones.

- Si su modelo no logra mejorar mucho, no se preocupe, es parte del aprendizaje.

El modelo mostró mejoras en relación con las métricas F1 Score y Accuracy aunque fue poco lo que mejoro, si demostro una mejora en comparacion al modelo anterior.

Paso 4. Ensambles.
 - Utilice el ensamble de VotingClassifier para mejorar el rendimiento.
 - Seleccione al menos 4 modelos simples diferentes y úselos dentro del ensamble (Stacking).
 - Entrene el meta-modelo y valide su rendimiento con f1_score, accuracy_score y classification_report.

 - Analizando el classification_report, ¿qué tal se desempeñó el modelo?

En general, el F1-score y la precisión ponderada son moderados, lo que indica un rendimiento aceptable del modelo.
Si realizamos una comparativa con el modelo simple, aunque ambos modelos tienen una precisión similar en términos de accuracy, el F1-score estos valores siguen siendo mas altos en el modelo simple, con los hiperparametros seleccionados gracias al gridserch, el primer modelo parece tener un rendimiento ligeramente mejor.

Considero que el modelo no reflejo mejoras considerables. 

Paso 5. Modelo supremo.
 - Con los resultados del paso 4 y 5, determine qué camino seguirá: tomar un modelo y mejorarlo o usar el meta-modelo y mejorarlo.
 - Mejore su modelo hasta el máximo, para eso se recomienda utilizar una Pipeline (puede ser con Pipeline o make_pipeline).
 - Dependiendo del modelo que haya seleccionado, debe buscar mejores hiperparámetros, escalar, normalizar, estandarizar o hacer cambios importantes en los datos (como seleccionar únicamente las variables de mayor correlación), también puede emplear PCA para reducir dimensionalidad.
 - El objetivo es que el modelo generado en este paso sea superior a los modelos del paso 4 y 5.
 - Para este paso también puede utilizar las SVM, RandomForest y Redes Neuronales Artificiales (SKLearn).

  # Etapa 4. Presentación del proyecto.
 - Emplee PCA con las columnas de la variable X (con los datos completos) y reduzca su dimensionalidad a 2.
 - Muestre un gráfico de dispersión entre esas dos características PCA y colorice con la columna "Status". Para esto puede construir un nuevo dataframe con las 2 columnas obtenidas por PCA y añadiendo la columna "Status" antes de la transformación.
  - Analice si los grupos se pueden separar dentro de ese gráfico.


Se denota que hace falta la presencia de una de las clases, es probable que este efecto sea párate de la reducción de dimensionalidad, al realizarla se pierden datos y tal vez por ese motivo ya no hay los suficientes para predecir o tal vez cometí un error en algún paso anterior. Pero ya que los puntos están superpuestos, considero que no es posible visualizar la separación.
