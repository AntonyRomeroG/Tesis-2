<h1>Proyecto de Tesis</h1>

<h2>Descripción</h2>

<p>Este proyecto de tesis se centra en desarrollar un modelo algorítmico que mejore la confiabilidad de los sensores de bajo costo de material particulado cuya presición se ve afectado por las variables climáticas locales y los cambios estacionales. Por otro lado, el repositorio contiene dos carpetas principales:</p>

<ol>
  <li>Preprocesamiento: Código para la vizualización y limpieza de los datos</li>
  <li>Entrenamiento: Código para el entrenamiento de los modelo</li>
  <li>Modelos: Modelos entrenados</li>
</ol>

<h2>Preprocesamiento</h2>

<p> Tener en cuenta las siguientes consideraciones: </p>

<ol>
  <li> Existen variaciones en el código según los datos de intercomparaciones del lugar que se desee preprocesar </li>
  <li> Cargar los datos de intercomparaciones correspondientes al lugar mencionado en el nombre del archivo del código </li>
</ol>

<h2>Entrenamiento</h2>

<p> Se implementa la siguiente arquitectura: </p>

<img src="https://github.com/AntonyRomeroG/Tesis-2/blob/main/fuente.png" height="70%" width="70%" />

<p> Se empelará un enfoque de calibración monosensor en el lugar fuente. Este enfoque, ajusta un modelo de corrección (modelo base) para un sensor candidato específico (LCS) utilizando datos de un período de mediciones simultáneas con un sensor de referencia coubicado. </p>

<img src="https://github.com/AntonyRomeroG/Tesis-2/blob/main/objetivo.png" height="70%" width="70%" />

<p> Se empelará un enfoque de calibración multisensor en el lugar objetivo. Este enfoque, utiliza los modelos base para corregir la medición PM del sensor objetivo, es decir, el valor medio de las predicciones de los modelos base es añadido como variable predictora </p>

<h2>Modelos</h2>

En el archivo se incluye un enlace donde se pueden ver los modelos base de Lima y los modelos de calibración de Arequipa. Estos se han evaluado utilizando todos los modelos base (_ap_) o solo aquellos del mismo fabricante (_mp_).
