"""Secciones informativas; leen los artefactos del entrenamiento sin alterarlos."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st


MODEL_DIRECTORY = Path(__file__).resolve().parent.parent / "modelos"


def read_json(filename):
    path = MODEL_DIRECTORY / filename
    if not path.is_file():
        st.info(f"Todavía no está disponible {filename} en la carpeta modelos.")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def section_intro(number, title, description):
    st.markdown(
        f'<div class="section-intro"><span class="section-number">{number}</span>'
        f'<div><h2>{title}</h2><p>{description}</p></div></div>',
        unsafe_allow_html=True,
    )


def render_data():
    section_intro("01", "Un conjunto de escenas reales", "D-Fire · Dos clases anotadas: humo y fuego.")
    st.write("El detector aprende con imágenes y cajas delimitadoras. Cada anotación identifica "
             "la clase, el centro y el tamaño de una región; una imagen puede contener ambas clases "
             "o no contener ninguna.")
    report_path = MODEL_DIRECTORY / "reporte_limpieza.csv"
    if report_path.is_file():
        report = pd.read_csv(report_path)
        counts = dict(zip(report["paso"], report["cantidad"]))
        partitions = {
            "Entrenamiento": counts.get("Imágenes finales de entrenamiento", 0),
            "Validación": counts.get("Imágenes finales de validación", 0),
            "Prueba": counts.get("Imágenes finales de prueba", 0),
        }
        columns = st.columns(4)
        columns[0].metric("Imágenes originales", f'{counts.get("Imágenes encontradas en el dataset original", 0):,}')
        for column, (label, count) in zip(columns[1:], partitions.items()):
            column.metric(label, f"{count:,}")
        st.caption("Conteos del reporte de limpieza exportado. La validación se utiliza para elegir el modelo "
                   "y los umbrales; prueba se reserva para la evaluación final.")
    st.divider()
    section_intro("02", "Preparar antes de entrenar", "Calidad de imagen, etiquetas consistentes y separación entre conjuntos.")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Imágenes y etiquetas**")
        st.write("Se revisan archivos ilegibles, imágenes pequeñas o casi uniformes y relaciones de aspecto extremas. "
                 "Las cajas se recortan al borde y se eliminan etiquetas inválidas o demasiado pequeñas.")
        st.markdown("**Tamaños de trabajo**")
        st.write("Las imágenes limpias se guardan a 384 × 384 píxeles. La red recibe entradas RGB de 320 × 320.")
    with right:
        st.markdown("**Similitud y fuga de información**")
        st.write("Un hash perceptual de 256 bits permite buscar duplicados y escenas similares. "
                 "La división de entrenamiento y validación mantiene juntas las imágenes de un mismo grupo.")
        st.markdown("**Imágenes sin objetos**")
        st.write("Se conservan ejemplos negativos para aprender el fondo, con un máximo del 30 % en entrenamiento.")
    st.divider()
    section_intro("03", "Revisión visual de las imágenes", "Ejemplos generados durante la auditoría del dataset.")
    duplicate_tab, leakage_tab = st.tabs(["Duplicados", "Escenas similares entre conjuntos"])
    for tab, filename, caption in (
        (duplicate_tab, "ejemplos_duplicados.jpg", "Pares identificados como duplicados o casi duplicados durante la limpieza."),
        (leakage_tab, "ejemplos_fuga.jpg", "Comparación entre prueba y escenas retiradas de entrenamiento o validación por similitud."),
    ):
        with tab:
            image_path = MODEL_DIRECTORY / filename
            if image_path.is_file():
                with st.container(height=560, border=False):
                    st.image(str(image_path), width="stretch")
                st.caption(caption)
            else:
                st.info("Este montaje todavía no está disponible en modelos/.")
    st.caption("Estos montajes muestran la limpieza de datos, no predicciones del detector. "
               "El dataset completo de entrenamiento no está incluido en esta carpeta local.")
    if report_path.is_file():
        with st.expander("Ver el reporte completo de limpieza"):
            st.dataframe(report, hide_index=True, width="stretch")


def render_modeling():
    model_config = read_json("detector_config.json")
    if model_config is None:
        return
    section_intro("01", "Una CNN entrenada desde cero", "Detector de una etapa inspirado en YOLO, con tres anclas por celda.")
    st.write("La red aprende a localizar y distinguir fuego y humo a partir de las anotaciones del dataset. "
             "El detector no utiliza pesos preentrenados. El asistente de preguntas y respuestas es un modelo separado.")
    columns = st.columns(3)
    columns[0].metric("Entrada RGB", f'{model_config["tam_imagen"]} × {model_config["tam_imagen"]}')
    columns[1].metric("Cuadrícula de salida", f'{model_config["grid"]} × {model_config["grid"]}')
    columns[2].metric("Anclas por celda", len(model_config["anclas"]))
    st.markdown('<div class="model-flow"><span>Imagen RGB</span><b>→</b><span>Bloques convolucionales</span>'
                '<b>→</b><span>Características + detalle</span><b>→</b><span>Cajas y clases</span></div>',
                unsafe_allow_html=True)
    architecture, predictions = st.columns([1.1, 1], gap="large")
    with architecture:
        st.markdown("**Extracción de características**")
        st.write("Convoluciones, normalización por lotes y LeakyReLU reducen la resolución de "
                 "320 → 160 → 80 → 40 → 20 píxeles por lado. Los bloques residuales facilitan "
                 "la propagación de información. Una conexión de detalle une características de 40 × 40 con la salida de 20 × 20.")
    with predictions:
        st.markdown("**Qué predice cada caja**")
        st.code("[tx, ty, tw, th, objeto, humo, fuego]", language="text")
        st.write("Con 20 × 20 celdas y tres anclas, la red produce 1 200 propuestas. "
                 "La confianza combina objetividad y puntuación de clase; NMS filtra las cajas repetidas.")
    st.divider()
    section_intro("02", "El proceso de aprendizaje", "Aumentos de datos, pérdida de detección y selección en validación.")
    st.dataframe(pd.DataFrame([
        ("Optimizador", "Adam · tasa máxima 0.001"),
        ("Duración configurada", "Hasta 60 épocas · lote de 16 imágenes"),
        ("Plan de aprendizaje", "3 épocas de calentamiento y descenso cosenoidal"),
        ("Parada temprana", "Paciencia de 12 épocas sin mejora"),
        ("Selección del checkpoint", "Mayor mAP50 en validación"),
        ("Aumentos", "Recorte, volteo horizontal, brillo, contraste, saturación y tono"),
        ("Precisión numérica", "Entrenamiento mixto; exportación float32"),
    ], columns=["Componente", "Configuración del notebook"]), hide_index=True, width="stretch")
    st.caption("Son los valores configurados en el código. Los archivos locales no incluyen el historial "
               "para confirmar cuántas épocas se ejecutaron.")
    st.markdown("**Una pérdida con cuatro componentes**")
    st.write("La función de pérdida combina localización, objetividad, fondo y clase. "
             "Una máscara evita penalizar como fondo las propuestas que ya se solapan suficientemente "
             "con una caja real, aunque no sean su ancla responsable.")
    st.divider()
    section_intro("03", "Del modelo a la aplicación", "Inferencia por imagen, seguimiento temporal y explicación.")
    st.write("El archivo .keras guarda la red y sus pesos. Su JSON contiene clases, anclas y tamaño de entrada. "
             "En video se confirman detecciones en al menos 4 de los últimos 8 fotogramas analizados. "
             "El subtipo de humo se estima por color y el asistente responde usando el contexto del evento.")
    st.caption("Las reglas temporales y el análisis de color son posteriores a la CNN; "
               "no son clases adicionales aprendidas con D-Fire.")
    with st.expander("Configuración del modelo exportado"):
        st.json(model_config)


def render_class_results(metrics, class_name):
    precision = metrics[f"precision_imagen_{class_name}"]
    recall = metrics[f"recall_imagen_{class_name}"]
    ap = metrics[f"AP50_{class_name}"]
    st.subheader(class_name.capitalize())
    left, right = st.columns(2)
    left.metric("AP50", f"{ap:.2%}")
    right.metric("F1-score por imagen", f'{metrics[f"f1_imagen_{class_name}"]:.2%}')
    left.metric("Precisión por imagen", f"{precision:.2%}")
    right.metric("Recall por imagen", f"{recall:.2%}")
    st.markdown(f"**Presencia de {class_name}.** De cada 100 imágenes donde el modelo anuncia "
                f"{class_name}, aproximadamente **{precision * 100:.1f} realmente lo contienen**. "
                f"De cada 100 imágenes que sí contienen {class_name}, reconoce aproximadamente "
                f"**{recall * 100:.1f}** y omite **{(1 - recall) * 100:.1f}**.")
    st.caption("Es una lectura proporcional de precisión y recall, no un conteo de 100 imágenes evaluadas. "
               "Estas cifras no describen la calidad de la ubicación de las cajas.")
    st.write(f"**Localización.** El AP50 de {ap:.2%} resume la curva precisión-recall de las cajas "
             "de esta clase con IoU mínimo de 0.50. No significa que ese porcentaje de imágenes esté correcto.")


def render_results():
    metrics = read_json("metricas_test.json")
    if metrics is None:
        return
    section_intro("01", "Evaluación del modelo exportado", "Resultados guardados sobre el conjunto de prueba, separados de validación.")
    columns = st.columns(2)
    columns[0].metric("mAP50", f'{metrics["mAP50"]:.2%}')
    columns[1].metric("Exactitud del evento por imagen", f'{metrics["exactitud_evento_imagen"]:.2%}')
    st.write("El **mAP50** resume la localización de humo y fuego. La **exactitud del evento** "
             "mide cuántas imágenes se asignaron correctamente a una de cuatro categorías: "
             "normal, solo humo, solo fuego o fuego y humo.")
    report_path = MODEL_DIRECTORY / "reporte_limpieza.csv"
    if report_path.is_file():
        report = pd.read_csv(report_path)
        test_counts = report.loc[report["paso"] == "Imágenes finales de prueba", "cantidad"]
        if not test_counts.empty:
            st.caption(f"El reporte de limpieza registra {int(test_counts.iloc[0]):,} imágenes de prueba. "
                       "Los valores de esta página se leen de metricas_test.json; no se recalculan al usar el detector.")
    st.divider()
    section_intro("02", "Resultados de fuego y humo", "Localización y presencia de cada clase en prueba.")
    for name in ("fuego", "humo"):
        if f"f1_imagen_{name}" not in metrics:
            precision = metrics[f"precision_imagen_{name}"]
            recall = metrics[f"recall_imagen_{name}"]
            metrics[f"f1_imagen_{name}"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fire_column, smoke_column = st.columns(2, gap="large")
    with fire_column:
        render_class_results(metrics, "fuego")
    with smoke_column:
        render_class_results(metrics, "humo")
    rows = [{
        "Clase": name.capitalize(),
        "AP50": f'{metrics[f"AP50_{name}"]:.2%}',
        "Precisión por imagen": f'{metrics[f"precision_imagen_{name}"]:.2%}',
        "Recall por imagen": f'{metrics[f"recall_imagen_{name}"]:.2%}',
        "F1-score por imagen": f'{metrics[f"f1_imagen_{name}"]:.2%}',
    } for name in ("fuego", "humo")]
    st.markdown("**Comparación de los resultados de prueba**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    comparison = pd.DataFrame({
        name.capitalize(): [metrics[f"AP50_{name}"] * 100,
                            metrics[f"precision_imagen_{name}"] * 100,
                            metrics[f"recall_imagen_{name}"] * 100,
                            metrics[f"f1_imagen_{name}"] * 100]
        for name in ("fuego", "humo")
    }, index=pd.Index(["AP50", "Precisión por imagen", "Recall por imagen", "F1-score por imagen"], name="Métrica"))
    st.bar_chart(comparison, color=["#d49a7f", "#b5c4a1"], stack=False,
                 y_label="Resultado (%)", height=300)
    st.subheader("Matriz de confusión por imagen")
    if "matriz_confusion" in metrics:
        categories = metrics["categorias_matriz"]
        matrix = pd.DataFrame(metrics["matriz_confusion"],
                              index=pd.Index(categories, name="Evento real"), columns=categories)
        st.caption("Filas = evento real. Columnas = evento predicho. Cada celda cuenta imágenes de prueba.")
        st.dataframe(matrix, width="stretch")
        if "umbrales_imagen" in metrics:
            thresholds = ", ".join(f"{name} {value:.2f}" for name, value in metrics["umbrales_imagen"].items())
            st.caption(f"Umbrales de confianza usados en prueba: {thresholds}.")
    else:
        st.info("Esta exportación no incluye la matriz. La evaluación del notebook la guarda en metricas_test.json.")

    st.subheader("Loss (pérdida)")
    if "loss" in metrics:
        st.metric("Pérdida media por imagen de prueba", f'{metrics["loss"]:.4f}')
        components = {"Caja": "perd_caja", "Objeto": "perd_obj", "Fondo": "perd_noobj", "Clase": "perd_clase"}
        st.dataframe(pd.DataFrame([{label: metrics[key] for label, key in components.items()}]),
                     hide_index=True, width="stretch")
    history_path = MODEL_DIRECTORY / "historial.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        history["Época"] = history["epoch"] + 1
        st.line_chart(history.set_index("Época")[["loss", "val_loss"]].rename(
            columns={"loss": "Entrenamiento", "val_loss": "Validación"}))
        with st.expander("Pérdida y componentes por época"):
            fields = [c for c in history if c in ("loss", "val_loss") or c.startswith(("perd_", "val_perd_"))]
            st.dataframe(history[["Época", *fields]], hide_index=True, width="stretch")
    else:
        st.caption("Las curvas de pérdida de entrenamiento y validación requieren el historial por época del entrenamiento original.")
    st.divider()
    section_intro("03", "Qué mide cada métrica", "Localizar una caja y reconocer la presencia de una clase son tareas diferentes.")
    localization, presence = st.columns(2, gap="large")
    with localization:
        st.markdown("**IoU · Intersección sobre unión**")
        st.write("Compara la caja predicha con la caja real: área compartida dividida entre el área "
                 "que ocupan juntas. Para AP50, una detección debe tener la clase correcta y una IoU "
                 "de al menos 0.50. Una caja real no puede contarse dos veces como acierto.")
        st.markdown("**AP50 · Precisión promedio por clase**")
        st.write("Se ordenan las detecciones por confianza y se calcula el área bajo la curva "
                 "precisión-recall con IoU mínimo de 0.50. Evalúa conjuntamente la localización y "
                 "la clase. Se obtiene un AP50 para fuego y otro para humo.")
        st.markdown("**mAP50 · Promedio de los AP50**")
        st.write("Es la media aritmética de AP50 de fuego y AP50 de humo, dando el mismo peso a ambas clases. "
                 "Aquí se evalúa IoU 0.50; no es la métrica mAP50:95.")
    with presence:
        st.markdown("**Precisión por imagen · ¿Cuántos avisos fueron correctos?**")
        st.write("De las imágenes donde el detector anuncia una clase, qué proporción realmente la contiene. "
                 "Una precisión menor implica más avisos de esa clase en imágenes que no la contienen.")
        st.markdown("**Recall por imagen · ¿Cuántos casos reales encontró?**")
        st.write("De las imágenes que realmente contienen una clase, qué proporción fue reconocida. "
                 "Un recall menor implica más imágenes con esa clase que pasan sin detectar su presencia.")
        st.markdown("**Exactitud del evento por imagen**")
        st.write("Proporción de imágenes cuya categoría completa coincide con la real. "
                 "Si una imagen contiene fuego y humo, reconocer únicamente fuego no basta para acertar el evento completo.")
        st.markdown("**F1-score por imagen**")
        st.write("Media armónica de la precisión y el recall de una clase. Baja cuando se pierden casos reales o aumentan las falsas alarmas.")
    with st.expander("Fórmulas y criterios para explicar la evaluación"):
        st.latex(r"\mathrm{IoU}=\frac{\mathrm{area\ de\ interseccion}}{\mathrm{area\ de\ union}}")
        st.latex(r"\mathrm{Precision}=\frac{VP}{VP+FP}\qquad\mathrm{Recall}=\frac{VP}{VP+FN}")
        st.latex(r"F_1=\frac{2\,\mathrm{Precision}\,\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}")
        st.latex(r"\mathrm{mAP50}=\frac{\mathrm{AP50}_{fuego}+\mathrm{AP50}_{humo}}{2}")
        st.latex(r"\mathrm{Exactitud}=\frac{\mathrm{imagenes\ con\ evento\ correcto}}{\mathrm{total\ de\ imagenes}}")
        st.caption("VP: verdaderos positivos; FP: falsos positivos; FN: falsos negativos. "
                   "En las métricas por imagen, estos términos se cuentan por presencia de clase, no por cajas. "
                   "AP50 usa el conjunto de detecciones y su curva, en lugar de un único umbral operativo.")
    st.divider()
    section_intro("04", "Cómo interpretar los resultados", "Una lectura de los valores guardados para la exposición.")
    fire_recall = metrics["recall_imagen_fuego"]
    smoke_recall = metrics["recall_imagen_humo"]
    st.markdown(
        f"> En el conjunto de prueba, el detector obtuvo un **mAP50 de {metrics['mAP50']:.2%}**. "
        f"Para fuego, la precisión por imagen fue **{metrics['precision_imagen_fuego']:.2%}** y el recall "
        f"**{fire_recall:.2%}**. Para humo, fueron **{metrics['precision_imagen_humo']:.2%}** y "
        f"**{smoke_recall:.2%}**, respectivamente. El reconocimiento de presencia de fuego tuvo mayor recall, "
        "mientras que humo obtuvo un AP50 superior en localización. Esto muestra por qué conviene evaluar "
        "la presencia de las clases y la calidad de sus cajas por separado."
    )
    st.write(f"**Omisiones por clase:** el complemento del recall es {(1 - fire_recall):.2%} para fuego "
             f"y {(1 - smoke_recall):.2%} para humo. Son proporciones de las imágenes que realmente contienen "
             "cada clase, no porcentajes sobre todas las imágenes ni tasas de falsas alarmas.")
    st.divider()
    section_intro("05", "El efecto del umbral", "Barrido realizado en validación, separado de la evaluación de prueba.")
    sweep_path = MODEL_DIRECTORY / "barrido_umbrales_val.csv"
    if sweep_path.is_file():
        sweep = pd.read_csv(sweep_path)
        selected_class = st.selectbox("Clase para explorar", ["humo", "fuego"])
        selected_rows = sweep[sweep["clase"] == selected_class]
        chart = selected_rows.set_index("umbral")[["recall_imagen", "precision_imagen", "falsas_normales"]]
        chart.columns = ["Recall por imagen", "Precisión por imagen", "Falsas alarmas en normales"]
        st.line_chart(chart, color=["#b5c4a1", "#d49a7f", "#b5a5cc"], height=320)
        st.caption("Proporciones entre 0 y 1. Al bajar el umbral suelen aumentar la sensibilidad y las falsas alarmas. "
                   "Explorar esta gráfica no modifica los umbrales del detector.")
        with st.expander("Consultar los valores del barrido"):
            st.dataframe(selected_rows, hide_index=True, width="stretch")
    else:
        st.info("El barrido de umbrales todavía no está disponible.")
    st.divider()
    section_intro("06", "Alcance de estos resultados", "Las métricas de imágenes y la experiencia en video son evaluaciones distintas.")
    st.write("Conviene probar escenas con fuego, humo, ambos y escenas normales, además de niebla, "
             "reflejos y atardeceres. Estos archivos no contienen mediciones de latencia, "
             "tiempo hasta la alerta ni falsas alarmas acumuladas en videos.")


def render_assistant_results():
    st.subheader("Evaluación del asistente en español")
    report = read_json("metricas_asistente.json")
    if report is None:
        st.code("python evaluar_asistente.py", language="bash")
        return
    st.caption(f"Modelo evaluado · {report['modelo']}  |  Umbral de extracción · {report['umbral']}")
    total = report["casos"]
    correct = report["respuestas_correctas"]
    abstained = report["abstenciones_correctas"]
    st.write(f"El asistente resolvió **{correct + abstained} de {total} casos**. "
             f"Dio {correct} respuestas correctas y se abstuvo correctamente en {abstained} casos.")
    descriptions = [
        ("Accuracy", "accuracy", "Proporción de casos resueltos, incluidas las abstenciones correctas."),
        ("Precision", "precision", "Proporción de las preguntas contestadas cuya respuesta fue correcta."),
        ("Recall", "recall", "Proporción de las preguntas con respuesta esperada que el asistente contestó correctamente."),
        ("F1", "f1", "Media armónica de precision y recall para las respuestas del asistente."),
    ]
    for column, (label, key, help_text) in zip(st.columns(4), descriptions):
        value = report[key]
        column.metric(label, f"{value:.2%}" if value is not None else "—", help=help_text)
    st.caption(f"{report['contestadas']} preguntas contestadas · {report['con_respuesta']} con respuesta esperada · "
               f"{total - report['con_respuesta']} sin evidencia. La evaluación incluye el aviso inicial del monitor.")


def render_project_section(section):
    {"Datos": render_data, "Modelado": render_modeling, "Resultados": render_results}[section]()
