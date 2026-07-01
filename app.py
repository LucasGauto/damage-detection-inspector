import gradio as gr
from callbacks.detection import classify_detect_parts
from callbacks.classification import classify_selected_part

with gr.Blocks(title="Analisis de Partes y Daños") as demo:
    gr.Markdown("# Analisis de partes y daños")
    gr.Markdown(
        "La aplicacion permite correr deteccion de partes con YOLO y clasificacion de daño "
        "condicionada por parte con el modelo `models/classificator/best.pth`."
    )

    with gr.Tab("Clasificacion de daño"):
        cls_detections_state = gr.State([])

        # Interfaz
        # Fila de Deteccion
        gr.Markdown("#### Deteccion de partes")
        with gr.Row():
            
            with gr.Column(scale=1):
                with gr.Row():
                    classification_image_input = gr.Image(type="pil", label="Imagen del Vehiculo")

            with gr.Column(scale=1):
                classification_image_detected = gr.Image(type="pil", label="Imagen con detecciones", interactive=True, scale=1)

        with gr.Row():
            cls_conf_slider = gr.Slider(minimum=0.00, maximum=1.0, value=0.1, step=0.05, label="Umbral de confianza (deteccion)")
            cls_detect_button = gr.Button("Detectar partes")

        # Fila de Clasificacion
        gr.Markdown("#### Clasificacion de daño")
        with gr.Row():
    
            with gr.Column():
                part_input = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Parte detectada",
                    interactive=True,
                )
                classify_button = gr.Button("Clasificar daño")
                
            with gr.Column():
                classification_summary_output = gr.Textbox(
                    label="Resultado",
                    interactive=False,
                    lines=3,
                )

        with gr.Row():
            classification_scores_output = gr.Label(label="Probabilidades")

        with gr.Row():
            with gr.Column():
                gradcam_s2_output = gr.Image(
                    type="pil",
                    label="Stage 2 (penultima capa)",
                    interactive=True,
                )
            with gr.Column():
                gradcam_s3_output = gr.Image(
                    type="pil",
                    label="Stage 3 (ultima capa)",
                    interactive=True,
                )

        cls_detect_button.click(
            fn=classify_detect_parts,
            inputs=[classification_image_input, cls_conf_slider],
            outputs=[classification_image_detected, cls_detections_state, part_input],
        )

        classify_button.click(
            fn=classify_selected_part,
            inputs=[classification_image_input, part_input, cls_detections_state],
            outputs=[classification_summary_output, classification_scores_output, gradcam_s2_output, gradcam_s3_output],
        )

if __name__ == "__main__":
    demo.launch(debug=True)