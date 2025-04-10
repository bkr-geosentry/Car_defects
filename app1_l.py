import gradio as gr
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from fpdf import FPDF

# Load the trained YOLO model
model = YOLO('best.pt')


# Function to detect car defects
def detect_defects(image):
    results = model(image)
    detections = [(model.names[int(box.cls)], box.xyxy.tolist()) for box in results[0].boxes]
    annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_img), detections

# Function to generate a report
def generate_report(original, processed, defects):
    report_filename = "car_defect_report.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, "Car Defects Detection Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Original Image vs Processed Image", ln=True, align="C")
    pdf.ln(5)
    
    original_path = "original_image.jpg"
    processed_path = "processed_image.jpg"
    
    original.save(original_path)
    processed.save(processed_path)
    
    pdf.image(original_path, x=10, y=50, w=90)
    pdf.image(processed_path, x=110, y=50, w=90)
    
    pdf.ln(100)
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, "Defect Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Detected Defects: {defects}")
    
    pdf.output(report_filename)
    return report_filename

# Custom CSS for styling
custom_css = """
.gradio-container {
    background-color: #E0F7FA !important;
    text-align: center;
    color: #000 !important;
}

h1 {
    text-align: center !important;
    font-size: 28px !important;
    font-weight: bold !important;
    color: #000 !important;
}

#upload-box, #detect-box {
    width: 100% !important;
    height: 350px !important;
    object-fit: contain;
    background-color: #FFFACD !important;
    border-radius: 10px !important;
    padding: 10px !important;
}

#download-report {
    font-weight: bold !important;
    color: #000 !important;
}
"""

# Create Gradio app
with gr.Blocks(css=custom_css) as app:
    gr.Markdown("# 🚗 Car Defects Detection System")
    
    with gr.Row():
        input_img = gr.Image(type="pil", label="Upload a High-Quality Car Image (Visible Defects: Paint Damage, Rust, Dents)", elem_id="upload-box")
        output_img = gr.Image(type="pil", label="Detected Defects", elem_id="detect-box", interactive=False)
    
    detect_btn = gr.Button("Detect Defects")
    report_btn = gr.Button("Generate Report")
    
    report_file = gr.File(label="Download Report", elem_id="download-report")
    
    detected_defects = gr.State()
    
    def process_detection(image):
        annotated, defects = detect_defects(image)
        return annotated, defects

    detect_btn.click(process_detection, inputs=input_img, outputs=[output_img, detected_defects])
    report_btn.click(generate_report, inputs=[input_img, output_img, detected_defects], outputs=report_file)

# Launch the app
app.launch(share=True)
