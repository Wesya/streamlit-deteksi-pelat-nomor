# app.py (dengan perbaikan konversi gambar)
import streamlit as st
import cv2
import os
import base64
import numpy as np

from detection_4 import detect_number_plates, recognize_number_plates, extract_tax_info, validate_tax
from io import BytesIO
from ultralytics import YOLO
from easyocr import Reader

st.set_page_config(page_title="Deteksi dan Validasi Status Pajak Kendaraan", layout="wide")
st.title('Deteksi Plat Nomor & Validasi Pajak')
st.markdown("---")

# Fungsi untuk mengonversi gambar numpy array ke base64
def image_to_base64(image_array):
    # Konversi array gambar ke format yang bisa diencode
    is_success, buffer = cv2.imencode(".jpg", image_array)
    if is_success:
        return base64.b64encode(buffer).decode("utf-8")
    return ""

# CSS untuk mengatur ukuran gambar
st.markdown(
    """
    <style>
    .fixed-image {
        max-height: 400px;
        width: auto;
        object-fit: contain;
        margin: 0 auto;
        display: block;
    }
    .result-container {
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Unggah Gambar Plat Nomor", type=["jpg", "jpeg", "png"])
upload_path = "uploads"

if uploaded_file is not None:
    # Simpan gambar
    image_path = os.path.join(upload_path, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Memproses gambar..."):
        # Memanggil model YOLO dan EasyOCR
        model = YOLO("best.pt")
        reader = Reader(['en'], gpu=True)
        
        # Baca gambar
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Konversi gambar ke base64
        image_base64 = image_to_base64(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        
        # Gunakan layout kolom
        col1, col2 = st.columns([2, 3])
        
        # Kolom kiri: gambar asli dengan ukuran tetap
        with col1:
            st.subheader("Gambar yang Diunggah")
            if image_base64:
                st.markdown(
                    f'<img src="data:image/jpeg;base64,{image_base64}" class="fixed-image">',
                    unsafe_allow_html=True
                )
            else:
                st.image(image_rgb, use_column_width=True)
        
        # Kolom kanan: hasil deteksi
        with col2:
            st.subheader("Hasil Deteksi")
            number_plate_list, _= detect_number_plates(image.copy(), model)
            if number_plate_list:
                number_plate_list = recognize_number_plates(image_path, reader, number_plate_list)
                
                with st.container():
                    for i, (box, text) in enumerate(number_plate_list):
                        st.divider()
                        st.subheader(f"Deteksi #{i+1}")
                        
                        cropped_plate = image_rgb[box[1]:box[3], box[0]:box[2]]
                        st.image(cropped_plate, caption="Area Plat Terdeteksi", width=300)
                        
                        st.code(f"Hasil Scan OCR : {text}", language="text")
                        
                        tax_month, tax_year = extract_tax_info(text)
                        
                        if tax_month and tax_year:
                            status, masa_berlaku = validate_tax(tax_month, tax_year)
                            
                            col_a, col_b = st.columns(2)
                            col_a.metric("Bulan Pajak", tax_month)
                            col_b.metric("Tahun Pajak", tax_year)
                            
                            st.metric("Masa Berlaku", masa_berlaku)
                            
                            if status == "AKTIF":
                                st.success(f"✅ Status Pajak: {status}")
                            elif status == "KADALUARSA":
                                st.error(f"❌ Status Pajak: {status}")
                            else:
                                st.warning(f"⚠️ {status}")
                        else:
                            st.warning("Informasi pajak tidak terdeteksi pada plat nomor")
            else:
                st.warning("🚫 Tidak terdapat plat nomor terdeteksi pada gambar.")
else:
    st.info("Silakan unggah gambar plat nomor kendaraan untuk memulai")