# app.py
import cv2
import easyocr
import numpy as np
import streamlit as st
import json
import re

st.set_page_config(page_title="AI Сканер", layout="centered", page_icon="📷")
st.title("📷 AI Сканер документов")
st.caption("Загрузи фото или сфоткай → текст распознается автоматически")


@st.cache_resource
def load_reader():
    return easyocr.Reader(['ru', 'en'], gpu=False)


reader = load_reader()


def preprocess(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 15, 8)


def parse_text(text):
    data = {"license_plates": [], "dates": [], "contacts": [], "general_text": []}
    plate_pat = re.compile(r'[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}', re.IGNORECASE)
    date_pat = re.compile(r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}')
    contact_pat = re.compile(r'[+]?[\d\s\-()]{7,}')

    for line in text.split('\n'):
        line = line.strip()
        if not line: continue
        if plate_pat.search(line):
            data["license_plates"].extend(plate_pat.findall(line))
        elif date_pat.search(line):
            data["dates"].extend(date_pat.findall(line))
        elif contact_pat.search(line):
            data["contacts"].extend(contact_pat.findall(line))
        else:
            data["general_text"].append(line)
    return data


uploaded = st.file_uploader("📤 Выбери фото или нажми 📷", type=["jpg", "jpeg", "png"],
                            accept_multiple_files=False)

if uploaded:
    img_bytes = uploaded.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("❌ Не удалось открыть изображение")
        st.stop()

    with st.spinner("⏳ ИИ обрабатывает..."):
        enhanced = preprocess(img_bgr)
        results = reader.readtext(enhanced, detail=1)
        filtered = [r for r in results if r[2] > 0.6]
        raw_text = "\n".join([r[1] for r in filtered])
        structured = parse_text(raw_text)

    st.subheader("📝 Распознанный текст")
    st.text_area("", raw_text, height=150, label_visibility="collapsed")

    st.subheader("📊 Структура")
    st.json(structured)

    payload = {"raw_text": raw_text, "structured_data": structured}
    st.download_button("💾 Скачать JSON", json.dumps(payload, ensure_ascii=False, indent=4),
                       file_name="scan_result.json", mime="application/json")