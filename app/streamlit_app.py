"""ECG Analyzer — приложение для анализа электрокардиограмм.

Запуск:
    .venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.predict import load_model, load_signal_from_csv, predict_signal
from src.preprocessing import load_config, preprocess, PROJECT_ROOT

st.set_page_config(
    page_title="ECG Analyzer",
    page_icon="❤️",
    layout="wide",
)

# ---------- конфигурация ----------
config = load_config()
CLASS_NAMES = config["model"]["class_names"]
CLASS_LABELS_RU = config["model"]["class_labels_ru"]
LEAD_NAMES = config["signal"]["lead_names"]
THRESHOLD = config["predict"]["threshold"]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

CLASS_EMOJI = {
    "NORM": "✅",
    "MI": "🫀",
    "STTC": "📉",
    "CD": "⚡",
    "HYP": "💪",
}


@st.cache_resource
def get_model():
    """Загружает модель один раз, кэширует между перезапусками интерфейса."""
    return load_model(config=config)


def plot_ecg(signal: np.ndarray, lead_names: list[str]) -> plt.Figure:
    """Рисует 12 отведений ЭКГ в сетке 4×3."""
    fig, axes = plt.subplots(4, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (lead, ax) in enumerate(zip(lead_names, axes)):
        ax.plot(signal[i], linewidth=0.7, color="steelblue")
        ax.set_title(lead, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    return fig


def show_results(probs: dict[str, float], threshold: float) -> None:
    """Выводит вероятности диагнозов в виде карточек и прогресс-баров."""
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    # вероятность нормы
    norm_prob = probs.get("NORM", 0.0)

    # главное заключение
    top_class, top_prob = sorted_probs[0]
    if norm_prob >= threshold:
        st.success(f"**Заключение:** признаков патологии не выявлено (вероятность нормы {norm_prob:.1%})")
    elif top_prob >= threshold:
        st.warning(
            f"**Заключение:** обнаружены признаки — "
            f"**{CLASS_LABELS_RU[top_class]}** ({top_prob:.1%}). "
            f"Рекомендуется консультация врача."
        )
    else:
        st.info("**Заключение:** результат неоднозначен. Рекомендуется консультация врача.")

    # прогресс-бары
    st.subheader("Вероятности диагнозов")
    for name, prob in sorted_probs:
        emoji = CLASS_EMOJI.get(name, "")
        label = f"{emoji} {CLASS_LABELS_RU[name]} ({name})"

        col1, col2, col3 = st.columns([3, 5, 1])
        with col1:
            st.markdown(f"**{label}**")
        with col2:
            st.progress(prob, text=f"{prob:.1%}")
        with col3:
            if prob >= threshold:
                st.markdown("🔴")
            else:
                st.markdown("⚪")


# =================== интерфейс ===================

st.title("❤️ ECG Analyzer")
st.markdown(
    "Учебное приложение для анализа электрокардиограмм "
    "с помощью нейронных сетей."
)
st.warning(
    "⚠️ **Демонстрационная версия.** Результаты не предназначены "
    "для диагностики и не заменяют консультацию врача."
)

# --- боковая панель ---
with st.sidebar:
    st.header("⚙️ Настройки")
    threshold = st.slider(
        "Порог классификации",
        min_value=0.1,
        max_value=0.9,
        value=THRESHOLD,
        step=0.05,
        help="Вероятность выше порога считается положительным диагнозом",
    )

    st.header("📊 О модели")
    model_card_path = PROJECT_ROOT / "models" / "model_card.json"
    if model_card_path.exists():
        with open(model_card_path, encoding="utf-8") as f:
            card = json.load(f)
        st.write(f"**Архитектура:** {card.get('architecture', '—')}")
        st.write(f"**Параметров:** {card.get('num_parameters', 0):,}")
        st.write(f"**Val AUC:** {card.get('val_auc', '—')}")

    st.header("📁 Демо-записи")
    demo_files = sorted(SAMPLES_DIR.glob("demo_*.csv")) if SAMPLES_DIR.exists() else []
    demo_choice = st.selectbox(
        "Или выберите демо-запись:",
        ["—"] + [f.name for f in demo_files],
    )

# --- загрузка файла ---
uploaded_file = st.file_uploader(
    "Загрузите файл ЭКГ (CSV, 12 отведений)",
    type=["csv"],
)

signal = None
source_name = None

if uploaded_file is not None:
    try:
        signal = load_signal_from_csv(uploaded_file)
        source_name = uploaded_file.name
    except Exception as e:
        st.error(f"❌ Ошибка чтения файла: {e}")

elif demo_choice != "—":
    try:
        signal = load_signal_from_csv(SAMPLES_DIR / demo_choice)
        source_name = demo_choice
    except Exception as e:
        st.error(f"❌ Ошибка чтения демо-файла: {e}")

# --- обработка и вывод ---
if signal is not None:
    st.subheader(f"📈 Сигнал ЭКГ — {source_name}")
    fig = plot_ecg(signal, LEAD_NAMES)
    st.pyplot(fig)
    plt.close(fig)

    if st.button("🔍 Анализировать", type="primary", use_container_width=True):
        with st.spinner("Модель анализирует сигнал..."):
            model = get_model()
            probs = predict_signal(signal, model, config)

        st.divider()
        show_results(probs, threshold)

else:
    st.info(
        "👆 Загрузите CSV-файл с ЭКГ или выберите демо-запись в боковой панели."
    )

    # показываем, как выглядит правильный формат
    with st.expander("ℹ️ Какой формат файла ожидается?"):
        st.markdown(
            """
            CSV-файл с **12 столбцами** (отведения) и **1000 строками** (10 секунд при 100 Гц):

            | I | II | III | aVR | aVL | aVF | V1 | V2 | V3 | V4 | V5 | V6 |
            |---|---|---|---|---|---|---|---|---|---|---|---|
            | -0.021 | 0.043 | 0.064 | -0.032 | ... | | | | | | | |
            | ... | | | | | | | | | | | |
            """
        )

st.divider()
st.caption(
    "ECG Analyzer — учебный проект. Классификация по 5 суперклассам PTB-XL: "
    "NORM, MI, STTC, CD, HYP. Не является медицинским изделием."
)