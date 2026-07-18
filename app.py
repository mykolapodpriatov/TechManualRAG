import streamlit as st

from ingest import extract_pages

PREVIEW_CHARS = 2000

def main():
    st.set_page_config(page_title="TechManualRAG", layout="wide")
    st.title("TechManualRAG: Инженерный Поиск")
    st.markdown("Система мультимодальной обработки технических руководств.")

    uploaded_file = st.file_uploader("Загрузить PDF руководство", type=["pdf"])

    if uploaded_file is not None:
        try:
            pages = extract_pages(uploaded_file.getvalue())
        except ValueError as exc:
            st.error(f"Не удалось обработать PDF: {exc}")
        else:
            st.success(f"Файл загружен. Извлечено страниц: {len(pages)}.")
            for page in pages:
                with st.expander(f"Страница {page.page_no}"):
                    if page.text:
                        st.text(page.text[:PREVIEW_CHARS])
                    else:
                        st.caption("(на этой странице нет извлекаемого текста)")
        # TODO: Кропы изображений и таблиц
        # TODO: Индексация в Qdrant

    query = st.text_input("Введите запрос к документации:")

    if st.button("Найти") and query:
        with st.spinner("Ищем в базе знаний..."):
            # TODO: Поиск по текстовым и визуальным эмбеддингам
            pass

if __name__ == "__main__":
    main()
