import streamlit as st

from ingest import extract_pages
from retrieve import DEFAULT_COLLECTION, search

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

    with st.sidebar:
        st.subheader("Страницы")
        page_from = st.number_input(
            "С страницы",
            min_value=1,
            value=None,
            step=1,
            placeholder="любая",
        )
        page_to = st.number_input(
            "По страницу",
            min_value=1,
            value=None,
            step=1,
            placeholder="любая",
        )
        min_score = st.slider(
            "Минимальный score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
        )

    query = st.text_input("Введите запрос к документации:")

    if st.button("Найти") and query:
        page_range = None
        if page_from is not None and page_to is not None:
            page_range = (int(page_from), int(page_to))
        score_floor = min_score if min_score > 0.0 else None
        with st.spinner("Ищем в базе знаний..."):
            try:
                results = search(
                    query,
                    DEFAULT_COLLECTION,
                    pages=page_range,
                    min_score=score_floor,
                )
            except ValueError as exc:
                st.error(str(exc))
                results = []

        if not results:
            st.info(
                "Ничего не найдено. Сначала проиндексируйте руководство: "
                "`python ingest.py manual.pdf --index`."
            )
        else:
            for result in results:
                with st.expander(
                    f"Страница {result.page_no} · score {result.score:.3f}"
                ):
                    st.write(result.text)


if __name__ == "__main__":
    main()
