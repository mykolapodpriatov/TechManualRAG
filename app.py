import streamlit as st

def main():
    st.set_page_config(page_title="TechManualRAG", layout="wide")
    st.title("TechManualRAG: Инженерный Поиск")
    st.markdown("Система мультимодальной обработки технических руководств.")
    
    uploaded_file = st.file_uploader("Загрузить PDF руководство", type=["pdf"])
    
    if uploaded_file is not None:
        st.success("Файл загружен. В разработке: Обработка...")
        # TODO: Извлечение текста, кропов изображений и таблиц
        # TODO: Индексация в Qdrant
        
    query = st.text_input("Введите запрос к документации:")
    
    if st.button("Найти") and query:
        with st.spinner("Ищем в базе знаний..."):
            # TODO: Поиск по текстовым и визуальным эмбеддингам
            pass

if __name__ == "__main__":
    main()
