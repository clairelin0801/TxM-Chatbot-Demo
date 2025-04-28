import streamlit as st
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

def plot_cbow_word2vec(sentences, vector_size=150, window=3, min_count=1):
    st.header("🟠 CBOW Word2Vec")

    # Preprocess the sentences
    tokenized_sentences = [simple_preprocess(remove_stopwords(sentence)) for sentence in sentences]
    if not tokenized_sentences or all(len(s) == 0 for s in tokenized_sentences):
        st.error("❌ No valid tokens found. Please input meaningful sentences.")
        return None, None

    # Train a CBOW Word2Vec model
    model = Word2Vec(tokenized_sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4, sg=0)

    # 顯示原始句子
    with st.expander("📄 Show Input Sentences", expanded=False):
        for i, s in enumerate(sentences, 1):
            st.markdown(f"**Sentence {i}:** {s}")

    # Query
    st.markdown("### 🔍 Try a word to find similar words")
    query_word = st.text_input("Enter a word to look up:", key="cbow_query")

    if query_word:
        if query_word in model.wv:
            st.success(f"Top 5 similar words to **{query_word}**:")
            st.table(model.wv.most_similar(query_word, topn=5))
        else:
            st.warning(f"⚠️ '{query_word}' not in vocabulary.")

    return None, model
