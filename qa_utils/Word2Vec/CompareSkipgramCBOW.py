import streamlit as st
import pandas as pd
import plotly.express as px
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

def compare_skipgram_cbow(sentences, vector_size=150, window=5, min_count=1):
    st.header("🔴 Compare Skip-gram and CBOW Models")

    # 預處理
    tokenized_sentences = [
        simple_preprocess(remove_stopwords(sentence)) for sentence in sentences
    ]
    if not tokenized_sentences or all(len(s) == 0 for s in tokenized_sentences):
        st.error("❌ No valid tokens found. Please input meaningful sentences.")
        return None, None

    # 訓練兩個模型
    skipgram_model = Word2Vec(tokenized_sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4, sg=1)
    cbow_model = Word2Vec(tokenized_sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4, sg=0)

    # 展示輸入句子
    with st.expander("📄 Show Input Sentences", expanded=False):
        for i, s in enumerate(sentences, 1):
            st.markdown(f"**Sentence {i}:** {s}")

    # 輸入要查的 Query Word
    query_word = st.text_input("🔍 Enter a word to explore:", key="compare_query")
    
    if query_word:
        if query_word in skipgram_model.wv and query_word in cbow_model.wv:
            # 顯示兩個模型的 top 5 相似詞
            st.markdown("### ✨ Top 5 Similar Words (Skip-gram)")
            st.table(skipgram_model.wv.most_similar(query_word, topn=5))

            st.markdown("### ✨ Top 5 Similar Words (CBOW)")
            st.table(cbow_model.wv.most_similar(query_word, topn=5))

            # 額外輸入 Compare Word
            st.markdown("---")
            compare_word = st.text_input("🔄 Enter another word to compare similarity:", value="development", key="similarity_query")
            
            if compare_word:
                if compare_word in skipgram_model.wv and compare_word in cbow_model.wv:
                    sim_skipgram = skipgram_model.wv.similarity(query_word, compare_word)
                    sim_cbow = cbow_model.wv.similarity(query_word, compare_word)

                    st.success(f"**Skip-gram similarity '{query_word}' ~ '{compare_word}': {sim_skipgram:.4f}**")
                    st.success(f"**CBOW similarity '{query_word}' ~ '{compare_word}': {sim_cbow:.4f}**")

                    # 🎯 畫出兩個詞的 Skip-gram vs CBOW 2D 比較
                    st.markdown("---")
                    st.subheader("📈 Visualization: Word Embedding 2D Comparison")

                    df = pd.DataFrame({
                        'Model': ['Skip-gram', 'CBOW', 'Skip-gram', 'CBOW'],
                        'Word': [query_word, query_word, compare_word, compare_word],
                        'X': [
                            skipgram_model.wv[query_word][0], cbow_model.wv[query_word][0],
                            skipgram_model.wv[compare_word][0], cbow_model.wv[compare_word][0]
                        ],
                        'Y': [
                            skipgram_model.wv[query_word][1], cbow_model.wv[query_word][1],
                            skipgram_model.wv[compare_word][1], cbow_model.wv[compare_word][1]
                        ]
                    })

                    fig = px.scatter(
                        df, x='X', y='Y', color='Model', hover_name='Word', text='Word',
                        title="Skip-gram vs CBOW Word Embedding Visualization"
                    )
                    fig.update_traces(textposition='top center')
                    fig.update_layout(width=800, height=600)
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.warning(f"⚠️ '{compare_word}' not found in the vocabulary.")
        else:
            st.warning(f"⚠️ '{query_word}' not found in the vocabulary.")

    return None, (skipgram_model, cbow_model)
