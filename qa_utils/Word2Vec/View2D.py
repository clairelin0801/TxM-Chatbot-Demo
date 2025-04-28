import numpy as np
import plotly.graph_objs as go
import streamlit as st
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt

def plot_word2vec_2d(sentences, vector_size=100, window=5, min_count=1):
    """
    Train a Word2Vec model and visualize its 2D PCA projection using Plotly.
    Args:
        sentences (list[str]): List of raw text sentences.
        vector_size (int): Dimensionality of word vectors.
        window (int): Window size for context.
        min_count (int): Minimum word frequency for inclusion.
    Returns:
        fig (plotly.graph_objs.Figure): The 2D plot.
        model (gensim.models.Word2Vec): Trained Word2Vec model.
    """
    st.header("🔵 2D Word Embedding Visualization")

    # Preprocess the sentences
    tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]
    if not tokenized_sentences or all(len(s) == 0 for s in tokenized_sentences):
        st.error("❌ No valid words found in your input.")
        return None, None
    
    # Train a Word2Vec model
    model = Word2Vec(tokenized_sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    # Get the word vectors
    word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key])
    if word_vectors.shape[0] < 3 or word_vectors.shape[1] < 3:
        st.error("❌ Not enough data to perform PCA.")
        return None, None
    
    # Reduce the dimensions to 2D using PCA
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(word_vectors)

    # Generate distinct colors
    cmap = plt.get_cmap('tab20', len(tokenized_sentences))
    hex_colors = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b, a in [cmap(i) for i in range(len(tokenized_sentences))]]

    word_colors = []
    for word in model.wv.index_to_key:
        for i, sentence in enumerate(tokenized_sentences):
            if word in sentence:
                word_colors.append(hex_colors[i])
                break
    
    # Plot points
    word_ids = [f"word-{i}" for i in range(len(model.wv.index_to_key))]

    scatter = go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers+text',
        text=model.wv.index_to_key,
        textposition='top center',
        marker=dict(color=word_colors, size=8),
        customdata=word_colors,
        ids=word_ids,
        hovertemplate="Word: %{text}<br>Color: %{customdata}"
    )

    # Create line traces for each displayed sentence
    display_array = [True] * len(tokenized_sentences)

    # Create line traces for each sentence
    line_traces = []
    for i, sentence in enumerate(tokenized_sentences):
        if display_array[i]:
            line_vectors = [reduced_vectors[model.wv.key_to_index[word]] for word in sentence]
            line_trace = go.Scatter(
                x=[vector[0] for vector in line_vectors],
                y=[vector[1] for vector in line_vectors],
                mode='lines',
                line=dict(color=hex_colors[i], width=1, dash='solid'),
                showlegend=True,
                name=f"Sentence {i+1}",  # Customize the legend text
                hoverinfo='all'  # Disable line trace hover info
            )
            # Set different marker symbols for the start and end words
            line_traces.append(line_trace)

    fig = go.Figure(data=[scatter] + line_traces)

    # Set the plot title and axis labels
    fig.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        title="2D Visualization of Word Embeddings",
        width=1000,  # Custom width
        height=1000  # Custom height
    )

    # Optional: Show input
    with st.expander("📄 Show Input Sentences", expanded=False):
        for i, s in enumerate(sentences, 1):
            st.markdown(f"**Sentence {i}:** {s}")

    st.plotly_chart(fig, use_container_width=True)
    return fig, model
