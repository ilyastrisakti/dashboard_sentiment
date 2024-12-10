import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud

# Fungsi Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

# Fungsi Menghitung Skor Sentimen
def calculate_sentiment_score(text, kamus_sentimen):
    words = text.split()
    score = sum([kamus_sentimen.get(word, 0) for word in words])
    return score

# Fungsi Melabeli Sentimen
def label_sentiment(score):
    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'

# Konfigurasi halaman
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("📊 Sentiment Analysis Dashboard")

# Sidebar untuk upload file
st.sidebar.header("📝 Upload Files")
comments_file = st.sidebar.file_uploader("Upload file CSV komentar YouTube:", type=["csv"])
kamus_file = st.sidebar.file_uploader("Upload file Excel kamus sentimen:", type=["xlsx"])

if comments_file and kamus_file:
    # Membaca file
    comments_df = pd.read_csv(comments_file)
    kamus_df = pd.read_excel(kamus_file)

    # Validasi kamus
    if 'term' not in kamus_df.columns or 'sentiment' not in kamus_df.columns:
        st.error("Kamus sentimen harus memiliki kolom 'term' dan 'sentiment'.")
    else:
        kamus_sentimen = dict(zip(kamus_df['term'], kamus_df['sentiment']))

        # Preprocessing
        comments_df['cleaned_comment'] = comments_df['Comment'].apply(preprocess_text)
        comments_df['sentiment_score'] = comments_df['cleaned_comment'].apply(
            lambda x: calculate_sentiment_score(x, kamus_sentimen)
        )
        comments_df['sentiment_label'] = comments_df['sentiment_score'].apply(label_sentiment)
        comments_df['comment_length'] = comments_df['cleaned_comment'].str.split().apply(len)

        # Tab Layout
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🛠 Model Training", "🔍 Prediksi Sentimen"])

        # Tab 1: Dashboard
        with tab1:
            st.header("📊 Dashboard")
            col1, col2 = st.columns([3, 2])

            with col1:
                st.subheader("Distribusi Sentimen")
                sentiment_counts = comments_df['sentiment_label'].value_counts()
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'], ax=ax)
                ax.set_title("Distribusi Sentimen")
                ax.set_xlabel("Kategori Sentimen")
                ax.set_ylabel("Jumlah")
                st.pyplot(fig)

                st.subheader("Distribusi Waktu Komentar")
                if 'Date' in comments_df.columns:
                    comments_df['Date'] = pd.to_datetime(comments_df['Date'])
                    time_distribution = comments_df['Date'].dt.date.value_counts().sort_index()
                    fig, ax = plt.subplots()
                    time_distribution.plot(ax=ax)
                    ax.set_title("Jumlah Komentar Per Hari")
                    ax.set_xlabel("Tanggal")
                    ax.set_ylabel("Jumlah Komentar")
                    st.pyplot(fig)
                else:
                    st.warning("Kolom 'Date' tidak tersedia dalam dataset.")

            with col2:
                st.subheader("Statistik Umum")
                st.metric("Total Komentar", len(comments_df))
                st.metric("Komentar Positif", sentiment_counts.get('positive', 0))
                st.metric("Komentar Negatif", sentiment_counts.get('negative', 0))
                st.metric("Komentar Netral", sentiment_counts.get('neutral', 0))

                st.subheader("☁️ Word Cloud")
                wordcloud_text = ' '.join(comments_df['cleaned_comment'])
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

            st.subheader("Panjang Komentar vs Sentimen")
            fig, ax = plt.subplots()
            for label in comments_df['sentiment_label'].unique():
                subset = comments_df[comments_df['sentiment_label'] == label]
                ax.hist(subset['comment_length'], alpha=0.5, label=label, bins=20)
            ax.set_title("Distribusi Panjang Komentar berdasarkan Sentimen")
            ax.set_xlabel("Panjang Komentar (jumlah kata)")
            ax.set_ylabel("Jumlah")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Filter Data Berdasarkan Sentimen")
            selected_label = st.selectbox("Pilih Sentimen:", ['positive', 'negative', 'neutral'])
            filtered_df = comments_df[comments_df['sentiment_label'] == selected_label]
            st.dataframe(filtered_df)

        # Tab 2: Model Training
        with tab2:
            st.header("🛠 Model Training")
            X = comments_df['cleaned_comment']
            y = comments_df['sentiment_label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            vectorizer = CountVectorizer()
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            models = {
                'Naive Bayes': MultinomialNB(),
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Support Vector Machine': SVC()
            }

            model_performance = {}
            for model_name, model in models.items():
                model.fit(X_train_vec, y_train)
                y_pred = model.predict(X_test_vec)
                accuracy = accuracy_score(y_test, y_pred)
                model_performance[model_name] = accuracy

                st.subheader(f"Model: {model_name}")
                st.write(f"Accuracy: {accuracy:.2f}")
                st.text(f"Classification Report:\n{classification_report(y_test, y_pred)}")

            st.subheader("📈 Perbandingan Akurasi Model")
            fig, ax = plt.subplots()
            ax.bar(model_performance.keys(), model_performance.values(), color=['orange', 'purple', 'cyan'])
            ax.set_title("Akurasi Model")
            ax.set_ylabel("Akurasi")
            st.pyplot(fig)

        # Tab 3: Prediksi Sentimen
        with tab3:
            st.header("🔍 Prediksi Sentimen Baru")
            new_comments = st.text_area("Masukkan komentar baru (pisahkan dengan garis baru):").split('\n')

            if st.button("Prediksi Sentimen"):
                new_comments_cleaned = [preprocess_text(comment) for comment in new_comments if comment]
                new_comments_vec = vectorizer.transform(new_comments_cleaned)

                best_model = models['Naive Bayes']
                new_predictions = best_model.predict(new_comments_vec)

                st.subheader("Hasil Prediksi:")
                for comment, sentiment in zip(new_comments, new_predictions):
                    st.write(f"- **{comment}** → Sentimen: **{sentiment}**")
