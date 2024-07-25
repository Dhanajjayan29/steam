import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# Define the categories
categories = [
    'airport', 'car rental', 'car dealer', 'taxi stand', 'train station', 
    'transit station', 'subway station', 'light rail station', 'rv park', 
    'gas station', 'parking', 'car repair', 'travel agency'
]

# Initialize LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(categories)

# Load the tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name)

def tokenize(sentences):
    if isinstance(sentences, list) and all(isinstance(sentence, str) for sentence in sentences):
        return tokenizer(sentences, padding=True, truncation=True, return_tensors='tf', max_length=512)
    else:
        raise ValueError("Input must be a list of strings")

def predict(texts):
    encodings = tokenize(texts)
    predictions = model(encodings)[0]
    predicted_labels = tf.argmax(predictions, axis=1)
    return label_encoder.inverse_transform(predicted_labels)

def main():
    st.title('Text Classification with BERT')
    
    st.write("Enter text below to classify:")
    
    user_input = st.text_area("Text Input", "Type your text here...")
    
    if st.button("Predict"):
        if user_input:
            texts = [user_input]
            predictions = predict(texts)
            st.write("Predicted Category:", predictions[0])
        else:
            st.write("Please enter some text.")

if __name__ == "__main__":
    main()
