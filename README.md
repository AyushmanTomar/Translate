# English-Hindi Translation

This project involves a **Bidirectional LSTM** for translating English to Hindi. The model is trained on a comprehensive parallel corpus of English and Hindi sentences, covering a wide range of topics from social sciences, humanities, natural sciences, to applied sciences and technology.

## Project Overview

- **Data Preprocessing**: The project includes a series of data preprocessing steps performed on an English-Hindi parallel corpus. The process includes cleaning the text by removing punctuation, non-Hindi characters, and extra spaces. It also tokenizes the English text using the Keras Tokenizer. The Hindi text is preprocessed by adding start-of-sentence (sos) and end-of-sentence (eos) tokens.

- **Model Architecture**: The model's architecture includes embedding layers, bidirectional LSTM for the encoder, LSTM for the decoder, and a dense layer with softmax activation for output.

- **Training**: The model is trained for 25 epochs with a batch size of 64. The training involves using callbacks like EarlyStopping and fitting the model on training data with validation on test data.

## Getting Started

To get started with this project, clone the repository and install the necessary dependencies. Then, run the `birirectional_lstm.ipynb` notebook to train the model and make predictions.

## Contributing

Contributions are welcome!

