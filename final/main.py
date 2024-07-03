import pickle
import re
import string
import nltk
import tensorflow
import numpy as np
from tensorflow import keras 
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional,Concatenate
from keras import Model
from nltk.corpus import stopwords
from nltk.corpus import indian
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwd = stopwords.words('english')
ps = PorterStemmer()

## Processing the input data
def clean(text):
    text = re.sub(' +', ' ', text)
    text = re.sub(',', ' ', text)
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))
    text = re.sub('”', '', text)
    text = re.sub('“', '', text)
    text = re.sub('’', '', text)
    text = re.sub('‘', '', text)
    text = re.sub('\\n', ' ', text)
    text = text.lower()
    return text


def tokenize_text(text):
    lst = nltk.word_tokenize(text)
    return lst



def stopwo(lst):
    for word in lst:
        if word in stopwd:
            lst.remove(word)
    return lst
    


def merge(english):
    eng_merged=[]
    sentence=""
    for word in english:
        sentence=sentence+word+" "
    eng_merged.append(sentence.strip())
    return eng_merged


with open('eng_tokenizer.pkl', 'rb') as file:
    tokenizer_eng = pickle.load(file)

eng_index=tokenizer_eng.word_index

with open('hindi_tokenizer.pkl', 'rb') as file:
    tokenizer_hindi = pickle.load(file)

hindi_index=tokenizer_hindi.word_index

def process_input(text):
    text = clean(text)
    text = tokenize_text(text)
    text = stopwo(text)
    text = merge(text)
    print(text)
    text = tokenizer_eng.texts_to_sequences(text)
    max_len_english = 260
    padded_input_sequences = pad_sequences(text, maxlen = max_len_english, padding='post')
    # print(padded_input_sequences)
    return padded_input_sequences




## defining model
def encoder_decoder_model(vocab_size_input, vocab_size_output, max_seq_length_input, max_seq_length_output, embedding_dim, hidden_units):
    # Define encoder input layer
    encoder_inputs = Input(shape=(max_seq_length_input,))
    
    # Define encoder embedding layer
    encoder_embedding = Embedding(input_dim=vocab_size_input, output_dim=embedding_dim)(encoder_inputs)
    
    # Define encoder LSTM layer
    encoder_lstm = Bidirectional(LSTM(hidden_units, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
    
    # Concatenate forward and backward states
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]
    
    # Define decoder input layer
    decoder_inputs = Input(shape=(max_seq_length_output-1,))
    
    # Define decoder embedding layer
    decoder_embedding = Embedding(input_dim=vocab_size_output, output_dim=embedding_dim)(decoder_inputs)
    
    # Define decoder LSTM layer with initial state set to encoder states
    decoder_lstm = LSTM(hidden_units * 2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    
    # Define decoder output layer
    decoder_dense = Dense(vocab_size_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()
    return model, encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense





## Defining model input and output data parameters
word_count_eng=72459
word_count_hindi= 76219
max_len_english=260
max_len_hindi= 418
embedding_dim = 100  
hidden_units = 256  
batch_size = 64

model, encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense= encoder_decoder_model(word_count_eng, word_count_hindi, max_len_english, max_len_hindi, embedding_dim, hidden_units)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

with open('weight_new_non_null2.pkl', 'rb') as file:
    data = pickle.load(file)

model.set_weights(data)
print(model.get_weights())

# Define the inference model for the encoder
encoder_model = Model(encoder_inputs, encoder_states)

# Define the initial state for the decoder
decoder_state_input_h = Input(shape=(hidden_units * 2,))
decoder_state_input_c = Input(shape=(hidden_units * 2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Define the decoder LSTM layer
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

# Define the decoder output layer
decoder_outputs = decoder_dense(decoder_outputs)

# Define the inference model for the decoder
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)




def get_predicted_sentence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = hindi_index['sos']

    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        print(max(output_tokens[0,0,:]))
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        print("output",sampled_token_index)
        if sampled_token_index==0:
          break
        else:   
         # convert max index number to hindi word
         for key, value in tokenizer_hindi.word_index.items():
            if value == sampled_token_index:
                print(f"The key for value 31723 is '{key}'")
                sampled_char=key
                break
        #  sampled_char = hindi_index[sampled_token_index]
        # aapend it to decoded sentence
        decoded_sentence += ' '+sampled_char
        
        # Exit condition: either hit max length or find stop token.
        if (sampled_char == 'eos' or len(decoded_sentence) >= 417):
            stop_condition = True
        
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]
    
    return decoded_sentence










## Sentence to predict
text = 'i lost everything i had'
test = process_input(text)
print(get_predicted_sentence(test)[:-4])
# print(test.shape)


