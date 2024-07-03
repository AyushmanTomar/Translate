import pickle
import re
import string
import tensorflow
import numpy as np
from tensorflow import keras 
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional,Concatenate
from keras import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences


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



with open('eng_tokenizer.pkl', 'rb') as file:
    tokenizer_eng = pickle.load(file)

eng_index=tokenizer_eng.word_index

with open('hindi_tokenizer.pkl', 'rb') as file:
    tokenizer_hindi = pickle.load(file)

hindi_index=tokenizer_hindi.word_index

def process_input(text):
    text = clean(text)
    print('\n\nCleaned Text:',text)
    text = tokenizer_eng.texts_to_sequences([text])
    max_len_english = 349
    # print(text)
    padded_input_sequences = pad_sequences(text, maxlen = max_len_english, padding='post')
    # print(padded_input_sequences)
    return padded_input_sequences


# ## Defining model input and output data parameters
word_count_eng=55966
word_count_hindi= 61211
max_len_english=349
max_len_hindi= 314
embedding_dim = 100  
hidden_units = 256  
batch_size = 64



##model
encoder_input = Input(shape=(None, ))
encoder_embd = Embedding(word_count_eng,100, mask_zero=True)(encoder_input)
encoder_lstm = Bidirectional(LSTM(256, return_state=True))
encoder_output, forw_state_h, forw_state_c, back_state_h, back_state_c = encoder_lstm(encoder_embd)
state_h_final = Concatenate()([forw_state_h, back_state_h])
state_c_final = Concatenate()([forw_state_c, back_state_c])

  # takeing only states and create context vector
encoder_states= [state_h_final, state_c_final]

  # Decoder
decoder_input = Input(shape=(None,))
  # For zero padding we have added +1 in hindi vocab size
decoder_embd = Embedding(word_count_hindi, 100, mask_zero=True)
decoder_embedding= decoder_embd(decoder_input)
  # We used bidirectional layer above so we have to double units of this lstm
decoder_lstm = LSTM(512, return_state=True,return_sequences=True )
  # Only output of this decoder dont need self states
decoder_outputs, _, _= decoder_lstm(decoder_embedding, initial_state=encoder_states)

  # convert predicted numbers into probability using softmax
decoder_dense= Dense(word_count_hindi, activation='softmax')
  # again feed predicted output into decoder to predict its next word
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_input, decoder_input], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.load_weights('model_weights.weights.h5')








## Inference model
encoder_model = Model(encoder_input, encoder_states)
decoder_state_input_h = Input(shape=(512,))
decoder_state_input_c= Input(shape=(512,))
decoder_states_input= [decoder_state_input_h, decoder_state_input_c]

dec_embd2 = decoder_embd(decoder_input)

decoder_output2,state_h2, state_c2 = decoder_lstm(dec_embd2, initial_state=decoder_states_input)
deccoder_states2= [state_h2, state_c2]

decoder_output2 = decoder_dense(decoder_output2)

decoder_model = Model(
                      [decoder_input]+decoder_states_input,
                      [decoder_output2]+ deccoder_states2)






def get_predicted_sentence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = tokenizer_hindi.word_index['sos']

    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # print(output_tokens[0,-1,:])
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        # print("output",sampled_token_index)
        if sampled_token_index==0:
          break
        else:
         # convert max index number to hindi word
         for key, value in tokenizer_hindi.word_index.items():
            if value == sampled_token_index:
                # print(f"The key for value 31723 is '{key}'")
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
text = 'The knife is not sharp.'
test = process_input(text)
# print(get_predicted_sentence()[:-4])
prediction = get_predicted_sentence(test.reshape(1,349))[:-4]
if(len(prediction)==0):
    print('Cannot do it as of now!!')
else:
    print(prediction)



