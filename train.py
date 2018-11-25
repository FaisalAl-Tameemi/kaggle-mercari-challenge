import numpy as np
import math

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K

from sklearn.model_selection import train_test_split


def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def get_model(X_train, max_text_len, max_brand_count, max_category_count, max_condition):
    #params
    dr_r = 0.1
    
    #Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    
    #Embeddings layers
    emb_name = Embedding(max_text_len, 50)(name)
    emb_item_desc = Embedding(max_text_len, 50)(item_desc)
    emb_brand_name = Embedding(max_brand_count, 10)(brand_name)
    emb_category_name = Embedding(max_category_count, 10)(category_name)
    emb_item_condition = Embedding(max_condition, 5)(item_condition)
    
    #rnn layer
    rnn_layer1 = GRU(16) (emb_item_desc)
    rnn_layer2 = GRU(8) (emb_name)
    
    #main layer
    main_l = concatenate([
        Flatten() (emb_brand_name)
        , Flatten() (emb_category_name)
        , Flatten() (emb_item_condition)
        , rnn_layer1
        , rnn_layer2
        , num_vars
    ])
    main_l = Dropout(dr_r) (Dense(128) (main_l))
    main_l = Dropout(dr_r) (Dense(64) (main_l))
    
    #output
    output = Dense(1, activation="linear") (main_l)
    
    #model
    model = Model([name, item_desc, brand_name
                   , category_name, item_condition, num_vars], output)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", rmsle_cust])
    
    return model


def prepare_data(dataset, max_seq_name, max_seq_item_desc):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=max_seq_name)
        ,'item_desc': pad_sequences(dataset.seq_item_description, maxlen=max_seq_item_desc)
        ,'brand_name': np.array(dataset.brand_name)
        ,'category_name': np.array(dataset.category_name)
        ,'item_condition': np.array(dataset.item_condition_id)
        ,'num_vars': np.array(dataset[['shipping']])
    }

    return X


def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5


def fit_predict(train, test, scaler, epochs = 5):
    MAX_NAME_SEQ = 10
    MAX_ITEM_DESC_SEQ = 75
    MAX_TEXT = np.max([np.max(train.seq_name.max())
                    , np.max(test.seq_name.max())
                    , np.max(train.seq_item_description.max())
                    , np.max(test.seq_item_description.max())])+2
    MAX_CATEGORY = np.max([train.category_name.max(), test.category_name.max()])+1
    MAX_BRAND = np.max([train.brand_name.max(), test.brand_name.max()])+1
    MAX_CONDITION = 5

    train_set, validation_set = train_test_split(train, random_state=123, train_size=0.99)

    X_train = prepare_data(train_set, MAX_NAME_SEQ, MAX_ITEM_DESC_SEQ)
    X_valid = prepare_data(validation_set, MAX_NAME_SEQ, MAX_ITEM_DESC_SEQ)
    X_test = prepare_data(test, MAX_NAME_SEQ, MAX_ITEM_DESC_SEQ)

    BATCH_SIZE = 20000
    

    model = get_model(X_train, MAX_TEXT, MAX_BRAND, MAX_CATEGORY, MAX_CONDITION)
    model.fit(X_train, train_set.target, epochs=epochs, batch_size=BATCH_SIZE
            , validation_data=(X_valid, validation_set.target)
            , verbose=1)

    val_preds = model.predict(X_valid)
    # val_preds = scaler.inverse_transform(val_preds)
    val_preds = np.exp(val_preds) + 1

    test_preds = model.predict(X_test)
    test_preds = np.exp(test_preds) + 1

    # mean_absolute_error, mean_squared_log_error
    y_true = np.array(validation_set.price.values)
    y_pred = val_preds[:,0]

    y_true_test = np.array(test.price.values)
    y_pred_test = y_true_test[:,0]
    
    val_score = rmsle(y_true, y_pred)
    test_score = rmsle(y_true_test, y_pred_test)

    return val_score, test_score