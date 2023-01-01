# Author Markus Rothkamm
import pandas as pd
from datetime import datetime as dt
import tensorflow as tf
import json
# Define static parameters
CLASS_TRANSLATOR = {"Normal": 0 , "Analysis": 1, "Backdoor": 2, "DoS": 3, "Exploits": 4, "Fuzzers": 5, "Generic": 6, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9}
SHUFFLE_BUFFER = 500
BATCH_SIZE = 128
EPOCHS = 20
DENSE_LAYERS_UNITS = [256, 256]

def stack_dict(inputs:dict, fun=tf.stack):
    '''
    Combines a given dict to a tensor
    returns the combined tensor
    '''
    values = []
    for key in sorted(inputs.keys()):
      values.append(tf.cast(inputs[key], tf.float32))
    return fun(values, axis=-1)

def preprocess_features(df:pd.DataFrame) -> tuple[dict, list]:
    '''
    Process the numerical, categorical and binary features
    returns the inputs and preprocessed Features
    '''
    # Classify feature names as numeric, categorical xor binary:
    numeric_feature_names = [
                "dur", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "sload", "dload",
                "sloss", "dloss", "sintpkt", "dintpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
                "synack", "ackdat", "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "ct_srv_src",
                "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
                "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    ]
    categorical_feature_names = ["proto", "state", "service"]
    binary_feature_names = ["is_ftp_login", "is_sm_ips_ports"]

    # Match dtype to each feature
    inputs = {}
    for name, column in df.items():
        if type(column[0]) == str:
            dtype = tf.string
        elif(name in categorical_feature_names):
            dtype = tf.int64
        else:
            dtype = tf.float32
        inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)

    # Preprocess the features
    preprocessed = []

    # Select numeric features from data frame and normalize them
    numeric_features = df[numeric_feature_names]
    tf.convert_to_tensor(numeric_features)
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(stack_dict(dict(numeric_features)))

    numeric_inputs = {}
    for name in numeric_feature_names:
        numeric_inputs[name]=inputs[name]

    numeric_inputs = stack_dict(numeric_inputs)
    numeric_normalized = normalizer(numeric_inputs)
    preprocessed.append(numeric_normalized)

    # building one hot convert strings to numbers, for categorical features.
    vocab_proto = df["proto"].unique().tolist()
    lookup_proto = tf.keras.layers.StringLookup(vocabulary=vocab_proto, output_mode="one_hot")

    x_proto = inputs["proto"][:, tf.newaxis]
    x_proto = lookup_proto(x_proto)
    preprocessed.append(x_proto)

    vocab_service = df["service"].unique().tolist()
    lookup_service = tf.keras.layers.StringLookup(vocabulary=vocab_service, output_mode="one_hot")
    x_service = inputs["service"][:, tf.newaxis]
    x_service = lookup_service(x_service)
    preprocessed.append(x_service)

    vocab_state = df["state"].unique().tolist()
    lookup_state = tf.keras.layers.StringLookup(vocabulary=vocab_state, output_mode="one_hot")
    x_state = inputs["state"][:, tf.newaxis]
    x_state = lookup_state(x_state)
    preprocessed.append(x_state)

    # preprocess binary features
    for name in binary_feature_names:
        input_temp = inputs[name]
        input_temp = input_temp[:, tf.newaxis]
        float_value = tf.cast(input_temp, tf.float32)
        preprocessed.append(float_value)

    return inputs, preprocessed

def get_df(df_path:str="UNSW-NB15_train.ftr") -> pd.DataFrame:
    '''
    Reads the given df path. Sets the Attack Categorie to integers. Reseting Index drop Wrong imported Index Column.
    Returns readed & modified df
    '''
    df = pd.read_feather(df_path)
    # replace the name of the classes with numbers
    df["attack_cat"] = df["attack_cat"].replace(CLASS_TRANSLATOR)
    df["attack_cat"] = df["attack_cat"].astype("float64")
    # drop the index column. Sometimes I get two index columns. Trying to reset it
    df = df.set_index("index")
    df = df.reset_index()
    df = df.drop(columns=["index"])
    return df


if __name__ == "__main__":
    # Define static parameters first at line 6
    # load training dataframe
    df = get_df(df_path="UNSW-NB15_train.ftr")
    #select the target feature -> we want an output like: .... Sample is: 0.90 Worms, 0.60 Shellcode, etc.
    target = df.pop("attack_cat")
    inputs, preprocessed = preprocess_features(df)
    preprocessed_result = tf.concat(preprocessed, axis=-1)
    preprocessor = tf.keras.Model(inputs, preprocessed_result)

    try:
        # rankdir=LR -> Picture horizontal (from left to right, rankdir=TB from top to bottom).
        # This step is not mendatory but it can help fix some problems. E.g.: Forgetting a feature
        tf.keras.utils.plot_model(preprocessor, rankdir="LR", show_shapes=True, to_file="plot_model.png")
    except ImportError:
        # if no Graphviz installed
        print("[Warning] Please install Graphiz to plot the model (see instructions https://graphviz.gitlab.io/download/).")

    # Creating layers of the neural network and putting everything together
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(DENSE_LAYERS_UNITS[0], activation="relu", input_shape=(205,)),
        tf.keras.layers.Dense(DENSE_LAYERS_UNITS[1], activation="relu"),
        tf.keras.layers.Dense(10)
    ])
    x = preprocessor(inputs)
    result = model(x, training=True)
    model = tf.keras.Model(inputs, result)
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    # Defining the Path to save the model
    model_file_name = dt.now().strftime("%Y-%m-%d-%H-%M-%S") + f"_epochs={EPOCHS}_batchsize={BATCH_SIZE}"

    # Creating checkpoints
    checkpoint_file_path = "Model_Checkpoint\\"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file_path,
        save_weights_only=False,
        monitor="loss",
        mode="min",
        save_best_only=True,
        save_freq="epoch",
        verbose=1
    )

    # Train the Model
    history = model.fit(dict(df), target, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[model_checkpoint])
    # Save everything that we want to save, loss & accuracy values of each epoch, the model itself
    model.save(model_file_name + "_Model_Save\\")
    history_dict = history.history
    # Print and write the history as a dict. It contains loss and accuracy values for each epoch
    history_json_string = json.dumps(history_dict)
    with open(checkpoint_file_path + "history.txt", "w") as file:
        file.write(history_json_string)
    
    print("[DEBUG] Done training!")