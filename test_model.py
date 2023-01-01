# Author Markus Rothkamm
import tensorflow as tf
import pandas as pd
import numpy as np
# Static parameter
CLASS_TRANSLATOR = {"Normal": 0 , "Analysis": 1, "Backdoor": 2, "DoS": 3, "Exploits": 4, "Fuzzers": 5, "Generic": 6, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9}

def df_to_dataset(df, batch_size=32) -> tf.data.Dataset:
    '''
    Converts a dataframe to a dataset
    Returns the dataset
    '''
    df = df.copy()
    labels = df.pop("attack_cat")
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.batch(batch_size)
    return ds

def multi_predictions(model_path:str, df_size:float=0.01, test_df:str="UNSW-NB15_test.ftr"):
    '''
    Predicts the given test dataset.
    model_path =  given trained model.
    df_size: 1 = Use 100% of the df, 0.5 use 50% of the df
    Writes the information to a file (array.dat and test_model.txt) inside the model_path
    '''
    df = pd.read_feather(test_df)
    df = df.set_index("index")
    df = df.reset_index()
    df = df.drop(columns=["index",])
    
    # Prints the information about the Dataframe
    print(df.head(5))
    print(f"[DEBUG] Sum NaN: {df.isna().sum().sum()}")
    print(f"[DEBUG] Sum of Rows: {df.shape[0]} , type={type(df.shape[0])}")
    t = df.shape[0]*df_size
    print(f"[DEBUG] Sum of actual df size used {int(t)}")
    
    # Cut the unwanted rows.
    df = df.drop(range(int(t), df.shape[0]))

    # convert df to ds
    ds = df_to_dataset(df, batch_size=len(df))

    # load model
    model = tf.keras.models.load_model(model_path)
    
    # make predictions
    predictions = model.predict(ds, verbose=0, use_multiprocessing=True, workers=4)
    """ 
    Store the predictions to a 10x10 matrix
    use an empty (full of zeros) matrix and increment the the values by each predictions by 1.
    E.g.: If it was a Normal was predicted as Normal increment (0,0) by 1.
    The diagonal left upper corner to right bottom corner is the True Positive and True Negative
    """
    array = np.zeros((10,10), dtype=int)
    true_assigned = 0
    for count, prediction in enumerate(predictions):
        pred_attack_cat = np.argmax(prediction)
        actual_attack_cat = df["attack_cat"][count]
        array[CLASS_TRANSLATOR[actual_attack_cat], pred_attack_cat] += 1
        if pred_attack_cat == CLASS_TRANSLATOR[actual_attack_cat]:
            true_assigned += 1

    summary = (f"Done, Result:\nTrue Assigned={true_assigned}\nSum of Rows={df.shape[0]}"
            f"\nFalse + Assigned={df.shape[0] - true_assigned}\nIn Percentage:\n"
            f"{(true_assigned/df.shape[0]) *100}%\nFalse Assigned:{(1-(true_assigned/df.shape[0])) *100}%")

    # Save and print the matrix and the summary
    print(f"[DEBUG] {summary}")
    print(f"[DEBUG] {array}")
    with open(model_path + "test_model.txt", "w") as file:
        file.write(summary)
    array.tofile(model_path + "array.dat")


if __name__ == "__main__":
    model_path = "\\path\\to\\model\\folder"
    multi_predictions(model_path, df_size=1.0)