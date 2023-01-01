# Author: Markus Rothkamm

# Importing Pandas as our Dataframe handler
import pandas as pd

def extract_column_names(csv_column_name="NUSW-NB15_features.csv")->list:
    '''
    Extract Column Names from NUSW-NB15_features.csv
    Returns the extraced Column Names as list
    '''

    columns = []
    content = ""
    with open(csv_column_name, "r") as file:
        content = file.readlines()[1:]

    for item in content:
        # splitting item by comma and select the second element e.g.: "1,srcip,nominal,Source IP address" -> srcip
        col = item.split(",")[1]
        # replacing whitespaces
        col = col.replace(" ", "")
        # make the names in lower cases
        col = col.lower()
        columns.append(col)

    return columns

def prepare_df(df:pd.DataFrame) -> pd.DataFrame:
    '''
    param1 df: Pandas Dataframe
    Prepares the given Dataframe:
        - Removing unwanted Columns (source port, destination ip, destianation port, )
    '''
    #drop unwanted columns
    df = df.drop(columns=["srcip", "sport", "dstip", "dsport", "stime", "ltime", "label"])

    # fill na
    df["attack_cat"] = df["attack_cat"].fillna("Normal")
    df["attack_cat"].mask(df["attack_cat"].str.startswith(" "), df["attack_cat"].str.strip(), inplace=True)
    df["ct_flw_http_mthd"] = df["ct_flw_http_mthd"].fillna(0)
    df["is_ftp_login"] = df["is_ftp_login"].fillna(0)
    df["ct_ftp_cmd"].mask(df["ct_ftp_cmd"] == " ", 0, inplace=True)
    
    #change dtypes
    df["proto"] = df["proto"].astype("string")
    df["state"] = df["state"].astype("string")
    df["service"] = df["service"].astype("string")
    df["ct_ftp_cmd"] = df["ct_ftp_cmd"].astype("float64")
    df["is_sm_ips_ports"] = df["is_sm_ips_ports"].astype("float64")
    df["is_ftp_login"] = df["is_ftp_login"].astype("float64")
    
    # change Backdoors to Backdoor, some values are a slight different in column attack_cat
    df.loc[df["attack_cat"] == "Backdoors", "attack_cat"] = "Backdoor"

    # console information: gather information about NaN Values
    for col in df:
        print("[DEBUG]", "col:", col, "Sum NaN", df[col].isna().sum(), "dtype:", df[col].dtype)

    # returning the prepared dataframe
    return df

def main(csv_paths:list=["..\\UNSW-NB15 - CSV Files\\UNSW-NB15_1.csv", "..\\UNSW-NB15 - CSV Files\\UNSW-NB15_2.csv", "..\\UNSW-NB15 - CSV Files\\UNSW-NB15_3.csv",
    "..\\UNSW-NB15 - CSV Files\\UNSW-NB15_4.csv"], csv_column_name="..\\UNSW-NB15 - CSV Files\\NUSW-NB15_features.csv") -> pd.DataFrame:
    # concat all 4 csv files to one dataframe
    temp_dfs = []
    columns = extract_column_names(csv_column_name)
    for csv in csv_paths:
        temp_dfs.append(pd.read_csv(csv, names=columns, low_memory=False))    
    df = pd.concat(temp_dfs, ignore_index=True)

    # looking at the first 3 rows of the dataframe if everything is okay at the first look
    print(df.head(3))
    # preparing the dataframe to remove unwanted values, etc.
    df = prepare_df(df)
    # spltting the df to train and test sets
    train, test = split_to_train_test_set(df)
    # each df to feather format (ressource friendly)
    df.to_feather("UNSW-NB15.ftr")
    train.to_feather("UNSW-NB15_train.ftr")
    test.to_feather("UNSW-NB15_test.ftr")

    # if we want to look into the test set, csv files are okay to look into it
    df.to_csv("UNSW-NB15.csv")
    train.to_csv("UNSW-NB15_train.csv")
    test.to_csv("UNSW-NB15_test.csv")

    return df

def split_to_train_test_set(df:pd.DataFrame) -> pd.DataFrame:
    '''
    splitting the dataframe into training and test sets: 90:10 Ratio
    '''
    # cutting df by row
    train = df.iloc[:int(0.9 * len(df)), :]
    test = df.iloc[int(0.9 * len(df)):, :]
    # resetting the index
    train = train.reset_index()
    test = test.reset_index()
    # console information
    print(f"[LOG] Training examples: {len(train)}")
    print(f"[LOG] Test examples: {len(test)}")
    return train, test


if __name__ == "__main__":
    # Change csv_paths parameter if you have an other path than me 
    main() 

    # console Information at the end
    df = pd.read_feather("UNSW-NB15.ftr")
    print(df.head(5))
    print(df.columns)
    