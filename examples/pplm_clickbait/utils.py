from sklearn.model_selection import train_test_split
import pandas as pd
def dataset_split(file_path, save_path, seed=123):
    clickbait = open(file_path + "/all_head.1", 'r').readlines()
    non_clickbait = open(file_path + "/all_head.0").readlines()
    labels = [1] * len(clickbait) + [0] * len(non_clickbait)
    features = clickbait + non_clickbait
    # ATTENTION: The dataset for clickbait is imbalance
    data = [(i.strip(), j) for i, j in zip(features, labels)]
    assert len(features) == len(labels), "Length of input text is unequal to the length of labels"
    df = pd.DataFrame(data, columns=["feature","label"])
    # similar to the PPLM
    train_data, test_data = train_test_split(df, random_state=seed, train_size=0.9)
    # save the file content
    train_data.to_csv("{}/train.csv".format(save_path), header=None, index=None)
    test_data.to_csv("{}/test.csv".format(save_path), header=None, index=None)

if __name__ == '__main__':
    file_path = "./data/social"
    save_path = "./data/social"
    dataset_split(file_path, save_path)