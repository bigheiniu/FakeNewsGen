from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
np.random.seed(123)
def dataset_split(file_path, save_path, seed=123, balance=True):
    clickbait = open(file_path + "/clickbait.txt", 'r').readlines()
    non_clickbait = open(file_path + "/genuine.txt", "r").readlines()
    if balance:
        min_length = len(clickbait) if len(clickbait) < len(non_clickbait) else min(len(non_clickbait))
        np.random.shuffle(clickbait)
        np.random.shuffle(non_clickbait)
        clickbait = clickbait[:min_length]
        non_clickbait = non_clickbait[:min_length]

    labels = [1] * len(clickbait) + [0] * len(non_clickbait)
    features = clickbait + non_clickbait
    data = [(i.strip(), j) for i, j in zip(features, labels)]
    assert len(features) == len(labels), "Length of input text is unequal to the length of labels"
    df = pd.DataFrame(data, columns=["feature","label"])

    train_data, test_data = train_test_split(df, random_state=seed, train_size=0.9)
    # save the file content
    train_data.to_csv("{}/train.csv".format(save_path), header=None, index=None)
    test_data.to_csv("{}/test.csv".format(save_path), header=None, index=None)



def prepare_openie(data_path, save_path):
    result = pd.read_csv(data_path, header=None, sep="\t")
    result_drop = result.dropna( how='any')
    if len(result) > len(result_drop):
        print("[WARNING] Remove empty samples")
        result_drop.to_csv(data_path, header=None, sep="\t", index=None)
    result = result_drop
    result = result.values.tolist()
    prompt = [i[1] for i in result]
    labels = [i[2] for i in result]
    contents = [i[-1] for i in result]
    os.system("rm {}/*".format(save_path))
    fout = open(save_path + "/openie_list.txt", 'w')
    for idx, c in enumerate(contents):
        file_name = "{}.txt".format(idx)
        fout.write(save_path + "/" + file_name + "\n")
        with open(save_path + "/" + file_name, "w") as f1:
            f1.write(c.strip().replace("\n", " ") + "\n")
    fout.close()

    def fact_tuple_extract(sentence_file, save_path, core_nlp_path=""):
        # a line for one document
        cmd = 'java -mx28g -cp "{}/*" edu.stanford.nlp.naturalli.OpenIE -filelist {} -output {} -format reverb'.format(
            core_nlp_path, sentence_file, save_path)
        print(cmd)
        os.system(cmd)
    def clean_fact_info(input_path):
        data = pd.read_csv(input_path, header=None, sep="\t")
        data["index"] = data.iloc[:, 0].apply(lambda x: x.split("/")[-1])
        data["fact"] = data[[2, 3, 4]].astype(str).apply(",".join, axis=1)
        # only take the first element
        data_highest = data.sort_values([11], ascending=False).groupby(["index", 1]).head(1).reset_index(drop=True)
        data_group = data_highest.groupby(["index"]).agg({"fact":"|".join}).reset_index(drop=True)
        return data_group[["index", "fact"]].valuest.tolist()

    fact_tuple_extract(save_path + "/openie_list.txt", save_path + "/openie.result")
    fact = clean_fact_info(save_path+"/openie.result")
    fact_news = []
    for index_fact in fact:
        index = index_fact[0]
        fact = index_fact[1]
        p = prompt[index]
        c = contents[index]
        l = labels[index]
        fact_news.append((p, c, fact, l))
    fact_news_df = pd.DataFrame(fact_news, columns=["prompt","content", "fact", "fake_label"])
    fact_news_df.to_csv(save_path+"/fact_news.csv", index=None)

if __name__ == '__main__':

    # this part is for the train/test dataset split
    file_path = "./data/clickbait-detector"
    save_path = "./data/clickbait-detector"
    dataset_split(file_path, save_path)