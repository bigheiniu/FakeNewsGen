from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
np.random.seed(123)
def dataset_split(file_path, save_path, seed=123, balance=True):
    # pos = open(file_path + "/clickbait.txt", 'r').readlines()
    # neg = open(file_path + "/genuine.txt", "r").readlines()
    data = pd.read_csv(file_path)
    pos = data[data["fake_label"] == 1].values.tolist()
    neg = data[data["fake_label"] == 0].values.tolist()
    columns = data.columns
    if balance:
        min_length = len(pos) if len(pos) < len(neg) else min(len(neg))
        np.random.shuffle(pos)
        np.random.shuffle(neg)
        pos = pos[:min_length]
        neg = neg[:min_length]

    # labels = [1] * len(pos) + [0] * len(neg)
    # features = pos + neg
    # data = [(i.strip(), j) for i, j in zip(features, labels)]
    # assert len(features) == len(labels), "Length of input text is unequal to the length of labels"
    # df = pd.DataFrame(data, columns=["feature","label"])
    df = pd.DataFrame(pos+neg, columns=columns)

    train_data, test_data = train_test_split(df, random_state=seed, train_size=0.9)
    # save the file content
    train_data.to_csv("{}/train.csv".format(save_path), header=None)
    test_data.to_csv("{}/test.csv".format(save_path), header=None)


def delete(save_path):
    train_data = pd.read_csv(save_path+"/train.csv", header=None)
    test_data = pd.read_csv(save_path+"/test.csv", header=None)
    train_data.to_csv(save_path + "/index_train.csv", header=None)
    test_data.to_csv(save_path + "/index_test.csv", header=None)

def prepare_openie(data_path, save_path):
    result = pd.read_csv(data_path, sep="," if "csv" in data_path else "\t")
    result_drop = result.dropna(how='any')
    if len(result) > len(result_drop):
        print("[WARNING] Remove empty samples")
        result_drop.to_csv(data_path, header=None, index=None, sep="," if "csv" in data_path else "\t")
    result = result_drop
    # result = result.values.tolist()
    th = "prompt\tref_fact\tref\tgen\tstyle_label"
    # prompt = [i[0] for i in result]
    # labels = [i[-1] for i in result]
    # contents = [i[-2] for i in result]
    # ref_facts = [i[1] for i in result]
    
    
    prompt = result['prompt'].values.tolist()
    labels = result['style_label'].values.tolist()
    contents = result['gen'].values.tolist()
    ref_facts = result['ref_fact'].values.tolist()
    
    # index = [i[0] for i in result]

    # labels = [i[-1] for i in result]
    # contents = [i[1] for i in result]
    # index_list = [i[0] for i in result]
    # prompt = [i[2] for i in result]

    os.system("rm {}/*".format(save_path))
    fout = open(save_path + "/openie_list.txt", 'w')
    for idx, c in enumerate(contents):
        file_name = "{}.txt".format(idx)
        fout.write(save_path + "/" + file_name + "\n")
        with open(save_path + "/" + file_name, "w") as f1:
            f1.write(c.strip().replace("\n", " ") + "\n")
    fout.close()

    def fact_tuple_extract(sentence_file, save_path, core_nlp_path="/home/yichuan/Materials/corenlp"):
        # a line for one document
        cmd = 'java -mx28g -cp "{}/*" edu.stanford.nlp.naturalli.OpenIE -filelist {} -output {} -format reverb'.format(
            core_nlp_path, sentence_file, save_path)
        print(cmd)
        os.system(cmd)
    def clean_fact_info(input_path):
        data = pd.read_csv(input_path, header=None, sep="\t")
        data["index"] = data.iloc[:, 0].apply(lambda x: int(x.split("/")[-1].split(".")[0]))
        data["fact"] = data[[2, 3, 4]].astype(str).apply(",".join, axis=1)
        # only take the first element
        data_highest = data.sort_values([11], ascending=False).groupby(["index", 1]).head(1).reset_index(drop=True)
        data_group = data_highest.groupby(["index"]).agg({"fact":"|".join}).reset_index()
        return data_group[["index", "fact"]].values.tolist()

    fact_tuple_extract(save_path + "/openie_list.txt", save_path + "/openie.result")
    fact = clean_fact_info(save_path+"/openie.result")
    fact_news = []
    for index_fact in fact:
        index = index_fact[0]
        fact = index_fact[1]
        p = prompt[index]
        c = contents[index]
        l = labels[index]
        ref_fact = ref_facts[index]
        fact_news.append((p, c, fact, l, ref_fact))
        # fact_news.append((p, c, fact, l, index))
    fact_news_df = pd.DataFrame(fact_news, columns=["prompt", "content", "fact_gen", "fake_label","fact_ref"])
    fact_news_df.to_csv(save_path+"/fact_gen.tsv", index=None, sep="\t")

if __name__ == '__main__':

    # this part is for the train/test dataset split
    # file_path = "./data/news_corpus/polit_fact/fact_news.csv"
    # save_path = "./data/news_corpus/polit_fact"
    # dataset_split(file_path, save_path)

    # prepare the fact information extraction
    data_path = "/home/yichuan/style_transfer/examples/pplm_clickbait/output_CLM_fact/test_generate.tsv"
    save_path = "/home/yichuan/style_transfer/examples/pplm_clickbait/output_CLM_fact/openie_fact"
    prepare_openie(data_path, save_path)
