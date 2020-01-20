from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import rouge
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import os
from clickbait_utils import prepare_openie

def get_bleu(candidate, ref):
    candidate = [i.split() for i in candidate]
    ref = [[i.split()] for i in ref]
    score = corpus_bleu(ref, candidate)
    print("BLEU score is {}".format(score))

def get_rouge(candidate, reference):
    candidate = [i for i in candidate]
    # print(np.mean([len(i.split()) for i in candidate]))
    reference = [[i] for i in reference]
    def prepare_results(m, p, r, f):
        return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(m, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

    for aggregator in ['Avg', 'Best', 'Individual']:
        print('Evaluation with {}'.format(aggregator))
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'

        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                                max_n=4,
                                limit_length=True,
                                length_limit=100,
                                length_limit_type='words',
                                apply_avg=apply_avg,
                                apply_best=apply_best,
                                alpha=0.5,  # Default F1_score
                                weight_factor=1.2,
                                stemming=True)

        hypothesis_1 = "King Norodom Sihanouk has declined requests to chair a summit of Cambodia 's top political leaders , saying the meeting would not bring any progress in deadlocked negotiations to form a government .\nGovernment and opposition parties have asked King Norodom Sihanouk to host a summit meeting after a series of post-election negotiations between the two opposition groups and Hun Sen 's party to form a new government failed .\nHun Sen 's ruling party narrowly won a majority in elections in July , but the opposition _ claiming widespread intimidation and fraud _ has denied Hun Sen the two-thirds vote in parliament required to approve the next government .\n"
        references_1 = [
            "Prospects were dim for resolution of the political crisis in Cambodia in October 1998.\nPrime Minister Hun Sen insisted that talks take place in Cambodia while opposition leaders Ranariddh and Sam Rainsy, fearing arrest at home, wanted them abroad.\nKing Sihanouk declined to chair talks in either place.\nA U.S. House resolution criticized Hun Sen's regime while the opposition tried to cut off his access to loans.\nBut in November the King announced a coalition government with Hun Sen heading the executive and Ranariddh leading the parliament.\nLeft out, Sam Rainsy sought the King's assurance of Hun Sen's promise of safety and freedom for all politicians.",
            "Cambodian prime minister Hun Sen rejects demands of 2 opposition parties for talks in Beijing after failing to win a 2/3 majority in recent elections.\nSihanouk refuses to host talks in Beijing.\nOpposition parties ask the Asian Development Bank to stop loans to Hun Sen's government.\nCCP defends Hun Sen to the US Senate.\nFUNCINPEC refuses to share the presidency.\nHun Sen and Ranariddh eventually form a coalition at summit convened by Sihanouk.\nHun Sen remains prime minister, Ranariddh is president of the national assembly, and a new senate will be formed.\nOpposition leader Rainsy left out.\nHe seeks strong assurance of safety should he return to Cambodia.\n",
            ]

        hypothesis_2 = "China 's government said Thursday that two prominent dissidents arrested this week are suspected of endangering national security _ the clearest sign yet Chinese leaders plan to quash a would-be opposition party .\nOne leader of a suppressed new political party will be tried on Dec. 17 on a charge of colluding with foreign enemies of China '' to incite the subversion of state power , '' according to court documents given to his wife on Monday .\nWith attorneys locked up , harassed or plain scared , two prominent dissidents will defend themselves against charges of subversion Thursday in China 's highest-profile dissident trials in two years .\n"
        references_2 = "Hurricane Mitch, category 5 hurricane, brought widespread death and destruction to Central American.\nEspecially hard hit was Honduras where an estimated 6,076 people lost their lives.\nThe hurricane, which lingered off the coast of Honduras for 3 days before moving off, flooded large areas, destroying crops and property.\nThe U.S. and European Union were joined by Pope John Paul II in a call for money and workers to help the stricken area.\nPresident Clinton sent Tipper Gore, wife of Vice President Gore to the area to deliver much needed supplies to the area, demonstrating U.S. commitment to the recovery of the region.\n"

        all_hypothesis = [hypothesis_1, hypothesis_2]
        all_references = [references_1, references_2]

        scores = evaluator.get_scores(candidate, reference)

        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            if not apply_avg and not apply_best:  # value is a type of list as we evaluate each summary vs each reference
                for hypothesis_id, results_per_ref in enumerate(results):
                    nb_references = len(results_per_ref['p'])
                    for reference_id in range(nb_references):
                        pass
                #         print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                #         print('\t' + prepare_results(metric, results_per_ref['p'][reference_id],
                #                                      results_per_ref['r'][reference_id],
                #                                      results_per_ref['f'][reference_id]))
                # print()
            else:
                print(prepare_results(metric, results['p'], results['r'], results['f']))
        print()



# def read_fact(ref_path, candidate_path, join_ref, join_candidate, fact_ref_index, fact_candidate_index):
def read_fact(file_path, fact_ref_index, fact_candidate_index):
    all = pd.read_csv(file_path, sep="\t")
    # all = pd.merge(gen, ref, left_on=join_candidate, right_on=join_ref, how="inner",suffixes=("_gen", "_ref"))


    def load_from_list(data, column_gen, column_ref):
        fact_list_gen = data[column_gen].values.tolist()
        fact_list_gen = [[j.split(",") for j in i.split("|")] for i in fact_list_gen]

        fact_list_ref = data[column_ref].values.tolist()
        fact_list_ref = [[j.split(",") for j in i.split("|")] for i in fact_list_ref]

        ratio_list = []
        for index, (gen_fact, ref_fact) in enumerate(zip(fact_list_gen, fact_list_ref)):
            cartesian = list(product(gen_fact, ref_fact))
            cartesian = [i for i in cartesian if
                         i[1][1].lower() == i[0][1].lower() and i[1][2].lower() == i[0][2].lower()]
            intersect = [i for i in cartesian if i[1][0].lower() == i[0][0].lower()]
            if len(intersect) != 0:
                print(intersect)
                print("The File is {}".format(index))
            ratio = len(intersect) * 1.0 / (len(fact_list_gen) + 1)
            ratio_list.append(ratio)
        return np.mean(ratio_list)
    print("Fact Acc is {}".format(load_from_list(all, column_gen=fact_candidate_index, column_ref=fact_ref_index)))
    # gen_fact_dic = load_from_list(gen)
    # ref_fact_dic = load_from_list(ref)
    #
    # # intersection
    # ratio_list = []
    # for key in tqdm(ref_fact_dic.keys()):
    #     if len(gen_fact_dic) == 0:
    #         continue
    #
    #     cartesian = list(product(ref_fact_dic[key], gen_fact_dic[key]))
    #     cartesian = [i for i in cartesian if i[1][1].lower() == i[0][1].lower() and i[1][2].lower() == i[0][2].lower()]
    #     intersect = [i for i in cartesian if i[1][0].lower() == i[0][0].lower()]
    #     if len(intersect) != 0:
    #         print(intersect)
    #         print("The File is {}".format(key))
    #     ratio = len(intersect) * 1.0 / (len(gen_fact_dic[key]) + 1)
    #     ratio_list.append(ratio)
    # return np.mean(ratio_list)



def evaluate_all(file_path):
    openie_path = "/".join(file_path.split("/")[:-1]) + "/openie_fact"
    if os.path.exists(openie_path) is False:
        os.mkdir(openie_path)
    prepare_openie(file_path, openie_path)
    read_fact(openie_path+"/fact_gen.tsv", "fact_ref", "fact_gen")

    data = pd.read_csv(file_path, sep="\t")
    candidate = data["gen"].values.tolist()
    ref = data["ref"].values.tolist()
    get_bleu(candidate, ref)
    get_rouge(candidate, ref)

import numpy as np
if __name__ == '__main__':
    # file_name = "/home/yichuan/style_transfer/examples/pplm_clickbait/output_gpt2_fact/test_generate.tsv"
    # file_name_2 = "/home/yichuan/style_transfer/examples/pplm_clickbait/output_CLM_fact/test_generate.tsv"
    # ref_file = "/home/yichuan/style_transfer/examples/pplm_clickbait/data/news_corpus/polit_fact/index_test.csv"
    #
    # candidate = pd.read_csv(file_name, header=None, sep="\t")
    # ref = pd.read_csv(ref_file, header=None)
    # ref = ref.iloc[1:, :]
    # candidate = candidate[1].values.tolist()
    # ref = ref[2].values.tolist()
    # # candidate = [i.split() for i in candidate]
    # # ref = [[i.split()] for i in ref]
    #
    # candidate = [i for i in candidate]
    # print(np.mean([len(i.split()) for i in candidate]))
    # ref = [[i] for i in ref]
    # print(np.mean([len(i[0].split()) for i in ref]))
    #
    #
    # get_rouge(candidate, ref)

    # file_name = "/home/yichuan/style_transfer/examples/pplm_clickbait/output_CLM_fact/openie_fact/fact_gen.tsv"
    # read_fact(file_name, fact_ref_index="fact_ref", fact_candidate_index="fact_gen")
    file_name = "/home/yichuan/style_transfer/examples/pplm_clickbait/output_CLM_style/test_generate.tsv"
    file_name = "/home/yichuan/style_transfer/examples/pplm_clickbait/grover_test_generated.tsv"
    evaluate_all(file_name)