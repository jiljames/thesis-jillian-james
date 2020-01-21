import random
import nltk
import json
import argparse

def remove_filler(generated):
    for j in range(len(generated)):
        generated[j] = [value for value in generated[j] if value != " _FILL_ "]
    return generated


def int_file_to_text_ls(f, int_to_word):
    '''
    Reads file in dataloader form and converts to text
    using dictionary mappings
    '''
    text_ls = []
 
    for line in f:
        line = line.strip()
        line = line.split()
        parse_line = [int_to_word[x] for x in line]
        " ".join(parse_line)
        text_ls.append(parse_line)
    return text_ls


def main():
    # Print 5 random examples from generated data when given an application.
    parser = argparse.ArgumentParser(description='Program for showing generator sample given previously run application.')
    parser.add_argument('app', metavar='application', type=str, default = 'obama',
            help='Enter either \'obama\', \'haiku\', or \'synth\'. Must run module_sequance_gan.py with specified application before.')
    args = parser.parse_args()

    if args.app == "haiku":
        int_to_word = json.load(open("haiku/int_to_word.json", 'r'))
        generated = int_file_to_text_ls(open("haiku/eval_file.txt", 'r'), int_to_word)
        references = int_file_to_text_ls(open("haiku/haiku_to_int.test.txt", 'r'), int_to_word)
    elif args.app == "obama":
        int_to_word = json.load(open("obama/int_to_word.json", 'r'))
        generated = int_file_to_text_ls(open("obama/eval_file.txt", 'r'), int_to_word)
        references = int_file_to_text_ls(open("obama/obama_to_int.test.txt", 'r'), int_to_word)
    else:
        int_to_word = json.load(open("synth/int_to_word.json", 'r'))
        generated = int_file_to_text_ls(open("synth/eval_file.txt", 'r'), int_to_word)
        references = int_file_to_text_ls(open("obama/text_to_int.test.txt", 'r'), int_to_word)

    print("Removing _FILL_ tokens ...")
    generated = remove_filler(generated)
    references = remove_filler(references)


    text_ls_sample = random.choices(generated, k = 10)

    for text in text_ls_sample:
        print("--------------------------------")
        print(text)
    
    print("Calculating BLEUscore ...")
    #Calculate BlEUscore for whole corpus (Takes a while)
    BLEUscore = nltk.translate.bleu_score.corpus_bleu([references]*len(generated), generated)
    print()
    print("********" + "Corpus BLEUscore is: " + str(BLEUscore)+ "********")


if __name__ == '__main__':
    main()