import gpt_2_simple as gpt2
import os
import requests
import sys
import argparse
from datautil import load_task
from synthetic import generate_random_sents, is_valid_phrase


# https://github.com/minimaxir/gpt-2-simple


def main():
    
    #  Create a parser to parse user input
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Program for running several SeqGan applications.')
        parser.add_argument('app', metavar='application', type=str, choices=['obama', 'haiku', 'synth'],
                        help='Enter either \'obama\' or \'haiku\'')
        parser.add_argument('n', metavar='num_steps', type=int, help= "The number of training steps")
        parser.add_argument('-numeat', metavar="num_eat", type = int, default = 500,
                        help = "For synthetic data generation. Determines number of eaters in vocab.")
        parser.add_argument('-numfeed', metavar="num_feed", type = int, default = 500,
                        help = "For synthetic data generation. Determines number of feeders in vocab.")
        parser.add_argument('-numsent', metavar="num_sent", type = int, default = 10000,
                        help = "For synthetic data generation. Determines number of sentences generated.")
        args = parser.parse_args()

        synth_gen_params = ("NA", "NA", "NA")
        if args.app == "synth":
            synth_gen_params = (args.numsent, args.numfeed, args.numeat)
            generate_random_sents("../data/synth/input.txt", args.numsent, args.numfeed, args.numeat)

        task = load_task(args.app)
    
        return task, args.n synth_gen_params
    
    task, num_steps, SYNTH_GEN_PARAMS = parse_arguments()

    model_name = "124M"
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/

    file_name = task.train_file
    if not os.path.isfile(file_name):
        print("Training data not present for " task.name)
        sys.exit(0)
        

    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess,
                file_name,
                model_name=model_name,
                steps=num_steps, restore_from="fresh")   # steps is max number of training steps


    gpt2.generate_to_file(sess,destination_path=task.eval_file,  nsamples = 10)



    #Writing results to CSV
    with open(task.eval_file) as f:
        generated = []
        for line in f:
            line = line.strip().split()
            generated.append(line)
        generated = task.vocab.decode(generated)
        f.close()

    with open(task.test_file) as f:
        references = []
        for line in f:
            line = line.strip().split()
            references.append(line)
        references = task.vocab.decode(references)  
        f.close()      

        
    if not os.path.exists("./results.csv"):
        os.mknod("./results.csv")

    with open("./results.csv", 'a') as csvfile:
        fieldnames = ["name", "task_name", "num_gen", "num_disc", "num_adv",
                    "num_sents", "num_feeders", "num_eaters", "BLEU", "prop_valid"]
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        csvfile.seek(0, os.SEEK_END) # go to end of file
        if not csvfile.tell(): # if current position is != 0)
            writer.writeheader()

        blue = corpus_bleu([references]*len(generated), generated)
        print("Run with args {} {} {}: BLEUscore = {}\n".format(gen_n, disc_n, adv_n, blue))
        
        prop = "NA"

        if task.name == "synth":
            total_correct = 0
            for sentence in generated:
                if is_valid_phrase(sentence):
                    total_correct +=1
            prop = total_correct/len(generated)

        writer.writerow({"name": MODEL_STRING, "task_name": task.name,  "num_gen": gen_n, 
                        "num_disc":disc_n, "num_adv": adv_n, "num_sents":SYNTH_GEN_PARAMS[0],
                        "num_feeders":SYNTH_GEN_PARAMS[1], "num_eaters":SYNTH_GEN_PARAMS[2],
                        "BLEU": blue, "prop_valid": prop})
        f.close()

if __name__ == "__main__":
    main()