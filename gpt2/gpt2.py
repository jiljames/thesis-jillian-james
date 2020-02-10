import gpt_2_simple as gpt2
import os
import requests
import sys


# https://github.com/minimaxir/gpt-2-simple


def main():
    
    #  Create a parser to parse user input
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Program for running several SeqGan applications.')
        parser.add_argument('app', metavar='application', type=str, choices=['obama', 'haiku', 'synth'],
                        help='Enter either \'obama\' or \'haiku\'')
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
    
        return task, synth_gen_params
    
    task, SYNTH_GEN_PARAMS = parse_arguments()

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
              steps=1000)   # steps is max number of training steps

gpt2.generate(sess, n_samples = 10)

