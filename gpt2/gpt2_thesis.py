#import gpt_2_simple as gpt2
import gpt_2 as gpt2
import os
import requests
import sys
import tensorflow as tf
import argparse
import sys



 #  Create a parser to parse user input
def parse_arguments():
    parser = argparse.ArgumentParser(description='Program for running GPT2.')
    parser.add_argument('output', metavar= "sample_path", type = str, default = "",
                    help = "Path of dir for gernerated samples to write to.")
    parser.add_argument('-model', type = str, default = "124M", choices=['124M', '355M', '774M'],
                    help = "Path of training data. Can be file or directory")                    
    parser.add_argument('-train', metavar="path", type = str, default = "",
                    help = "Path of training dataset. Can be file or directory.")
    parser.add_argument('-valid', metavar="path", type = str, default = "",
                    help = "Path of validation dataset. Can be file or directory.")
    parser.add_argument('-steps', type = int, default = 0,
                    help = "Number of finetune steps for GPT2. ")
    parser.add_argument('-prefix', type = str, default = "<|startoftext|>",
                    help = "Prefix for generation. Either string or filename.")
    args = parser.parse_args()

    if args.steps > 0:
        if not os.path.isdir(args.train) or not os.path.isdir(args.train):
            sys.exit("A correct path for the training dataset must be passed with -train argument in order to finetune.")
        if not os.path.isdir(args.valid) or not os.path.isdir(args.valid):
            sys.exit("A correct path for the validation dataset must be passed with -valid argument in order to finetune.")
            
    
    
    if not os.path.isdir(args.output):
        sys.exit("Incorrect path provided to -output argument required for generation.")

    if os.path.isfile(args.prefix):
        f = open(args.prefix, "r")
        prefix = "".join(f.readlines())
        f.close()
    else:
        prefix = args.prefix

    return args.model, args.output, args.train, args.valid, args.steps, prefix

GEN_LENGTH = 1023
LENGTH = 10*GEN_LENGTH
NUM_SAMPLES = 5


MODEL, OUTPUT_DIR, TRAIN_PATH, VALID_PATH, TRAIN_STEPS, PREFIX =  parse_arguments()


#For Mark's machine
os.environ["CUDA_VISIBLE_DEVICES"]="1,2" 


if not os.path.isdir(os.path.join("models",  MODEL)):
    print(f"Downloading {MODEL} model...")
    gpt2.download_gpt2(model_name=MODEL)   # model is saved into current directory under /models/124M/


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = gpt2.start_tf_sess()

if TRAIN_STEPS > 0:
    gpt2.finetune(sess,
                dataset=TRAIN_PATH,
                model_name=MODEL,
                steps=TRAIN_STEPS,
                multi_gpu = True,
                val_dataset = VALID_PATH,
                val_every = 10
                )
    model_name = None
else:
    model_name =MODEL
    gpt2.load_gpt2(sess, model_name = model_name)



for i in range(NUM_SAMPLES):
    prefix = PREFIX
    current_length = len(PREFIX)
    full = [PREFIX]

    while current_length < LENGTH - GEN_LENGTH:
        gen = gpt2.generate(sess, prefix = prefix,
                return_as_list = True, length=GEN_LENGTH, model_name = model_name)[0][len(prefix):]
        print("Length generated:", len(gen))
        full.append(gen)
        prefix = gen[len(gen)//2:]
        current_length += len(gen)

    final_text = gpt2.generate(sess, prefix = prefix,
            truncate = "$!END!$", return_as_list = True, length=GEN_LENGTH, model_name = model_name)[0][len(prefix):]
    full.append(final_text)


    print(full)
    sample_file_name = OUTPUT_DIR+"/sample"+str(i)+".txt"
    with open(sample_file_name , 'w') as f:
        for text in full:
            f.write("%s\n" % text)
            f.write("_________________________________________\n")
