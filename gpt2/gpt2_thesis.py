#import gpt_2_simple as gpt2
import gpt_2 as gpt2
import os
import requests
import sys
import tensorflow as tf
import argparse
import sys

GEN_LENGTH = 1023
LENGTH = 10*GEN_LENGTH
NUM_SAMPLES = 5
# <|startoftext|>
# How Laziness Leads to Ingenuity: Exploring Lengthy Language Generation

# A Thesis
# Presented to
# Mathematics and Natural Sciences
# Reed College

# In Partial Fulfillment
# of the Requirements for the Degree
# Bachelor of Arts

# Jillian James
# May 2020

# Approved for the Division
# (Mathematics)
# Mark Hopkins

# Introduction
# While language generation has a large number of applications, each of these applications begins with
# data. In order to do natural language generation you must first have data that you would like to convert
# '''
# phrase1 = '''
# Extended Red Emission (ERE) is a widely observed optical emission process, present in a wide range
# of circumstellar and interstellar environments in the Milky Way galaxy as well as other galaxies.
# '''
# phrase2 = '''
# I consider decision-making constrained by considerations of morality, rationality, or other virtues.
# The decision maker (DM) has a true preference over outcomes, but feels compelled to choose among
# outcomes that are top-ranked by some preference that he considers "justifiable." 
# '''
# phrase3 = '''
# Cascading large-amplitude bursts in neural activity, termed avalanches, are thought to provide insight
# into the complex spatially distributed interactions in neural systems. 
# '''
# PREFIX = [phrase1, phrase2, phrase3]



 #  Create a parser to parse user input
def parse_arguments():
    parser = argparse.ArgumentParser(description='Program for running GPT2.')
    parser.add_argument('output', metavar= "sample_path", type = str, default = "",
                    help = "Path of dir for gernerated samples to write to.")
    parser.add_argument('-model', type = str, default = "124M", choices=['124M', '355M', '774M'],
                    help = "Path of training data. Can be file or directory")                    
    parser.add_argument('-data', metavar="data_path", type = str, default = "",
                    help = "Path of training data. Can be file or directory.")
    parser.add_argument('-train', type = int, default = 0,
                    help = "Number of finetune steps for GPT2. ")
    parser.add_argument('-prefix', type = str, default = "<|startoftext|>",
                    help = "Prefix for generation. Either string or filename.")
    args = parser.parse_args()

    if args.train > 0:
        if not os.path.isdir(args.data) or not os.path.isdir(args.data):
            sys.exit("Incorrect path provided to -data argument required for training.")
    
    
    if not os.path.isdir(args.output):
        sys.exit("Incorrect path provided to -output argument required for generation.")

    if os.path.isfile(args.prefix):
        f = open(args.prefix, "r")
        prefix = "".join(f.readlines())
        f.close()
    else:
        prefix = args.prefix

    return args.model, args.output, args.data, args.train, prefix



MODEL, OUTPUT_DIR, TRAIN_PATH, TRAIN_STEPS, PREFIX =  parse_arguments()

VALID_PATH = "valid_clean_etheses"

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
