import gpt_2_simple as gpt2
import os
import requests
import sys
import tensorflow as tf

GEN_LENGTH = 1023
LENGTH = 10*GEN_LENGTH
NUM_SAMPLES = 5
PREFIX = '''
<|startoftext|>
How Laziness Leads to Ingenuity: Exploring Lengthy Language Generation

A Thesis
Presented to
Mathematics and Natural Sciences
Reed College

In Partial Fulfillment
of the Requirements for the Degree
Bachelor of Arts

Jillian James
May 2020

Approved for the Division
(Mathematics)
Mark Hopkins

Introduction
While language generation has a large number of applications, each of these applications begins with
data. In order to do natural language generation you must first have data that you would like to convert
'''


 #  Create a parser to parse user input
def parse_arguments():
    parser = argparse.ArgumentParser(description='Program for running GPT2.')

    parser.add_argument('-model', type = str, default = "124M", choices=['124M', '355M']
                    help = "Path of training data. Can be file or directory")              
    parser.add_argument('-output', metavar= "sample_path", type = str, default = "",
                    help = "Path of dir for gernerated samples to write to.")      
    parser.add_argument('-data', metavar="data_path", type = str, default = "",
                    help = "Path of training data. Can be file or directory.")
    parser.add_argument('-train', type = int, default = 0,
                    help = "Number of finetune steps for GPT2. ")
    parser.add_argument('-prefix', type = int, default = "<|startoftext|>",
                    help = "Prefix for generation. Either string or filename.")
    args = parser.parse_args()

    if args.train > 0:
        if not os.isdir(args.data) or not os.isdir(args.data):
            sys.exit("Incorrect path provided to -data argument required for training.")
    
    
    if not os.isdir(args.output):
        sys.exit("Incorrect path provided to -output argument required for generation.")

    if os.isfile(args.prefix):
        f = open(args.prefix, "r")
        prefix = "".join(f.readlines())
        f.close()
    
    return args.model, args.output, args.data, args.train, args.prefix



MODEL, OUTPUT_DIR, TRAIN_PATH, TRAIN_STEPS, PREFIX =  parse_arguments()



#For Mark's machine
os.environ["CUDA_VISIBLE_DEVICES"]="1,2" 


if not os.path.isdir(os.path.join("models",  MODEL)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = gpt2.start_tf_sess()

if TRAIN_STEPS > 0:
    gpt2.finetune(sess,
                dataset=TRAIN_PATH,
                model_name=model_name,
                steps=TRAIN_STEPS)   # steps is max number of training steps
                multi_gpu = TRUE
else:
    gpt2.load_gpt2(sess)



for i in range(NUM_SAMPLES):
    prefix = PREFIX
    current_length = len(PREFIX)
    full = [PREFIX]

    while current_length < LENGTH - GEN_LENGTH:
        gen = gpt2.generate(sess, prefix = prefix,
                return_as_list = True, length=GEN_LENGTH)[0][len(prefix):]
        print("Length generated:", len(gen))
        full.append(gen)
        prefix = gen[len(gen)//2:]
        current_length += len(gen)

    final_text = gpt2.generate(sess, prefix = prefix,
            truncate = "$!END!$", return_as_list = True, length=GEN_LENGTH)[0][len(prefix):]
    full.append(final_text)


    print(full)
    sample_file_name = OUTPUT_DIR+"/sample"+str(i)+".txt"
    with open(sample_file_name , 'w') as f:
        for text in full:
            f.write("%s\n" % text)
            f.write("_________________________________________\n")
