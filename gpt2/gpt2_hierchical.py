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
            help = "Model type. Represents number of parameters")                    
    parser.add_argument('-steps', type = int, default = 0,
                    help = "Number of finetune steps for GPT2. ")
    parser.add_argument('-title', type = str, default = "Hierarchical Generation For Lengthy Language Tasks" ,
                    help = "Title for the generated thesis")
    args = parser.parse_args()
    
    if not os.path.isdir(args.output):
        sys.exit("Incorrect path provided to -output argument required for generation.")

    # if os.path.isfile(args.prefix):
    #     f = open(args.prefix, "r")
    #     prefix = "".join(f.readlines())
    #     f.close()
    # else:
    #     prefix = args.prefix

    return args.model, args.output, args.steps, args.title

GEN_LENGTH = 250
NUM_SAMPLES = 5
TRAIN_TOPIC_PATH = "./train_topics"
VALID_TOPIC_PATH = "./test_topics"
TRAIN_PATH = "./train_clean_etheses"
VALID_PATH = "./test_clean_etheses"

MODEL, OUTPUT_DIR, TRAIN_STEPS, TITLE=  parse_arguments()


#For Mark's machine
os.environ["CUDA_VISIBLE_DEVICES"]="1,2" 


if not os.path.isdir(os.path.join("models",  MODEL)):
    print(f"Downloading {MODEL} model...")
    gpt2.download_gpt2(model_name=MODEL)   # model is saved into current directory under /models/124M/


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = gpt2.start_tf_sess()


#gpt2.load_gpt2(sess, model_name = MODEL)
# Train Topic generator
print("TRAINING TOPIC GENERATOR...")
if TRAIN_STEPS > 0:
    gpt2.finetune(sess,
                dataset=TRAIN_TOPIC_PATH,
                model_name=MODEL,
                steps=TRAIN_STEPS,
                checkpoint_dir= "topic_gen_chkpts",
                multi_gpu = True,
                val_dataset = VALID_TOPIC_PATH,
                val_every = 10
                )

    #gpt2.load_gpt2(sess, model_name = MODEL)
    tf.get_variable_scope().reuse_variables()

    sess = gpt2.reset_session(sess)
    # Train Thesis generator
    print("TRAINING THESIS GENERATOR...")
    gpt2.finetune(sess,
                dataset=TRAIN_PATH,
                model_name=MODEL,
                steps=TRAIN_STEPS,
                checkpoint_dir= "thesis_gen_chkpts",
                multi_gpu = True,
                val_dataset = VALID_PATH,
                val_every = 10
                )


for i in range(NUM_SAMPLES):
    full = []
    sess = gpt2.reset_session(sess)
    gpt2.load_gpt2(sess, checkpoint_dir="topic_gen_chkpts")
    topic = gpt2.generate(sess, checkpoint_dir="topic_gen_chkpts", return_as_list=True, prefix = TITLE)[0]
    prefixes = topic.split("<|SEP|>")
    print("NUM PREFIXES = ", len(prefixes))

    sess = gpt2.reset_session(sess)
    gpt2.load_gpt2(sess, checkpoint_dir="thesis_gen_chkpts")
    for prefix in prefixes:
        gen = gpt2.generate(sess, checkpoint_dir="thesis_gen_chkpts", prefix = prefix, return_as_list = True,
                            length=GEN_LENGTH)[0]
        print("Length generated:", len(gen)-len(prefix))
        full.append(gen)

    # final_text = gpt2.generate(sess, prefix = prefix,
    #         truncate = "$!END!$", return_as_list = True, length=GEN_LENGTH, model_name = model_name)[0][len(prefix):]
    # full.append(final_text)


    print(full)
    sample_file_name = OUTPUT_DIR+"/sample"+str(i)+".txt"
    with open(sample_file_name , 'w') as f:
        for text in full:
            f.write("%s\n" % text)
            f.write("_________________________________________\n")
    topic_file_name = OUTPUT_DIR+"/topic"+str(i)+".txt"
    with open(topic_file_name, "w") as f:
        f.write(topic)
