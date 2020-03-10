import gpt_2_simple as gpt2
import os
import requests
import sys
import tensorflow as tf


TRAIN_PATH= "clean_etheses"
TRAIN_STEPS=3000
LENGTH = 10*1023
NUM_SAMPLES = 5
PREFIX= ''' 
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



os.environ["CUDA_VISIBLE_DEVICES"]="2" 

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/




config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              dataset=TRAIN_PATH,
              model_name=model_name,
              steps=TRAIN_STEPS)   # steps is max number of training steps



for i in range(NUM_SAMPLES):
    prefix = PREFIX
    current_length = len(PREFIX)
    full = [PREFIX]

    while current_length < LENGTH - 1023:
        gen = gpt2.generate(sess, prefix = prefix,
                return_as_list = True, length=1023)[0][len(prefix):]
        print("Length generated:", len(gen))
        full.append(gen)
        prefix = gen[len(gen)//2:]
        current_length += len(gen)

    final_text = gpt2.generate(sess, prefix = prefix,
            truncate = "$!END!$", return_as_list = True, length=1023)[0][len(prefix):]
    full.append(final_text)


    print(full)
    sample_file_name = "./sample"+str(i)+".txt"
    with open(sample_file_name , 'w') as f:
        for text in full:
            f.write("%s\n" % text)
            f.write("_________________________________________\n")
