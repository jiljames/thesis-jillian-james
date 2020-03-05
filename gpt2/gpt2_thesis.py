import gpt_2_simple as gpt2
import os
import requests
import sys

LENGTH = 5*1023
NUM_SAMPLES = 3
PREFIX= ''' 
$!BEGIN!$
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
'''
os.environ["CUDA_VISIBLE_DEVICES"]="1,2" 

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


file_name = "all_theses.txt"
if not os.path.isfile(file_name):
    print("Thesis training data not present")
    sys.exit(0)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              file_name,
              model_name=model_name,
              steps=1000)   # steps is max number of training steps


for i in range(NUM_SAMPLES):
    full = [PREFIX]
    next_prefix = PREFIX
    current_length = len(PREFIX)

    while current_length < LENGTH:
        next_prefix = gpt2.generate(sess, prefix = next_prefix,
                    return_as_list = True, length=1023)[0][len(next_prefix:)]
        full.append(next_prefix)
        current_length += len(next_prefix)

    full.append(gpt2.generate(sess, prefix = next_prefix,
                truncate = "$!END!$", return_as_list = True, length=1023)[0[len(next_prefix):]])

    sample_file_name = "./sample"+str(i)+".txt"
    if not os.path.exists(sample_file_name):
                        os.mknod(sample_file_name)
    f = open(sample_file_name , 'w')
    for sample in full:
        f.write("%s\n" % sample)
        f.write("__________________________________________")