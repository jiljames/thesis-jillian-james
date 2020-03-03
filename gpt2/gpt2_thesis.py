import gpt_2_simple as gpt2
import os
import requests
import sys

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


file_name = "all_theses.txt"
if not os.path.isfile(file_name):
    print("Thesis training data not present")
    sys.exit(0)
    

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              file_name,
              model_name=model_name,
              steps=1000)   # steps is max number of training steps

length = 5000
num_samples = 1
next_prefix = gpt2.generate(sess, prefix = "$!BEGIN!$", include_prefix=True, return_as_list = True, length=1023)[0]
full = []

current_length = len(next_prefix)
while next_prefix < length:
    next_prefix = gpt2.generate(sess, prefix = next_prefix, return_as_list = True, length=1023)[0]
    full.extend(next_prefix)
    current_length += len(next_prefix)

print("Current length: ", current_length)
full.extend(gpt2.generate(sess, prefix = next_prefix, truncate = "$!END!$", return_as_list = True, length=1023)[0])

sample_file_name = "sample.txt"
f = open("sample.txt", w)
for line in full:
    f.write("%s\n" % line)
