#!/bin/bash
echo "Running gan.py synth 10 10 10"
python3 gan.py synth 10 10 10
echo "Running gan.py synth 20 20 20"
python3 gan.py synth 20 20 20
echo "Running gan.py synth 30 30 30"
python3 gan.py synth 30 30 30
echo "Running gan.py synth 40 40 40"
python3 gan.py synth 40 40 40
echo "Running gan.py synth 50 50 50"
python3 module_sequence_gan.py synth 50 50 50
echo "Running gan.py synth 30 30 1"
python3 module_sequence_gan.py synth 30 30 1
echo "Running gan.py synth 30 30 5"
python3 module_sequence_gan.py synth 30 30 5
echo "Running gan.py synth 30 30 10"
python3 module_sequence_gan.py synth 30 30 10
echo "Running gan.py synth 30 30 20"
python3 module_sequence_gan.py synth 30 30 20
echo "Running gan.py synth 30 30 30"
python3 module_sequence_gan.py synth 30 30 30
echo "Running gan.py synth 30 30 40"
python3 module_sequence_gan.py synth 30 30 40
echo "Running gan.py synth 30 30 50"
python3 module_sequence_gan.py synth 30 30 50
