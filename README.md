# MEDMNIST-Classificcation

MedMNIST classification implementation used from here https://github.com/MedMNIST/experiments/tree/main for PathMNIST dataset with 128x128 resolution

ResNet18:
python train_and_eval.py --data_flag=bloodmnist --num_epochs=100 --batch_size=128 --as_rgb --model_flag=resnet18 --output_root=./output 

ResNet50:
python train_and_eval_128.py --data_flag=bloodmnist --num_epochs=100 --batch_size=128 --as_rgb --model_flag=resnet50 --output_root=./output
