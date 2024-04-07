import subprocess
import os

script_path1 = os.path.abspath("maps/densenet/brain_tumor_classiciation_DenseNet.py")
script_path2 = os.path.abspath("maps/resnet50/resnet50_outputs_map_trained.py")
script_path3 = os.path.abspath("maps/resnet101/brain_tumor_classification_RESNET101_trained.py")
script_path4 = os.path.abspath("maps/resnet152/brain_tumor_classification_RESNET152_trained.py")
script_path6 = os.path.abspath("maps/vgg16/brain_tumor_classification_VGG16_trained.py")
script_path7 = os.path.abspath("maps/densenet_untrain/brain_tumor_classiciation_DenseNet_untrained.py")
script_path8 = os.path.abspath("maps/resnet50_untrain/resnet50_outputs_map_untrained.py")
script_path9 = os.path.abspath("maps/resnet101_untrain/brain_tumor_classification_RESNET101_untrained.py")
script_path10 = os.path.abspath("maps/resnet152_untrain/brain_tumor_classification_RESNET152_untrained.py")
script_path11 = os.path.abspath("maps/vgg16_untrain/brain_tumor_classification_VGG16_untrained.py")

try:
    subprocess.run(["python", script_path1], check=True)
    print("\nbrain_tumor_classiciation_DenseNet.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nbrain_tumor_classiciation_DenseNet.py execution failed with error:", e)

try:
    subprocess.run(["python", script_path2], check=True)
    print("\nresnet50_outputs_map_trained.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nresnet50_outputs_map_trained.py execution failed with error:", e)

try:
    subprocess.run(["python", script_path3], check=True)
    print("\nbrain_tumor_classification_RESNET101_trained.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nbrain_tumor_classification_RESNET101_trained.py execution failed with error:", e)

try:
    subprocess.run(["python", script_path4], check=True)
    print("\nbrain_tumor_classification_RESNET152_trained.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nbrain_tumor_classification_RESNET152_trained.py execution failed with error:", e)

try:
    subprocess.run(["python", script_path6], check=True)
    print("\nbrain_tumor_classification_VGG16_trained.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nbrain_tumor_classification_VGG16_trained.py execution failed with error:", e)

try:
    subprocess.run(["python", script_path7], check=True)
    print("\nbrain_tumor_classiciation_DenseNet_untrained.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nbrain_tumor_classiciation_DenseNet_untrained.py execution failed with error:", e)

try:
    subprocess.run(["python", script_path8], check=True)
    print("\nresnet50_outputs_map_untrained.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nresnet50_outputs_map_untrained.py execution failed with error:", e)

try:
    subprocess.run(["python", script_path9], check=True)
    print("\nbrain_tumor_classification_RESNET101_untrained.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nbrain_tumor_classification_RESNET101_untrained.py execution failed with error:", e)

try:
    subprocess.run(["python", script_path10], check=True)
    print("\nbrain_tumor_classification_RESNET152_untrained.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nbrain_tumor_classification_RESNET152_untrained.py execution failed with error:", e)

try:
    subprocess.run(["python", script_path11], check=True)
    print("\nbrain_tumor_classification_VGG16_untrained.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("\nbrain_tumor_classification_VGG16_untrained.py execution failed with error:", e)

