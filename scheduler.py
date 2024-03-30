import subprocess
import os

script_path = os.path.abspath("brain_tumor_classification_RESNET101_untrained.py")

try:
    subprocess.run(["python", script_path], check=True)
    print("brain_tumor_classification_RESNET101_untrained.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("brain_tumor_classification_RESNET101_untrained.py execution failed with error:", e)
    exit(1)

script_path = os.path.abspath("brain_tumor_classiciation_RESNET50_trained.py")

try:
    subprocess.run(["python", script_path], check=True)
    print("brain_tumor_classiciation_RESNET50_trained.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("brain_tumor_classiciation_RESNET50_trained.py execution failed with error:", e)
    exit(1)

script_path = os.path.abspath("brain_tumor_classiciation_RESNET50_untrained.py")

try:
    subprocess.run(["python", script_path], check=True)
    print("brain_tumor_classiciation_RESNET50_untrained.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("brain_tumor_classiciation_RESNET50_untrained.py execution failed with error:", e)
    exit(1)

script_path = os.path.abspath("brain_tumor_classification_RESNET152_trained.py")

try:
    subprocess.run(["python", script_path], check=True)
    print("brain_tumor_classification_RESNET152_trained.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("brain_tumor_classification_RESNET152_trained.py execution failed with error:", e)
    exit(1)

script_path = os.path.abspath("brain_tumor_classification_RESNET152_untrained.py")

try:
    subprocess.run(["python", script_path], check=True)
    print("brain_tumor_classification_RESNET152_untrained.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("brain_tumor_classification_RESNET152_untrained.py execution failed with error:", e)
    exit(1)

script_path = os.path.abspath("brain_tumor_classification_VGG16_trained.py")

try:
    subprocess.run(["python", script_path], check=True)
    print("brain_tumor_classification_VGG16_trained.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("brain_tumor_classification_VGG16_trained.py execution failed with error:", e)
    exit(1)

script_path = os.path.abspath("brain_tumor_classification_VGG16_untrained.py")

try:
    subprocess.run(["python", script_path], check=True)
    print("brain_tumor_classification_VGG16_untrained.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("brain_tumor_classification_VGG16_untrained.py execution failed with error:", e)
    exit(1)


script_path = os.path.abspath("brain_tumor_classiciation_UNET.py")

try:
    subprocess.run(["python", script_path], check=True)
    print("brain_tumor_classiciation_UNET.py execution completed successfully.")
except subprocess.CalledProcessError as e:
    print("brain_tumor_classiciation_UNET.py execution failed with error:", e)
    exit(1)