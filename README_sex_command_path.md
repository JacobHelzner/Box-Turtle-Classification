# Sex-Classification Command Path (Augmented Train + Standard Test Features)

This documents the reproducible command path for the sex-classification pipeline:
- Augment train images (rotation, light crop, color jitter)
- Extract ResNet-101 features for augmented train data
- Extract ResNet-101 features for standard (non-augmented) test data

## Step 1: Build ID-level Train/Test Split (if needed)

If split files already exist and you are keeping the same random seed, you can skip this.

```bash
python prepare_sex_data.py --exclude-juvenile-sex
python split_sex_data.py
python check_sex_split.py
```

Expected key outputs:
- sex_train.csv
- sex_test.csv

## Step 2: Augment Train Split

This creates one augmented image per input train image and writes a new train CSV pointing to the augmented image paths.

```bash
python augment_sex_train_data.py \
  --input-csv sex_train.csv \
  --output-csv sex_train_augmented.csv \
  --output-image-dir sex_train_augmented_images \
  --preview-out sex_train_augmentation_preview.jpg
```

Expected key outputs:
- sex_train_augmented.csv
- sex_train_augmented_images/
- sex_train_augmentation_preview.jpg

## Step 3: Extract ResNet-101 Features for Train (Augmented) + Test (Standard)

This command uses the augmented train CSV as train input and the original test CSV as test input.

```bash
python sex_classification_resnet101.py \
  --input-csv sex_train_augmented.csv \
  --output-csv sex_train_features_resnet101.csv \
  --test-csv sex_test.csv \
  --test-output-csv sex_test_features_resnet101.csv
```

Expected key outputs:
- sex_train_features_resnet101.csv
- sex_test_features_resnet101.csv

## Notes

- Train is augmented; test is kept deterministic and non-augmented.
- If you want stronger augmentation later, edit parameters in augment_sex_train_data.py.
- sex_head_training_loop.py is reserved for the classifier-head training logic.
