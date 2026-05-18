import pickle
import os
import argparse
import random

from brainflow.board_shim import BoardShim
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.optimizers import AdamW
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import tensorflow as tf

from model import create_classifier
from pipeline import preprocess_data, extract_features

SAVE_FILENAME = "recorded_eeg"
SAVE_EXTENSION = ".pkl"


def segment_data(eeg_data, samples_per_window, overlap=0):
    _, total_samples = eeg_data.shape
    step_size = samples_per_window - overlap
    if step_size <= 0:
        raise ValueError("overlap must be smaller than samples_per_window")

    windows = []
    for start in range(0, total_samples - samples_per_window + 1, step_size):
        end = start + samples_per_window
        window = eeg_data[:, start:end]
        windows.append(window)

    return np.array(windows)


def get_recording_filenames():
    return sorted(
        d for d in os.listdir()
        if d.startswith(SAVE_FILENAME) and d.endswith(SAVE_EXTENSION)
    )


def load_and_merge_recordings(file_names):
    merged_action_dict = {}
    board_id = None
    window_seconds_values = set()

    for filename in file_names:
        print("Opening " + filename + "...")
        with open(filename, 'rb') as f:
            current_data = pickle.load(f)

        current_board_id = current_data["board_id"]
        current_window_seconds = current_data.get("window_seconds", None)
        current_action_dict = current_data.get("action_dict", {})

        if board_id is None:
            board_id = current_board_id
        elif current_board_id != board_id:
            raise ValueError(
                f"Training aborted. File {filename} uses board_id {current_board_id}, "
                f"but earlier files use board_id {board_id}. Do not mix board types in one training run."
            )

        if current_window_seconds is not None:
            window_seconds_values.add(current_window_seconds)

        for raw_action_idx, sessions in current_action_dict.items():
            raw_action_idx = int(raw_action_idx)
            if raw_action_idx not in merged_action_dict:
                merged_action_dict[raw_action_idx] = []

            merged_action_dict[raw_action_idx].extend(list(sessions))

    if board_id is None:
        raise FileNotFoundError(
            f"No {SAVE_FILENAME}*{SAVE_EXTENSION} files were found in the current directory."
        )

    return {
        "board_id": board_id,
        "window_seconds_values": sorted(window_seconds_values),
        "action_dict": merged_action_dict
    }


def print_session_summary(action_dict):
    print("\n=== Session Summary ===")
    for action_idx in sorted(action_dict.keys()):
        session_count = len(action_dict[action_idx])
        print(f"Action {action_idx}: {session_count} recorded session(s)")


def validate_sessions(action_dict):
    if len(action_dict) < 2:
        raise ValueError(
            "Training aborted. Fewer than 2 unique action indices were found across all recording files. "
            "Record at least two actions before training a classifier."
        )

    missing_actions = [action_idx for action_idx, datas in action_dict.items() if len(datas) == 0]
    if missing_actions:
        raise ValueError(
            "Training aborted. The following action indices have no recorded sessions: "
            + ", ".join(str(i) for i in missing_actions)
            + ". Record data for these actions before training."
        )


def print_action_mapping(raw_action_ids, raw_to_class_idx):
    print("\n=== Action Mapping ===")
    for raw_action_id in raw_action_ids:
        print(f"Raw action {raw_action_id} -> class {raw_to_class_idx[raw_action_id]}")


def print_window_summary(action_windows):
    print("\n=== Window Summary ===")
    total_train = 0
    total_test = 0

    for action_idx in sorted(action_windows.keys()):
        train_windows, test_windows = action_windows[action_idx]
        train_count = len(train_windows)
        test_count = len(test_windows)
        total_train += train_count
        total_test += test_count
        print(f"Action {action_idx}: {train_count} train window(s), {test_count} validation window(s)")

    print(f"Total: {total_train} train window(s), {total_test} validation window(s)")


def validate_windows(action_windows):
    missing_train = []
    missing_test = []

    for action_idx, (train_windows, test_windows) in action_windows.items():
        if len(train_windows) == 0:
            missing_train.append(action_idx)
        if len(test_windows) == 0:
            missing_test.append(action_idx)

    if missing_train or missing_test:
        messages = []
        if missing_train:
            messages.append(
                "no training windows for action(s): " + ", ".join(str(i) for i in missing_train)
            )
        if missing_test:
            messages.append(
                "no validation windows for action(s): " + ", ".join(str(i) for i in missing_test)
            )

        raise ValueError(
            "Training aborted because some classes became unusable after segmentation/splitting: "
            + "; ".join(messages)
            + ". Try recording more sessions or lowering --test-size."
        )


def warn_if_imbalanced(action_windows, imbalance_ratio_threshold=2.0):
    train_counts = {
        action_idx: len(train_windows)
        for action_idx, (train_windows, _) in action_windows.items()
    }

    nonzero_counts = [count for count in train_counts.values() if count > 0]
    if len(nonzero_counts) < 2:
        return

    min_count = min(nonzero_counts)
    max_count = max(nonzero_counts)

    if min_count == 0:
        return

    ratio = max_count / min_count

    print("\n=== Class Balance Check ===")
    for action_idx in sorted(train_counts.keys()):
        print(f"Action {action_idx}: {train_counts[action_idx]} training window(s)")
    print(f"Max/min train window ratio: {ratio:.2f}")

    if ratio >= imbalance_ratio_threshold:
        print(
            "WARNING: Class imbalance detected. "
            f"The largest class has {ratio:.2f}x as many training windows as the smallest class. "
            "This can bias the classifier toward the larger classes."
        )


def windows_from_datas(datas, eeg_channels, window_size, overlap, test_size, sample_size):
    eegs = [data[eeg_channels] for data in datas]
    windows_per_session = [
        segment_data(eeg, window_size, overlap)
        for eeg in eegs
        if eeg.shape[1] >= window_size
    ]

    if len(windows_per_session) == 0:
        return [], []

    all_windows = np.concatenate(windows_per_session)

    if len(all_windows) == 0:
        return [], []

    split_idx = int(len(all_windows) * (1 - test_size))

    if split_idx <= 0:
        split_idx = 1
    if split_idx >= len(all_windows):
        split_idx = len(all_windows) - 1

    if split_idx <= 0 or split_idx >= len(all_windows):
        return [], []

    windows_train = list(all_windows[:split_idx])
    windows_test = list(all_windows[split_idx:])

    sampled_train_count = int(len(windows_train) * sample_size)
    sampled_test_count = int(len(windows_test) * sample_size)

    if len(windows_train) > 0 and sampled_train_count <= 0:
        sampled_train_count = 1
    if len(windows_test) > 0 and sampled_test_count <= 0:
        sampled_test_count = 1

    if sampled_train_count < len(windows_train):
        windows_train = random.sample(windows_train, k=sampled_train_count)
    if sampled_test_count < len(windows_test):
        windows_test = random.sample(windows_test, k=sampled_test_count)

    return windows_train, windows_test


def process_windows(windows, sampling_rate):
    feature_windows = []
    for session_data in windows:
        preprocessed_data = preprocess_data(session_data, sampling_rate)
        features = extract_features(preprocessed_data)
        feature_windows.append(features)
    return feature_windows


def build_class_weight_dict(i_train):
    unique_classes = np.unique(i_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=i_train
    )
    return {int(cls): float(weight) for cls, weight in zip(unique_classes, weights)}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample-size', type=float, required=False, default=1.0,
                        help='random sample proportion of recorded data to use')
    parser.add_argument('--test-size', type=float, required=False, default=0.2,
                        help='proportion of sampled data to reserve for validation')
    parser.add_argument('--window-seconds', type=float, required=False, default=1.0,
                        help='length of each segmented training window in seconds')
    parser.add_argument('--overlap-percent', type=float, required=False, default=0.25,
                        help='window overlap percent from 0.0 to <1.0, e.g. 0.25 for 25 percent overlap')

    parser.add_argument('--learning-rate', type=float, required=False, default=1e-5,
                        help='AdamW learning rate')
    parser.add_argument('--batch-size', type=int, required=False, default=32,
                        help='training batch size')
    parser.add_argument('--epochs', type=int, required=False, default=500,
                        help='maximum number of epochs')
    parser.add_argument('--patience', type=int, required=False, default=50,
                        help='early stopping patience')
    parser.add_argument('--early-stopping-min-delta', type=float, required=False, default=1e-4,
                        help='minimum val_loss improvement required to reset early stopping patience')

    parser.add_argument('--weight-decay', type=float, required=False, default=1e-4,
                        help='AdamW weight decay')
    parser.add_argument('--label-smoothing', type=float, required=False, default=0.05,
                        help='label smoothing for categorical crossentropy, 0.0 disables it')
    parser.add_argument('--grad-clipnorm', type=float, required=False, default=1.0,
                        help='gradient clipnorm for AdamW, 0 disables it')

    parser.add_argument('--imbalance-ratio-threshold', type=float, required=False, default=2.0,
                        help='warn when max/min class train window ratio meets or exceeds this value')
    parser.add_argument('--use-class-weights', type=int, required=False, default=1,
                        help='1 to use balanced class weights during training, 0 to disable')

    parser.add_argument('--reduce-lr-on-plateau', type=int, required=False, default=1,
                        help='1 to reduce learning rate when validation loss plateaus, 0 to disable')
    parser.add_argument('--reduce-lr-factor', type=float, required=False, default=0.8,
                        help='factor applied when reducing learning rate')
    parser.add_argument('--reduce-lr-patience', type=int, required=False, default=10,
                        help='plateau patience before reducing learning rate')
    parser.add_argument('--reduce-lr-min', type=float, required=False, default=1e-6,
                        help='minimum learning rate for ReduceLROnPlateau')
    parser.add_argument('--reduce-lr-min-delta', type=float, required=False, default=1e-4,
                        help='minimum val_loss improvement required to avoid LR reduction')
    parser.add_argument('--reduce-lr-cooldown', type=int, required=False, default=2,
                        help='cooldown epochs after LR reduction')

    parser.add_argument('--head-hidden-units', type=int, required=False, default=64,
                        help='hidden units in the classifier head')
    parser.add_argument('--head-dropout-rate', type=float, required=False, default=0.4,
                        help='dropout rate for classifier head hidden layers')
    parser.add_argument('--head-l2', type=float, required=False, default=1e-4,
                        help='L2 regularization strength for classifier head dense layers')
    parser.add_argument('--use-second-head-dense', type=int, required=False, default=1,
                        help='1 to use a second dense layer in the classifier head, 0 to disable')
    parser.add_argument('--second-head-units', type=int, required=False, default=32,
                        help='hidden units in the optional second classifier head layer')
    parser.add_argument('--pooling-dropout-rate', type=float, required=False, default=0.15,
                        help='dropout rate applied after attention pooling')
    parser.add_argument('--use-sequence-shuffle', type=int, required=False, default=1,
                        help='1 to enable sequence shuffle during training, 0 to disable')
    parser.add_argument('--trainable-encoder-layers', type=int, required=False, default=0,
                        help='number of encoder layers from the end to leave trainable')

    parser.add_argument('--seed', type=int, required=False, default=42,
                        help='random seed for reproducibility')
    parser.add_argument('--tsne-perplexity', type=float, required=False, default=30.0,
                        help='t-SNE perplexity for latent visualization')
    parser.add_argument('--tsne-learning-rate', type=float, required=False, default=200.0,
                        help='t-SNE learning rate for latent visualization')
    parser.add_argument('--tsne-iterations', type=int, required=False, default=1000,
                        help='t-SNE iteration count for latent visualization')

    args = parser.parse_args()

    if not (0.0 < args.sample_size <= 1.0):
        raise ValueError("--sample-size must be > 0.0 and <= 1.0")
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test-size must be > 0.0 and < 1.0")
    if not (0.0 <= args.overlap_percent < 1.0):
        raise ValueError("--overlap-percent must be >= 0.0 and < 1.0")
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.patience < 0:
        raise ValueError("--patience must be >= 0")
    if args.label_smoothing < 0.0 or args.label_smoothing >= 1.0:
        raise ValueError("--label-smoothing must be >= 0.0 and < 1.0")
    if args.grad_clipnorm < 0.0:
        raise ValueError("--grad-clipnorm must be >= 0.0")
    if args.head_hidden_units <= 0:
        raise ValueError("--head-hidden-units must be > 0")
    if args.head_dropout_rate < 0.0 or args.head_dropout_rate >= 1.0:
        raise ValueError("--head-dropout-rate must be >= 0.0 and < 1.0")
    if args.head_l2 < 0.0:
        raise ValueError("--head-l2 must be >= 0.0")
    if args.second_head_units <= 0:
        raise ValueError("--second-head-units must be > 0")
    if args.pooling_dropout_rate < 0.0 or args.pooling_dropout_rate >= 1.0:
        raise ValueError("--pooling-dropout-rate must be >= 0.0 and < 1.0")
    if args.trainable_encoder_layers < 0:
        raise ValueError("--trainable-encoder-layers must be >= 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("Finding data files...")
    file_names = get_recording_filenames()

    if len(file_names) == 0:
        raise FileNotFoundError(
            f"No {SAVE_FILENAME}*{SAVE_EXTENSION} files were found in the current directory."
        )

    recorded_data = load_and_merge_recordings(file_names)

    board_id = recorded_data['board_id']
    action_dict = recorded_data['action_dict']

    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)

    window_size = int(args.window_seconds * sampling_rate)
    overlap = int(window_size * args.overlap_percent)

    print(f"\nUsing sampling rate: {sampling_rate}")
    print(f"Using window size: {window_size} samples ({args.window_seconds:.3f} seconds)")
    print(f"Using overlap: {overlap} samples ({args.overlap_percent * 100:.1f}%)")

    if len(recorded_data["window_seconds_values"]) > 1:
        print(
            "WARNING: Multiple source recording window lengths were found in the dataset files: "
            f"{recorded_data['window_seconds_values']}. "
            "Training will still use the current --window-seconds segmentation setting."
        )

    print_session_summary(action_dict)
    validate_sessions(action_dict)

    raw_action_ids = sorted(action_dict.keys())
    raw_to_class_idx = {raw_action_id: class_idx for class_idx, raw_action_id in enumerate(raw_action_ids)}
    class_to_raw_action = {class_idx: raw_action_id for raw_action_id, class_idx in raw_to_class_idx.items()}

    print_action_mapping(raw_action_ids, raw_to_class_idx)

    action_windows = {
        raw_action_id: windows_from_datas(
            datas,
            eeg_channels=eeg_channels,
            window_size=window_size,
            overlap=overlap,
            test_size=args.test_size,
            sample_size=args.sample_size
        )
        for raw_action_id, datas in action_dict.items()
    }

    print_window_summary(action_windows)
    validate_windows(action_windows)
    warn_if_imbalanced(action_windows, imbalance_ratio_threshold=args.imbalance_ratio_threshold)

    processed_windows = {
        raw_action_id: (
            process_windows(windows_train, sampling_rate),
            process_windows(windows_test, sampling_rate)
        )
        for raw_action_id, (windows_train, windows_test) in action_windows.items()
    }

    i_train_raw = np.concatenate([
        [raw_action_id] * len(windows_train)
        for raw_action_id, (windows_train, _) in processed_windows.items()
    ])
    i_train = np.array([raw_to_class_idx[action_id] for action_id in i_train_raw])

    shuffle_indexes = list(range(len(i_train)))
    random.shuffle(shuffle_indexes)

    X_train = np.concatenate([
        windows_train for windows_train, _ in processed_windows.values()
    ])[shuffle_indexes]
    y_train = to_categorical(i_train, num_classes=len(processed_windows))[shuffle_indexes]

    i_test_raw = np.concatenate([
        [raw_action_id] * len(windows_test)
        for raw_action_id, (_, windows_test) in processed_windows.items()
    ])
    i_test = np.array([raw_to_class_idx[action_id] for action_id in i_test_raw])

    X_test = np.concatenate([
        windows_test for _, windows_test in processed_windows.values()
    ])
    y_test = to_categorical(i_test, num_classes=len(processed_windows))

    class_weight = None
    if args.use_class_weights == 1:
        class_weight = build_class_weight_dict(i_train)
        print("\nUsing class weights:")
        for class_idx in sorted(class_weight.keys()):
            raw_action_id = class_to_raw_action[class_idx]
            print(f"Class {class_idx} (raw action {raw_action_id}): {class_weight[class_idx]:.4f}")

    pretrained_encoder = keras.models.load_model("physionet_encoder.keras")

    classes = len(processed_windows)
    input_shape = X_train.shape[1:]

    model = create_classifier(
        pretrained_encoder,
        classes,
        input_shape,
        head_hidden_units=args.head_hidden_units,
        head_dropout_rate=args.head_dropout_rate,
        head_l2=args.head_l2,
        use_second_head_dense=args.use_second_head_dense,
        second_head_units=args.second_head_units,
        pooling_dropout_rate=args.pooling_dropout_rate,
        use_sequence_shuffle=args.use_sequence_shuffle,
        trainable_encoder_layers=args.trainable_encoder_layers
    )

    optimizer_kwargs = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay
    }
    if args.grad_clipnorm > 0:
        optimizer_kwargs["clipnorm"] = args.grad_clipnorm

    model.compile(
        optimizer=AdamW(**optimizer_kwargs),
        loss=CategoricalCrossentropy(label_smoothing=args.label_smoothing)
    )

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            min_delta=args.early_stopping_min_delta,
            restore_best_weights=True,
            verbose=1
        )
    ]

    if args.reduce_lr_on_plateau == 1:
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=args.reduce_lr_factor,
                patience=args.reduce_lr_patience,
                min_lr=args.reduce_lr_min,
                min_delta=args.reduce_lr_min_delta,
                cooldown=args.reduce_lr_cooldown,
                verbose=1
            )
        )

    fit_history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    model.summary()
    model.save('shallow.keras')

    predictions_prob = model.predict(X_test)
    predictions = np.argmax(predictions_prob, axis=1)
    y_test_idxs = np.argmax(y_test, axis=1)

    print("Model evaluation:")
    model.evaluate(X_test, y_test, verbose=1)

    target_names = [
        f"raw_action_{class_to_raw_action[class_idx]}"
        for class_idx in range(classes)
    ]
    print(classification_report(y_test_idxs, predictions, target_names=target_names))

    plt.style.use('dark_background')

    plt.figure()
    plt.plot(fit_history.history['loss'])
    plt.plot(fit_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylim(0, max(1.0, float(np.max(fit_history.history['val_loss'])) + 0.1))
    plt.savefig('loss.png')

    seq_model = Sequential(model.layers[:-1])
    latent = seq_model(X_test)

    samples = latent.shape[0]
    latent_flat = tf.reshape(latent, (samples, -1)).numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(latent_flat)

    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        learning_rate=args.tsne_learning_rate,
        n_iter=args.tsne_iterations,
        random_state=args.seed
    )
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=i_test, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Class Index')
    plt.title('t-SNE Visualization of Labeled Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.axis('square')
    plt.savefig('tsne.png')


if __name__ == "__main__":
    main()