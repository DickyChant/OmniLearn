import os
import numpy as np
import tensorflow as tf
from sklearn import metrics
import horovod.tensorflow.keras as hvd
import argparse
from scipy.special import softmax

from PET import PET, get_logsnr_alpha_sigma
import utils


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate generative classification using diffusion likelihood and compare with classifier head.")
    parser.add_argument("--dataset", type=str, default="top", help="Dataset name")
    parser.add_argument("--folder", type=str, default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
    parser.add_argument("--batch", type=int, default=500, help="Batch size")
    parser.add_argument("--load", action='store_true', help="Load pre-evaluated npy files")
    parser.add_argument("--mode", type=str, default="all", help="Loss type the model was trained with")
    parser.add_argument("--fine_tune", action='store_true', help="Fine tune a model")
    parser.add_argument("--nid", type=int, default=0, help="Training ID for multiple trainings")
    parser.add_argument("--local", action='store_true', help="Use local embedding")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
    parser.add_argument("--simple", action='store_true', help="Use simplified head model")
    parser.add_argument("--talking_head", action='store_true', help="Use talking head attention")
    parser.add_argument("--layer_scale", action='store_true', help="Use layer scale in the residual connections")
    parser.add_argument("--num_timesteps", type=int, default=100, help="Number of MC timesteps for likelihood estimation")
    parser.add_argument("--weighting", type=str, default="uniform", choices=["uniform", "snr"],
                        help="Timestep weighting: uniform or SNR-weighted ELBO")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--freeze_body", action='store_true', help="Use PEFT checkpoint (frozen body)")
    parser.add_argument("--freeze_heads", action='store_true', help="Frozen heads checkpoint")
    parser.add_argument("--lora_rank", type=int, default=0, help="LoRA rank for body adaptation (0=disabled)")
    parser.add_argument("--pretrained_only", action='store_true',
                        help="Evaluate pretrained JetClass checkpoint directly (no fine-tuning, untrained baseline)")
    args = parser.parse_args()
    return args


def print_metrics(y_pred, y, thresholds, multi_label=False):
    if multi_label:
        print("AUC: {}".format(metrics.roc_auc_score(y, y_pred, average='macro', multi_class='ovo')))

        one_hot_predictions = np.zeros_like(y_pred)
        one_hot_predictions[np.arange(len(y_pred)), y_pred.argmax(axis=-1)] = 1

        print('Acc: {}'.format(metrics.accuracy_score(y, one_hot_predictions)))

        bkg_idx = 0

        for idx in range(np.shape(y)[-1]):
            if idx == bkg_idx: continue
            mask = (y[:, idx] == 1) | (y[:, bkg_idx] == 1)
            pred_sb = y_pred[mask, idx] / (y_pred[mask, idx] + y_pred[mask, bkg_idx])
            fpr, tpr, _ = metrics.roc_curve(y[mask, idx], pred_sb)

            for threshold in thresholds:
                bineff = np.argmax(tpr > threshold)
                print('Class {} effS at {} 1.0/effB = {}'.format(idx, tpr[bineff], 1.0 / fpr[bineff]))

    else:
        print("AUC: {}".format(metrics.roc_auc_score(y, y_pred)))
        print('Acc: {}'.format(metrics.accuracy_score(y, y_pred > 0.5)))

        fpr, tpr, _ = metrics.roc_curve(y, y_pred)

        for threshold in thresholds:
            bineff = np.argmax(tpr > threshold)
            print('effS at {} 1.0/effB = {}'.format(tpr[bineff], 1.0 / fpr[bineff]))

        tpr = tpr[fpr > 1e-4]
        fpr = fpr[fpr > 1e-4]
        sic = np.ma.divide(tpr, np.sqrt(fpr)).filled(0)
        print("Max SIC: {}".format(np.max(sic)))


def get_data_info(flags):
    multi_label = True
    if flags.dataset == 'top':
        test = utils.TopDataLoader(os.path.join(flags.folder, 'TOP', 'test_ttbar.h5'),
                                   flags.batch, rank=hvd.rank(), size=hvd.size())
        threshold = [0.3, 0.5]
        folder_name = 'TOP'

    if flags.dataset == 'opt':
        test = utils.TopDataLoader(os.path.join(flags.folder, 'Opt', 'test_ttbar.h5'),
                                   flags.batch, rank=hvd.rank(), size=hvd.size())
        threshold = [0.3, 0.5]
        folder_name = 'Opt'

    elif flags.dataset == 'qg':
        test = utils.QGDataLoader(os.path.join(flags.folder, 'QG', 'test_qg.h5'),
                                  flags.batch, rank=hvd.rank(), size=hvd.size())
        threshold = [0.3, 0.5]
        folder_name = 'QG'

    elif flags.dataset == 'atlas':
        test = utils.AtlasDataLoader(os.path.join(flags.folder, 'ATLASTOP', 'test_atlas.h5'),
                                     flags.batch, rank=hvd.rank(), size=hvd.size())
        threshold = [0.5, 0.8]
        folder_name = 'ATLASTOP'
        multi_label = False
    elif flags.dataset == 'atlas_small':
        test = utils.AtlasDataLoader(os.path.join(flags.folder, 'ATLASTOP', 'test_atlas.h5'),
                                     flags.batch, rank=hvd.rank(), size=hvd.size())
        threshold = [0.5, 0.8]
        folder_name = 'ATLASTOP'
        multi_label = False
    elif flags.dataset == 'h1':
        test = utils.H1DataLoader(os.path.join(flags.folder, 'H1', 'test.h5'),
                                  flags.batch, rank=hvd.rank(), size=hvd.size())
        threshold = [0.5, 0.1]
        folder_name = 'H1'

    elif flags.dataset == 'cms':
        test = utils.CMSQGDataLoader(os.path.join(flags.folder, 'CMSQG', 'test_qgcms_pid.h5'),
                                     flags.batch, rank=hvd.rank(), size=hvd.size())
        threshold = [0.5, 0.8]
        folder_name = 'CMSQG'

    elif flags.dataset == 'jetclass':
        test = utils.JetClassDataLoader(os.path.join(flags.folder, 'JetClass', 'test',
                                                     rank=hvd.rank(), size=hvd.size()),
                                        flags.batch)
        threshold = [0.5]
        folder_name = 'JetClass/test'

    return test, multi_label, threshold, folder_name


def compute_snr_weight(t, logsnr_min=-20., logsnr_max=20.):
    """Compute -d(logsnr)/dt for the cosine schedule, used for ELBO weighting."""
    b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
    a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
    arg = a * tf.cast(t, tf.float32) + b
    weight = 2.0 * a / tf.math.sin(2.0 * arg)
    return tf.abs(weight)


def compute_generative_losses(model, X_dataset, num_classes, num_timesteps, weighting, seed):
    """Compute per-class diffusion v-prediction losses for all samples."""
    ema_body = model.ema_body
    ema_gen = model.ema_generator_head
    num_diffusion = model.num_diffusion

    tf.random.set_seed(seed)

    all_losses = []
    num_batches = 0
    for batch in X_dataset:
        features = batch['input_features']
        mask = batch['input_mask']
        jet = batch['input_jet']
        B = tf.shape(features)[0]

        # Expand mask to (B, P, 1) for consistency with model expectations
        if len(mask.shape) == 2:
            mask_3d = mask[:, :, None]
        else:
            mask_3d = mask

        class_losses = tf.zeros((B, num_classes))

        for step in range(num_timesteps):
            t = tf.random.uniform((B, 1))
            # Clamp t away from boundaries for SNR weighting stability
            if weighting == "snr":
                t = tf.clip_by_value(t, 1e-4, 1.0 - 1e-4)

            _, alpha, sigma = get_logsnr_alpha_sigma(t)

            # Sample noise, masked to valid particles
            eps = tf.random.normal(tf.shape(features), dtype=tf.float32) * mask_3d

            # Only diffuse first num_diffusion features
            mask_diffusion = tf.concat([
                tf.ones_like(eps[:, :, :num_diffusion], dtype=tf.bool),
                tf.zeros_like(eps[:, :, num_diffusion:], dtype=tf.bool)
            ], axis=-1)
            eps = tf.where(mask_diffusion, eps, tf.zeros_like(eps))

            # Forward process: x_t = alpha * x_0 + sigma * eps
            perturbed_x = alpha[:, None] * features + sigma[:, None] * eps
            perturbed_x = tf.where(mask_diffusion, perturbed_x, tf.zeros_like(perturbed_x))

            # Body forward pass (class-independent, done once per timestep)
            encoded = ema_body([perturbed_x, perturbed_x[:, :, :2], mask, t], training=False)

            # V-prediction target (only diffused features)
            v_target = alpha[:, None] * eps - sigma[:, None] * features
            v_target = v_target[:, :, :num_diffusion]

            # Timestep weight
            if weighting == "snr":
                w = compute_snr_weight(t)  # (B, 1)
            else:
                w = tf.ones_like(t)  # (B, 1)

            # Number of valid particles per sample
            n_part = tf.reduce_sum(mask_3d[:, :, 0], axis=-1)  # (B,)
            n_part = tf.maximum(n_part, 1.0)  # avoid division by zero

            # Compute loss for each class
            for c in range(num_classes):
                label_c = tf.one_hot(tf.fill([B], c), num_classes, dtype=tf.float32)
                v_pred = ema_gen([encoded, jet, mask, t, label_c], training=False)
                v_pred = v_pred[:, :, :num_diffusion]

                # Per-particle squared error, summed over diffusion features
                sq_diff = tf.reduce_sum(tf.square(v_target - v_pred), axis=-1)  # (B, P)
                # Mask invalid particles and average
                per_sample = tf.reduce_sum(sq_diff * mask_3d[:, :, 0], axis=-1)  # (B,)
                per_sample = per_sample / (num_diffusion * n_part)  # (B,)

                # Weighted accumulation
                weighted_loss = w[:, 0] * per_sample  # (B,)
                # Add to the c-th column of class_losses
                class_losses = class_losses + tf.expand_dims(weighted_loss, 1) * tf.one_hot(c, num_classes, dtype=tf.float32)

        class_losses = class_losses / tf.cast(num_timesteps, tf.float32)
        all_losses.append(class_losses.numpy())
        num_batches += 1

        if hvd.rank() == 0 and num_batches % 10 == 0:
            print(f"  Processed {num_batches} batches...", flush=True)

    return np.concatenate(all_losses, axis=0)


def evaluate_classifier_head(model, X_dataset):
    """Run the classifier head and return predictions."""
    # Create a version with softmax activation for evaluation
    pred = model.classifier.predict(X_dataset, verbose=hvd.rank() == 0)
    # pred is [logits, regression]; apply softmax to logits
    logits = pred[0]
    return tf.nn.softmax(logits).numpy()


def build_add_string(flags):
    """Build checkpoint add_string matching the naming convention in train.py."""
    add_string = ""
    if flags.nid > 0:
        add_string += "_{}".format(flags.nid)
    if flags.freeze_body:
        add_string += "_peft"
    if flags.freeze_heads:
        add_string += "_frozenheads"
    if flags.lora_rank > 0:
        add_string += "_lora{}".format(flags.lora_rank)
    return add_string


def main():
    utils.setup_gpus()
    flags = parse_arguments()

    test, multi_label, thresholds, folder_name = get_data_info(flags)

    add_string = build_add_string(flags)
    if flags.pretrained_only:
        npy_suffix = "_pretrained_only_generative.npy"
    else:
        npy_suffix = "_generative.npy"
    npy_file = os.path.join(flags.folder, folder_name, 'npy', '{}'.format(
        utils.get_model_name(
            flags, fine_tune=flags.fine_tune,
            add_string=add_string).replace('.weights.h5', npy_suffix)))

    if flags.load:
        if hvd.rank() == 0:
            print("Loading saved npy files")
        data = np.load(npy_file, allow_pickle=True).item()
        y = data['y']
        gen_scores = data['gen_scores']
        cls_pred = data.get('cls_pred', None)
    else:
        # Build model with target dataset dimensions (e.g. top: 2 classes)
        model = PET(num_feat=test.num_feat,
                    num_jet=test.num_jet,
                    num_classes=test.num_classes,
                    local=flags.local,
                    num_layers=flags.num_layers,
                    drop_probability=flags.drop_probability,
                    simple=flags.simple, layer_scale=flags.layer_scale,
                    talking_head=flags.talking_head,
                    mode=flags.mode,
                    freeze_body=flags.freeze_body,
                    freeze_heads=flags.freeze_heads,
                    lora_rank=flags.lora_rank)

        X, y = test.make_eval_data()

        # Mark model as built so load_weights works on the subclassed model.
        model.built = True

        if flags.pretrained_only:
            # Load JetClass pretrained checkpoint directly; skip mismatched layers
            # (classifier output + generator label embedding won't match due to num_classes)
            pretrained_name = utils.get_model_name(
                flags, fine_tune=flags.fine_tune).replace(
                    flags.dataset, 'jetclass').replace(
                        'fine_tune', 'baseline').replace(flags.mode, 'all')
            pretrained_path = os.path.join(flags.folder, 'checkpoints', pretrained_name)
            if hvd.rank() == 0:
                print(f"Loading pretrained JetClass checkpoint: {pretrained_name}")
                print("  (head layers with mismatched shapes will be randomly initialized)")
            model.load_weights(pretrained_path, by_name=True, skip_mismatch=True)
        else:
            checkpoint_name = utils.get_model_name(
                flags, fine_tune=flags.fine_tune, add_string=add_string)
            if hvd.rank() == 0:
                print(f"Loading checkpoint: {checkpoint_name}")
            model.load_weights(os.path.join(flags.folder, 'checkpoints', checkpoint_name))

        # --- Always run BOTH evaluation methods ---
        # Even if only one head was trained, we want to compare both methods
        # (the untrained head gives us a baseline for comparison).
        if hvd.rank() == 0:
            print("\nRunning classifier head evaluation...")
        cls_pred = evaluate_classifier_head(model, X)
        cls_pred = hvd.allgather(tf.constant(cls_pred)).numpy()

        if hvd.rank() == 0:
            print(f"\nRunning generative classification (T={flags.num_timesteps}, weighting={flags.weighting})...")
        gen_losses = compute_generative_losses(
            model, X, test.num_classes, flags.num_timesteps, flags.weighting, flags.seed)
        gen_losses = hvd.allgather(tf.constant(gen_losses)).numpy()
        gen_scores = softmax(-gen_losses, axis=-1)

        y = hvd.allgather(tf.constant(y)).numpy()

        # Save results
        if hvd.rank() == 0:
            save_dir = os.path.join(flags.folder, folder_name, 'npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_data = {'y': y}
            if gen_scores is not None:
                save_data['gen_scores'] = gen_scores
                save_data['gen_losses'] = gen_losses
            if cls_pred is not None:
                save_data['cls_pred'] = cls_pred
            np.save(npy_file, save_data)

    # --- Print results ---
    if hvd.rank() == 0:
        if cls_pred is not None:
            print("\n" + "=" * 50)
            print("=== Classifier Head ===")
            print("=" * 50)
            if multi_label:
                print_metrics(cls_pred, y, thresholds, multi_label=True)
            else:
                print_metrics(cls_pred[:, 1], y[:, 1], thresholds, multi_label=False)

        if gen_scores is not None:
            print("\n" + "=" * 50)
            print(f"=== Generative Classification (T={flags.num_timesteps}, {flags.weighting}) ===")
            print("=" * 50)
            if multi_label:
                print_metrics(gen_scores, y, thresholds, multi_label=True)
            else:
                print_metrics(gen_scores[:, 1], y[:, 1], thresholds, multi_label=False)


if __name__ == '__main__':
    main()
