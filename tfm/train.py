import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

import lcpfn.encoders as encoders
from lcpfn.train_lcpfn import train_lcpfn

LCBENCH_DATA_PATH = (
    "/Users/marcelhofmann/UAM_Deep_Learning/TFM_Implementation/LCBench/data/exported"
)
NATS_DATA_PATH = (
    "/Users/marcelhofmann/UAM_Deep_Learning/TFM_Implementation/NATS-Bench/exported"
)
SCALING_LAW_PATHS = [
    "/Users/marcelhofmann/UAM_Deep_Learning/TFM_Implementation/scaling_law_derivation/scaling_laws/lcbench",
    "/Users/marcelhofmann/UAM_Deep_Learning/TFM_Implementation/scaling_law_derivation/scaling_laws/nats",
]

DEFAULT_STYLE_KEYS = [
    "architecture.param_count",
    "architecture.flops",
    "dataset_metadata.num_input_features",
    "dataset_metadata.num_classes",
    "dataset_metadata.num_samples",
    "dataset_metadata.openml_task_id",
    "dataset_metadata.image_size",
    "dataset_metadata.num_input_channels",
    "dataset_metadata.train_samples",
    "dataset_metadata.test_samples",
    "epochs",
    "config_id",
    "seed",
    "hp",
    "eval_split",
]

DEFAULT_SCALING_LAW_KEYS = [
    "alpha",
    "y_inf",
    "A",
    "r2",
    "param_min",
    "param_max",
    "max_y",
    "mean_y",
    "frontier_slope_small",
    "frontier_slope_large",
    "frontier_curvature",
    "n_models",
    "alpha_boot_mean",
    "alpha_boot_std",
]


def setup_logger(log_level, log_file) -> logging.Logger:
    """
    Configure and return the project logger.

    Parameters:
        log_level (`str`): Logging level ("DEBUG", "INFO", "WARNING", "ERROR").

    Returns:
        logging.Logger: The logger instance.
    """
    logger = logging.getLogger("lcpfn")

    # Prevent duplicate handlers if setup_logger() is called multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(getattr(logging, log_level.upper()))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    # console
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # logging file
    # if log_file is not None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def extract_nested(d: dict, keys: tuple[str, ...]) -> Any:
    """
    Based on the key structure in the key tuple, extract the value from the dict.

    Parameters:
        d (`dict`): The dict the data will be retrieved from.
        keys (`tuple`): Tuple containing the key hierarchy.

    Returns:
        Any: Extracted value, should be a number to continue without problems.

    """
    for k in keys:
        d = d[k]

    if isinstance(d, dict):
        raise TypeError(
            f"tuple indicating json path '{keys}' seems incomplete, got a dict returned."
        )
    return d


def load_scaling_law(path: str | Path, feature_keys: list[str]) -> np.array:
    """
    Given a path to a derived scaling law, use it to retrieve the data from an `.npz` file.

    Parameters:
        path (`str | Path`): The path in the filetree to the scaling law
        feature_keys (`list[str]`): The features that should be retrieved from the scaling law.

    Returns:
        np.array: A numpy array containing scaling law information.
    """

    data = np.load(path, allow_pickle=True)

    features = {
        key: np.asarray(data[key]).item() for key in feature_keys if key in data.files
    }

    frontier_x = data["frontier_x"].astype(np.float32)
    frontier_y = data["frontier_y"].astype(np.float32)

    # scalar feature vector
    feature_vector = np.array(
        [features.get(key, 0.0) for key in feature_keys],
        dtype=np.float32,
    )

    # concatenate everything properly
    full_vector = np.concatenate(
        [
            feature_vector,
            frontier_x,
            frontier_y,
        ]
    ).astype(np.float32)

    return full_vector


def dataset_name_from_scaling_law(path: Path) -> str:
    """
    Having the a given path to a scaling law, use it to extract the name of the dataset the scaling law belongs to.
    Benefits from the fact that names are constructed as follows: `sl__dataset={dataset_name}__in={in_param}__out={out_param}``

    Parameters:
        path (`Path`): Path to a scaling law containing the scaling law file name.

    Returns:
        str: The dataset name the scaling law belongs to.
    """
    name = path.name
    prefix = "sl__dataset="
    suffix = "__"
    if name.startswith(prefix) and suffix in name[len(prefix) :]:
        return name[len(prefix) :].split(suffix, 1)[0]
    return path.stem


def build_scaling_law_index(paths: Sequence[str], feature_keys: list[str]):
    """
    Builds a dict containing the scaling law metadata for each dataset a scaling law has been created.
    Dict keys represent dataset names, values represent metadata vector.

    Parameters:
        paths (`Sequence[str]`): Paths for scaling laws.
        feature_keys (`list[str]`): The features that should be retrieved from the scaling law.

    Returns:
        tuple[dict, int]: The dict with scaling law metadata as vector and the amount of dimensions each scaling law representation has.
    """
    index = {}
    vector_dim = 0

    for base_path in paths:
        for path in sorted(Path(base_path).glob("*.npz")):
            # loading of scaling law vector
            vector = load_scaling_law(path, feature_keys)
            # dataset: vector
            index[dataset_name_from_scaling_law(path)] = vector
            # dimension of scaling law data vector
            vector_dim = max(vector_dim, len(vector))

    if vector_dim:
        # assure always the same length of vector with scaling law metadata for all laws
        for dataset, vector in list(index.items()):
            if len(vector) < vector_dim:
                index[dataset] = np.pad(vector, (0, vector_dim - len(vector)))
            elif len(vector) > vector_dim:
                index[dataset] = vector[:vector_dim]

    return index, vector_dim


def extract_numeric(value: Any, default: float = 0.0) -> float:
    """
    Assures that a value becomes a numeric value. If not convertible, gets value `default`.

    Parameters:
        value (`Any`): A value that should be made numeric.
        default (`float`): If `value` can not be made numeric, it gets this value.

    Returns:
        float: Numerized value.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def extract_style_vector(
    payload: Any,
    style_keys: Optional[list[str]],
    scaling_law_index: dict[str, np.ndarray],
    scaling_law_dim: int,
):
    """
    Function for extracting the style data (additional metadata such as scaling law info).

    Parameters:
        payload (`Any`): The loaded json.
        style_keys (`list[str] | None`): The keys in form `<parent>.<child>` saved to a list.
        scaling_law_index (`dict[str, np.ndarray]`): The dict containing scaling law info in format `dataset:scaling_law_vector`, is concatenated to additional metadata
        scaling_law_dim (`int`): The dimension of the scaling law vectors, used for filling up non existing scaling law samples.

    Returns:
        np.NDArray | None: Nunmpy array containing the metadata. Can be `None` if no metadata selected.
    """
    # if no keys and no scaling law data given, return none
    if not style_keys and not scaling_law_dim:
        return None

    style_values = []
    # for each key
    for key in style_keys or []:
        # try to retrieve it...
        try:
            value = extract_nested(payload, tuple(key.split(".")))
        except KeyError:
            value = None
        style_values.append(extract_numeric(value))

    style_values = np.asarray(style_values, dtype=np.float32)

    # use the dataset name to retrieve the scaling law data belonging to it
    dataset = payload.get("dataset")
    scaling_law_features = scaling_law_index.get(dataset)
    if scaling_law_features is None:
        scaling_law_features = np.zeros(scaling_law_dim, dtype=np.float32)

    # concat metadata from json with scaling law describing data
    style_vector = np.concatenate(
        [style_values, scaling_law_features.astype(np.float32)]
    ).astype(np.float32)

    return style_vector


def load_learning_curves(
    data_paths: list[str],
    split: str = "eval",
    metric: str = "accuracy",
    normalize: bool = True,
    enforce_monotonic: bool = True,
    clip: bool = True,
    style_keys: Optional[list[str]] = None,
    scaling_law_paths: Optional[list[str]] = None,
    scaling_law_keys: Optional[list[str]] = None,
):
    """
    Function to load the learning curves and their additional metadata.
    This can be training information itself that is in the json training metadata or scaling law information.

    Parameters:
        data_paths (`list[str]`): The paths leading to the learning curve jsons, usually leading to LCBench and NATS Bench samples.
        split (`str`): Learning curves for what optimization phase should be used? Either `train` or `eval`.
        metric (`str`): Metric the learning curve represents, usually `accuracy` or `loss`.
        normalize (`bool`): Should values be normalized in this function?
        enforce_monotonic (`bool`): If set `True`, values never switch direction, they go monotonically up or down.
        clip (`bool`): Should value be in a range between 0 and 1?
        style_keys (`list[str] | None`): Keys for retrieving additional metadata from the learning curve json in form `<parent>.<child>` saved to a list.
        scaling_law_paths (`list[str] | None`): The paths leading to locations where scaling law metadata is saved.

    Returns:
        tuple: Returns a list of learning curves and an `np.array` containing additional metadata and scaling law infos if wanted by the user.
    """
    curves = []
    styles = []
    all_paths = []

    # load all samples in json format from given paths
    for data_path in data_paths:
        all_paths.extend(sorted(glob.glob(str(Path(data_path) / "*.json"))))

    if not all_paths:
        raise FileNotFoundError(f"No JSON files found in {data_paths}")

    # retrieval of scaling laws based on paths and value keys selected
    scaling_law_index, scaling_law_dim = build_scaling_law_index(
        scaling_law_paths or [], scaling_law_keys
    )

    # load all scaling law samples
    for path in all_paths:
        with open(path, "r") as handle:
            payload = json.load(handle)

        curve = np.asarray(payload["curves"][split][metric], dtype=np.float32)

        if normalize and metric == "accuracy":
            curve = curve / 100.0

        if enforce_monotonic:
            curve = np.maximum.accumulate(curve)

        if clip:
            curve = np.clip(curve, 0.0, 1.0)

        if np.isnan(curve).any():
            continue

        # extraction of the style vector belonging to the learning curve sample
        style_vector = extract_style_vector(
            payload,
            style_keys,
            scaling_law_index=scaling_law_index,
            scaling_law_dim=scaling_law_dim,
        )

        curves.append(curve)
        styles.append(style_vector)

    if not curves:
        raise RuntimeError("No valid learning curves were loaded from disk.")

    # if there are style value set None, dont return any style values
    if any(style is None for style in styles):
        return curves, None

    return curves, np.stack(styles).astype(np.float32)


def split_curves(
    curves: list[np.ndarray],
    styles: Optional[np.ndarray],
    val_fraction: float = 0.10,  # 0.15
    seed: int = 0,
):
    """
    Splits by curve (not by window), so no window of a validation curve
    ever appears in a training batch.

    Parameters:
        curves (`list[np.ndarray]`): All curves retrieved.
        styles (`Optional[np.ndarray]`): All styles retrieved. Not mandatory since model can be trained without metadata vectors.
        val_fraction (`float`): The percentage of samples that should make the validation set.
        seed (`int`): Seed for train-validation dataset reproducibility.

    Returns:
        tuple: The train and validation curves and their styles if possible. Format: `(train_curves, val_curves, train_styles, val_styles)`
    """
    n = len(curves)
    rng = np.random.RandomState(seed)
    permuted = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    val_idx = set(permuted[:n_val].tolist())

    train_curves, val_curves = [], []
    train_styles, val_styles = [], []
    for i, curve in enumerate(curves):
        if i in val_idx:
            val_curves.append(curve)
            if styles is not None:
                val_styles.append(styles[i])
        else:
            train_curves.append(curve)
            if styles is not None:
                train_styles.append(styles[i])

    train_styles_arr = (
        np.stack(train_styles).astype(np.float32) if styles is not None else None
    )
    val_styles_arr = (
        np.stack(val_styles).astype(np.float32) if styles is not None else None
    )
    return train_curves, val_curves, train_styles_arr, val_styles_arr


def make_get_batch_func(curves: list, styles: Optional[np.array] = None):
    """
    Function for batch creation from curves and style information.

    Parameters:
        curves (`list`): A list containing all learning curves.
        styles (`np.array | None`): Additional metadata belonging to the learning curves.

    Returns:
        tuple: All data necessary to perform a LC-PFN training iteration, format: `(x, y, y, style)`
    """
    curve_lengths = np.asarray([len(curve) for curve in curves], dtype=np.int32)

    def get_batch_from_disk(
        batch_size: int,
        seq_len: int,
        num_features: int,
        device: str = "cpu",
        # single_eval_pos=None,
        **_,
    ):
        """
        The actual function that is used for batch building.

        Parameters:
            batch_size (`int`): Amount of samples in one minibatch.
            seq_len (`int`): The length of the sequence fed into the LC-PFN.
            num_features (`int`): Feature count.
            device (`str`): On which device should the batch live?

        Returns:
            tuple: All data necessary to perform a LC-PFN training iteration, format: `(x, y, y, style)`
        """
        assert num_features == 1

        # ensure all curves can support a full window
        eligible_indices = np.where(curve_lengths >= seq_len)[0]

        if len(eligible_indices) == 0:
            raise ValueError("No curves long enough for seq_len")

        # select sample indices for a batch and create an empty array to safe selected samples
        indices = np.random.choice(eligible_indices, size=batch_size, replace=True)
        batch_curves = np.empty((batch_size, seq_len), dtype=np.float32)

        # extract curves index by index
        for i, curve_idx in enumerate(indices):
            curve = curves[curve_idx]

            # sliding window approach: select a random start and add "seq_len" next positions
            max_start = len(curve) - seq_len
            start = np.random.randint(0, max_start + 1)

            window = curve[start : start + seq_len]
            batch_curves[i] = window

        assert not np.isnan(batch_curves).any()

        # LCPFN expects sequence-major inputs: [T, B, F].
        x = (
            torch.arange(seq_len)
            .float()
            .unsqueeze(1)
            .unsqueeze(-1)
            .repeat(1, batch_size, 1)
            .to(device)
        )

        y = torch.from_numpy(batch_curves.T.copy()).float().to(device)

        style = None
        if styles is not None:
            style = torch.from_numpy(styles[indices].copy()).float().to(device)

        # IMPORTANT: no global alignment needed
        return x, y, y, style

    return get_batch_from_disk


def make_eval_batch(
    curves: list[np.ndarray],
    styles: Optional[np.ndarray],
    seq_len: int,
    seed: int = 0,
):
    """
    Builds one fixed (reproducible) batch covering every validation curve long enough to support `seq_len`.
    Unlike training, the window per curve is sampled once with a fixed seed so evaluation numbers are stable across runs/epochs.

    Parameters:
        curves (`list[np.ndarray]`): All curves that should be used for validation
        styles (`Optional[np.ndarray]`): The vector containing the style metadata. Is optional since not mandatory for training.
        seq_len (`int`): Sequence length of the input.
        seed (`int`): Random seed for reproducible validation dataset.

    Returns:
        tuple: Contains the eval batch curves and the style data.

    """
    rng = np.random.RandomState(seed)
    # get all curves that are bigger than sequence length to do effective sampling
    eligible = [i for i, c in enumerate(curves) if len(c) >= seq_len]
    if not eligible:
        return None, None

    # array that will contain the eval curves
    batch_curves = np.empty((len(eligible), seq_len), dtype=np.float32)
    # iterate through all possible curves and cut out a sequence of length seq_len
    for row, idx in enumerate(eligible):
        curve = curves[idx]
        max_start = len(curve) - seq_len
        start = rng.randint(0, max_start + 1)
        batch_curves[row] = curve[start : start + seq_len]

    style_batch = styles[eligible] if styles is not None else None
    return batch_curves, style_batch


@torch.no_grad()
def evaluate_model(
    model,
    val_curves: list[np.ndarray],
    val_styles: Optional[np.ndarray],
    seq_len: int,
    eval_positions: Sequence[int],
    device: str = "cpu",
    denormalize_to_percent: bool = True,
    seed: int = 0,
):
    """
    For each position in `eval_positions`, feeds the model the curve up to that position and
    compares its predicted mean (via the trained BarDistribution's `.mean()`) against the true continuation.

    Parameters:
        model: The LC-PFN to be evaluated.
        val_curves (`list[np.ndarray]`): The validation set of curves.
        val_styles (`Optional[np.ndarray]`): The validation set of style vectors. Optional since not mandatory for training setup.
        seq_len (`int`): The input sequence length.
        eval_positions (`Sequence[int]`): The cutoff points between the input and target values, gets multiple points to check how model behaves for different curve lengths as input.
        device (`str`): Where should the eval happen?
        denormalize_to_percent (`bool`): Should Mean Absolute Errors be normalized to percent again? For example, MAE would be `2.29` instead of `0.0229`.
        seed (`int`): Random seed for result reproduction, used for batch creation.

    Returns:
        dict[int, dict]: per-position {"nll": ..., "mae": ..., "n_curves": ...}
    """
    # batch creation
    batch_curves, style_batch = make_eval_batch(val_curves, val_styles, seq_len, seed)
    if batch_curves is None:
        return {}

    model.eval()
    # putting inputs into correct format
    batch_size = batch_curves.shape[0]
    x = (
        torch.arange(seq_len)
        .float()
        .unsqueeze(1)
        .unsqueeze(-1)
        .repeat(1, batch_size, 1)
        .to(device)
    )
    y = torch.from_numpy(batch_curves.T.copy()).float().to(device)
    style = (
        torch.from_numpy(style_batch.copy()).float().to(device)
        if style_batch is not None
        else None
    )

    results = {}
    # for each eval position, run the inference
    for pos in eval_positions:
        pos = int(pos)
        if not (0 < pos < seq_len):
            continue
        logits = model((style, x, y), single_eval_pos=pos)
        target = y[pos:]

        # metric calculation
        nll = model.criterion(logits, target).mean().item()
        pred_mean = model.criterion.mean(logits)
        abs_err = (pred_mean - target).abs()
        mae = abs_err.mean().item()
        if denormalize_to_percent:
            mae *= 100.0  # back to accuracy points, assuming /100 normalization

        results[pos] = {"nll": nll, "mae": mae, "n_curves": batch_size}

    model.train()
    return results


def make_epoch_callback(
    val_curves: list[np.ndarray],
    val_styles: Optional[np.ndarray],
    seq_len: int,
    eval_positions: Sequence[int],
    total_epochs: int,
    eval_every: int,
    logger: logging.Logger,
    run_dir: str,
    denormalize_to_percent: bool = True,
    seed: int = 0,
):
    """
    Function used as callback for performing a validation epoch after every nth training epoch.

    Parameters:
        val_curves (`list[np.ndarray]`): The validation set of curves.
        val_styles (`Optional[np.ndarray]`): The validation set of style vectors. Optional since not mandatory for training setup.
        seq_len (`int`): The input sequence length.
        eval_positions (`Sequence[int]`): The cutoff points between the input and target values, gets multiple points to check how model behaves for different curve lengths as input.
        total_epochs (`int`): Entire amount of training epochs for the execution. Used for calculation if validation should happen or not.
        eval_every (`int`): After how many epochs should a validation happen?
        logger (`logging.Logger`): Logger for logging information.
        run_dir (`str`): The directory for storing run information. Used for storing validation metrics.
        denormalize_to_percent (`bool`): Should Mean Absolute Errors be normalized to percent again? For example, MAE would be `2.29` instead of `0.0229`.
        seed (`int`): Seed for reproducibility

    Returns:
        None: Just logs eval information.
    """
    history_path = Path(run_dir) / "val_metrics_history.jsonl"

    def epoch_callback(model, progress):
        epoch = round(progress * total_epochs)
        if epoch == 0 or epoch % eval_every != 0:
            return

        metrics = evaluate_model(
            model,
            val_curves,
            val_styles,
            seq_len=seq_len,
            eval_positions=eval_positions,
            denormalize_to_percent=denormalize_to_percent,
            seed=seed,
        )
        for pos, m in metrics.items():
            logger.info(
                f"[val][epoch {epoch:3d}] eval_pos={pos:3d}/{seq_len} | "
                f"NLL={m['nll']:.4f} | MAE={m['mae']:.4f} pts | n_curves={m['n_curves']}"
            )

        with open(history_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, "metrics": metrics}) + "\n")

    return epoch_callback


def parse_args():
    """Function that retrieves user arguments."""
    parser = argparse.ArgumentParser(
        description="Train LC-PFN from learning curves stored in DATA_PATH."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--run_name", default="train", help="The name of the run, used for logging."
    )
    parser.add_argument(
        "--data-path",
        action="append",
        default=None,
        help="Directory with exported JSON files. Can be passed multiple times.",
    )
    parser.add_argument(
        "--no-style", action="store_true", help="Train without metadata style features."
    )
    parser.add_argument(
        "--no-scaling-laws",
        action="store_true",
        help="Train without scaling law features.",
    )
    parser.add_argument(
        "--scaling-law-path",
        action="append",
        default=None,
        help="Directory with scaling-law .npz files. Can be passed multiple times.",
    )
    parser.add_argument(
        "--style-key",
        action="append",
        default=None,
        help="Numeric JSON metadata key in dotted form. Can be passed multiple times.",
    )
    parser.add_argument(
        "--scaling-law-key",
        action="append",
        default=None,
        help="Feature from the scaling-law .npz. Can be repeated.",
    )
    parser.add_argument("--split", default="eval", choices=["train", "eval"])
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--emsize", type=int, default=256)
    parser.add_argument("--nlayers", type=int, default=3)
    parser.add_argument("--num-borders", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=8)
    # parser.add_argument("--output-path", default="trained_lcpfn_from_data.pt")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Training horizon. Defaults to the shortest loaded curve so every data source is eligible.",
    )
    parser.add_argument(
        "--disable-monotonic-fix",
        action="store_true",
        help="Keep the original curve values instead of applying cumulative max.",
    )
    parser.add_argument(
        "--disable-normalization",
        action="store_true",
        help="Do not scale accuracy curves from [0, 100] to [0, 1].",
    )
    parser.add_argument(
        "--eval-every-n-epochs",
        type=int,
        default=2,
        help="Run held-out MAE/NLL evaluation every N epochs during training. Set to 0 to disable.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,  # 0.15,
        help="Fraction of curves (held out whole, not by window) used for validation.",
    )
    parser.add_argument(
        "--val-seed",
        type=int,
        default=0,
        help="Seed for the train/val curve split and the fixed validation windows.",
    )
    parser.add_argument(
        "--eval-fraction",
        action="append",
        type=float,
        default=None,
        help="Fraction(s) of seq_len at which to compute held-out MAE/NLL. "
        "Can be repeated. Default: 0.25 0.5 0.75",
    )
    return parser.parse_args()


def main():
    # setup
    args = parse_args()
    run_dir = f"results/{int(time.time())}_{args.run_name}"
    logger = setup_logger(
        log_level=args.log_level,
        log_file=f"{run_dir}/log.log",
    )
    logger.info(f"Args: {args}")

    # get the paths and variables to be used, either from user or use standard ones
    data_paths = args.data_path or [LCBENCH_DATA_PATH, NATS_DATA_PATH]
    scaling_law_paths = args.scaling_law_path or [
        path for path in SCALING_LAW_PATHS if Path(path).exists()
    ]
    # define if style keys should be used
    if args.no_style:
        style_keys = []
    else:
        style_keys = args.style_key or DEFAULT_STYLE_KEYS
    # define if scaling law data should be used
    if args.no_scaling_laws:
        scaling_law_keys = []
    else:
        scaling_law_keys = args.scaling_law_key or DEFAULT_SCALING_LAW_KEYS

    # curve loading
    curves, styles = load_learning_curves(
        data_paths=data_paths,
        split=args.split,
        metric=args.metric,
        normalize=not args.disable_normalization,
        enforce_monotonic=not args.disable_monotonic_fix,
        style_keys=style_keys,
        scaling_law_paths=scaling_law_paths,
        scaling_law_keys=scaling_law_keys,
    )

    # doing train-eval split
    train_curves, val_curves, train_styles, val_styles = split_curves(
        curves, styles, val_fraction=args.val_fraction, seed=args.val_seed
    )
    logger.info(
        f"Split {len(curves)} curves into {len(train_curves)} train / "
        f"{len(val_curves)} val (val_fraction={args.val_fraction}, seed={args.val_seed})"
    )

    # defining helpful variables
    max_curve_len = max(len(curve) for curve in curves)
    min_curve_len = min(len(curve) for curve in curves)
    seq_len = args.seq_len or min_curve_len
    # create the batch creation function, used in train_lcpfn() function
    get_batch_func = make_get_batch_func(curves, styles)

    logger.info(f"Loaded {len(curves)} curves from {data_paths}")
    if train_styles is not None:
        logger.info(f"Using style vectors with dimension {train_styles.shape[1]}")
    logger.info(
        f"Training on split={args.split}, metric={args.metric}, seq_len={seq_len}, curve_len_range=({min_curve_len}, {max_curve_len})"
    )

    # preparation of the evaluation sequence length fractions
    eval_fractions = args.eval_fraction or [0.25, 0.5, 0.75]
    eval_positions = sorted(
        {min(seq_len - 1, max(1, int(seq_len * f))) for f in eval_fractions}
    )
    # create a callback only if a validation every n epochs is selected by user
    epoch_callback = None
    if args.eval_every_n_epochs > 0:
        epoch_callback = make_epoch_callback(
            val_curves,
            val_styles,
            seq_len=seq_len,
            eval_positions=eval_positions,
            total_epochs=args.epochs,
            eval_every=args.eval_every_n_epochs,
            logger=logger,
            run_dir=run_dir,
            denormalize_to_percent=not args.disable_normalization,
            seed=args.val_seed,
        )

    # do the actual training
    training_result = train_lcpfn(
        get_batch_func=get_batch_func,
        seq_len=seq_len,
        emsize=args.emsize,
        nlayers=args.nlayers,
        num_borders=args.num_borders,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        style_encoder_generator=encoders.StyleEncoder if styles is not None else None,
        logger=logger,
        epoch_callback=epoch_callback,
    )
    # save the model
    model = (
        training_result[2] if isinstance(training_result, tuple) else training_result
    )
    torch.save(model.state_dict(), f"{run_dir}/model.pt")
    logger.info(f"Saved model weights to {run_dir}")


if __name__ == "__main__":
    main()
