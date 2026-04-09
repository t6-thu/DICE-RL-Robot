# Converts raw collected episodes (images + JSON) into a zarr DirectoryStore.
#
# Usage:
#   1. Set DICE_RAW_DATASET_FOLDERS and DICE_DATASET_FOLDERS environment variables
#   2. Edit input_dir and output_dir below to point to your data
#   3. Run: python scripts/process_raw_data.py
#
# After this script finishes, run scripts/postprocess_add_virtual_target.py to compute virtual target labels if you want to predict force

import os
import pathlib
import shutil
import concurrent.futures

import zarr

from utils.data_processing.processing_functions import process_one_episode_into_zarr, generate_meta_for_zarr
from utils.imagecodecs_numcodecs import register_codecs

CORRECTION = False   # set to True if you want to use correction data

# check environment variables
if "DICE_RAW_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable DICE_RAW_DATASET_FOLDERS")
if "DICE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable DICE_DATASET_FOLDERS")


# specify the input and output directories
id_list = [0]  # [0] for single robot, [0, 1] for bimanual

ft_sensor_configuration = "handle_on_robot"  # "handle_on_sensor" or "handle_on_robot"

input_dir = pathlib.Path(
    os.environ.get("DICE_RAW_DATASET_FOLDERS") + "/your_task"  # TODO: SET raw data folder name
)
output_dir = pathlib.Path(
    os.environ.get("DICE_DATASET_FOLDERS") + "/your_task_processed"  # TODO: SET output folder name
)

# clean and create output folders
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# open the zarr store
store = zarr.DirectoryStore(path=output_dir)
root = zarr.open(store=store, mode="a")

print("Reading data from input_dir: ", input_dir)
episode_names = sorted(os.listdir(input_dir))

episode_config = {
    "input_dir": input_dir,
    "output_dir": output_dir,
    "id_list": id_list,
    "ft_sensor_configuration": ft_sensor_configuration,
    "num_threads": 10,
    "has_correction": CORRECTION,
    "save_video": False,
    "max_workers": 32
}

with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
    # map each Future to its episode_name
    future_to_ep = {
        executor.submit(
            process_one_episode_into_zarr,
            episode_name,
            root,
            episode_config,
        ): episode_name
        for episode_name in episode_names
    }

    failed_eps = []

    for future in concurrent.futures.as_completed(future_to_ep):
        ep = future_to_ep[future]
        try:
            result = future.result()
            if not result:
                print(f"[WARNING] Episode {ep} returned False and will be skipped.")
                failed_eps.append(ep)
        except Exception as e:
            print(f"[EXCEPTION] Episode {ep} raised an exception and will be skipped:\n{e}")
            failed_eps.append(ep)

    if failed_eps:
        print(f"\n[SUMMARY] Skipped {len(failed_eps)} episode(s) due to errors:")
        for ep in failed_eps:
            print(f"  - {ep}")


print("Finished reading. Now start generating metadata")
from utils.imagecodecs_numcodecs import register_codecs

register_codecs()

count = generate_meta_for_zarr(root, episode_config)
print(f"All done! Generated {count} episodes in {output_dir}")
print("Next step: run scripts/postprocess_add_virtual_target.py")
