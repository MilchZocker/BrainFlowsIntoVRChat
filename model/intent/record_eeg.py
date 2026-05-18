import argparse
import time
import pickle
import os

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

from sound_helper import SoundHelper

SAVE_FILENAME = 'recorded_eeg'
SAVE_EXTENSION = '.pkl'


def create_filename(number):
    filename = SAVE_FILENAME
    if number != 0:
        filename += str(number)
    return filename + SAVE_EXTENSION


def parse_action_indices(action_indices_arg):
    if action_indices_arg is None or action_indices_arg.strip() == "":
        raise ValueError(
            "--action-indices is required for sparse recording mode. "
            "Example: --action-indices 0 or --action-indices 0,2,5"
        )

    indices = []
    for part in action_indices_arg.split(','):
        part = part.strip()
        if not part:
            continue

        idx = int(part)
        if idx < 0:
            raise ValueError(f"Action index {idx} is invalid. Action indices must be >= 0.")

        if idx not in indices:
            indices.append(idx)

    if not indices:
        raise ValueError("No valid action indices were provided.")

    return sorted(indices)


def get_recording_filenames():
    return sorted(
        filename for filename in os.listdir()
        if filename.startswith(SAVE_FILENAME) and filename.endswith(SAVE_EXTENSION)
    )


def overwrite_selected_actions_in_existing_files(selected_actions):
    recording_files = get_recording_filenames()

    if not recording_files:
        print("No existing recording files found. Selected action overwrite has nothing to modify.")
        return

    for filename in recording_files:
        with open(filename, 'rb') as f:
            record_data = pickle.load(f)

        action_dict = record_data.get("action_dict", {})
        changed = False

        for action_idx in selected_actions:
            if action_idx in action_dict and len(action_dict[action_idx]) > 0:
                action_dict[action_idx] = []
                changed = True

        record_data["action_dict"] = action_dict

        if changed:
            with open(filename, 'wb') as f:
                pickle.dump(record_data, f)

    print(f"Cleared existing recordings for selected actions: {selected_actions}")


def get_next_available_filename():
    current_number = 0
    while True:
        current_filename = create_filename(current_number)
        if not os.path.isfile(current_filename):
            return current_filename
        current_number += 1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--timeout', type=int, required=False, default=0,
                        help='timeout for device discovery or connection')
    parser.add_argument('--ip-port', type=int, required=False, default=0,
                        help='ip port')
    parser.add_argument('--ip-protocol', type=int, required=False, default=0,
                        help='ip protocol, check IpProtocolType enum')
    parser.add_argument('--ip-address', type=str, required=False, default='',
                        help='ip address')
    parser.add_argument('--serial-port', type=str, required=False, default='',
                        help='serial port')
    parser.add_argument('--mac-address', type=str, required=False, default='',
                        help='mac address')
    parser.add_argument('--other-info', type=str, required=False, default='',
                        help='other info')
    parser.add_argument('--serial-number', type=str, required=False, default='',
                        help='serial number')
    parser.add_argument('--file', type=str, required=False, default='',
                        help='file')
    parser.add_argument('--actions', type=int, required=False, default=None,
                        help='legacy compatibility only; no longer required for sparse recording')
    parser.add_argument('--sessions', type=int, required=False, default=2,
                        help='number of sessions per action to record')
    parser.add_argument('--window-length', type=int, required=False, default=10,
                        help='length in seconds of eeg data pulled per session')
    parser.add_argument('--window-buffer', type=int, required=False, default=2,
                        help='time in seconds before eeg data is recorded each session (delay after hitting enter)')
    parser.add_argument('--overwrite', type=int, required=False, default=0,
                        help='1 to overwrite/remove old recordings, 0 to add results as an additional data file')
    parser.add_argument('--overwrite-selected', type=int, required=False, default=0,
                        help='1 to clear only selected action recordings in existing files before recording')
    parser.add_argument('--action-indices', type=str, required=True, default=None,
                        help='comma-separated action indices to record, e.g. 3 or 0,3,5')
    parser.add_argument('--board-id', type=str, required=True,
                        help='board id or name, check docs to get a list of supported boards')
    parser.add_argument('--start-delay', type=int, required=False, default=3,
                        help='delay between pressing enter and recording start')
    parser.add_argument('--enable-sounds', type=bool, required=False, default=True,
                        help='enables sound indicators for starting / stopping a recording')

    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file

    session_count = args.sessions
    window_length = args.window_length
    window_buffer = args.window_buffer
    recording_delay = args.start_delay
    sounds_enabled = args.enable_sounds

    sound_helper = SoundHelper(sounds_enabled)

    do_overwrite_all = args.overwrite == 1
    do_overwrite_selected = args.overwrite_selected == 1

    if do_overwrite_all and do_overwrite_selected:
        raise ValueError("Use either --overwrite 1 or --overwrite-selected 1, not both.")

    selected_actions = parse_action_indices(args.action_indices)

    try:
        master_board_id = int(args.board_id)
    except ValueError:
        master_board_id = BoardIds[args.board_id.upper()]

    board = BoardShim(master_board_id, params)

    sampling_rate = BoardShim.get_sampling_rate(master_board_id)
    sampling_size = sampling_rate * window_length

    action_dict = {action_idx: [] for action_idx in selected_actions}
    record_data = {
        "board_id": master_board_id,
        "window_seconds": window_length,
        "action_dict": action_dict
    }

    if do_overwrite_all:
        for filename in get_recording_filenames():
            os.remove(filename)
        print("Deleted all previous recording files.")
    elif do_overwrite_selected:
        overwrite_selected_actions_in_existing_files(selected_actions)

    board.prepare_session()
    board.start_stream()

    wait_seconds = 2
    print("Get ready in {} seconds".format(wait_seconds))
    time.sleep(wait_seconds)

    try:
        for action_idx in selected_actions:
            for session_idx in range(session_count):
                input(
                    "Ready to record action {} (session {}/{}). Press enter to continue".format(
                        action_idx, session_idx + 1, session_count
                    )
                )

                countdown = recording_delay
                while countdown > 0:
                    sound_helper.play_sound(u"sounds/boop.wav")
                    print(f"Recording in {countdown}... ", end="\r")
                    time.sleep(1)
                    countdown -= 1

                sound_helper.play_sound(u"sounds/start.wav")

                print("Think Action {} for {} seconds".format(
                    action_idx, window_length + window_buffer
                ))
                time.sleep(window_length + window_buffer)

                data = board.get_current_board_data(sampling_size)
                action_dict[action_idx].append(data)

                sound_helper.play_sound(u"sounds/done.wav")
                print(
                    "Recorded action {} session {}/{}".format(
                        action_idx, session_idx + 1, session_count
                    )
                )

        print("Saving Data")
        filename_target = get_next_available_filename()

        with open(filename_target, 'wb') as f:
            pickle.dump(record_data, f)

        print(f"Saved recording file: {filename_target}")
        print(f"Recorded action indices in this file: {sorted(action_dict.keys())}")

    finally:
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()